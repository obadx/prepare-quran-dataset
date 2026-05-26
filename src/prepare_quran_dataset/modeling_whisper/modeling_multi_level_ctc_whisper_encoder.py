from typing import Optional, Union

import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder,
    WhisperPreTrainedModel,
    _HIDDEN_STATES_START_POSITION,
)
from transformers.utils import auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from .configuration_for_multi_level_ctc import WhisperEncoderForMultilevelCTCConfig


class WhisperEncoderVariableLength(WhisperEncoder):
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_features,
        attention_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
                `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
                the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
        """

        expected_seq_length = (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length <= {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to be <= {expected_seq_length}."
            )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        seq_len = inputs_embeds.shape[-1]

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        # Instead of chosing all positional embedding chosing only that correspoding to the input length
        positions = torch.arange(seq_len, device=inputs_embeds.device)

        hidden_states = inputs_embeds + self.embed_positions(positions)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    None,
                    **kwargs,
                )

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class WhisperEncoderForMultilevelCTC(WhisperPreTrainedModel):
    config_class = WhisperEncoderForMultilevelCTCConfig

    def __init__(self, config):
        super().__init__(config)

        self.whisper_encoder = WhisperEncoderVariableLength(config)
        self.dropout = nn.Dropout(config.dropout)

        if config.level_to_vocab_size == {}:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `WhisperEncoderForMultilevelCTC.from_pretrained(..., level_to_vocab_size=level_to_vocab_size)`. "
                "or define `level_to_vocab_size` of your model's configuration."
            )
        self.level_to_lm_head = nn.ModuleDict(
            {
                level: nn.Linear(config.d_model, vocab_size)
                for level, vocab_size in config.level_to_vocab_size.items()
            }
        )

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[dict[str, torch.Tensor]] = None,
    ) -> Union[tuple, CausalLMOutput]:
        r"""
        labels (dict[`str`, `torch.LongTensor`] level_name to its labels of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError(
                    f"Label has to be a dict for level to its tartget labels got `{type(labels)}`"
                )
            for level in labels:
                if labels[level].max() >= self.config.level_to_vocab_size[level]:
                    raise ValueError(
                        f"Label values must be <= vocab_size: {self.config.level_to_vocab_size[level]} for level: `{level}`"
                    )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.whisper_encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        level_to_logits = {}
        for level in self.level_to_lm_head:
            level_to_logits[level] = self.level_to_lm_head[level](hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(
                    input_features.shape[:2],
                    device=input_features.device,
                    dtype=torch.long,
                )
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum([-1])
            ).to(torch.long)

            loss = 0.0
            for level in labels:
                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels[level] >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels[level].masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(
                    level_to_logits[level], dim=-1, dtype=torch.float32
                ).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss += self.config.level_to_loss_weight[
                        level
                    ] * nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )

        if not return_dict:
            output = (level_to_logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=level_to_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["WhisperEncoderForMultilevelCTC", "WhisperEncoderVariableLength"]
