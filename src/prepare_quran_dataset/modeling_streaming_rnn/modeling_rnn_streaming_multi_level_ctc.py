from typing import Optional, Union

from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
    Wav2Vec2BertPreTrainedModel,
    Wav2Vec2BertModel,
    _HIDDEN_STATES_START_POSITION,
)
from transformers.utils import auto_docstring
from transformers.modeling_outputs import CausalLMOutput
import torch
from torch import nn
import numpy as np

from .configuration_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCConfig,
)

# TODO:
# [ ] Add LSTM
# [ ] Add conifigs
# [ ] del samples < chunk


# TODO: To be added configs:
# max_chunk_batch= 1
# lookback_frames= 5
# chunk_frames= 25
# loopahead_frames= 5
# rnn_hidden_size = 128
# rnn_dropout = 0.1


def convert_input_to_chunked_for_offline(
    input: torch.Tensor,
    lookahead: int = 5,
    chunk: int = 25,
    lookback: int = 5,
) -> torch.Tensor:
    """Convert input from `(batch, seq_len, features)` to chunked format
    `(batch * num_chunks, lookback + chunk + lookahead, features)`.

    Each output chunk is assembled as `[lookback | chunk | lookahead]` with
    padding/overlap between consecutive chunks.

    The function accepts both 3D feature tensors `(batch, seq_len, features)` and
    2D tensors `(batch, seq_len)` — for example, an attention mask — which is
    unsqueezed internally, chunked identically, and squeezed back to 2D on return.

    Args:
        input (torch.Tensor):
            Input tensor of shape `(batch, seq_len)` or `(batch, seq_len, features)`.
        lookahead (int, optional):
            Number of frames after the chunk from the next segment. Defaults to `5`.
        chunk (int, optional):
            Number of frames per chunk. Defaults to `25`.
        lookback (int, optional):
            Number of frames before the chunk from the previous segment. Defaults to `5`.

    Returns:
        torch.Tensor: Chunked tensor of shape
        `(batch * num_chunks, lookback + chunk + lookahead, features)`.
        If the input was 2D, the feature dimension is squeezed away.

    Raises:
        ValueError:
            If `input` is not 2D or 3D.
        ValueError:
            If `seq_len < chunk`.

    Example:

        >>> import torch
        >>> input = torch.tensor([[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]])
        >>> out = convert_input_to_chunked_for_offline(input, lookback=2, chunk=5, lookahead=3)
        >>> out.shape
        torch.Size([3, 10, 1])
        >>> out
        tensor([[[ 0.],
                 [ 0.],
                 [ 1.],
                 [ 2.],
                 [ 3.],
                 [ 4.],
                 [ 5.],
                 [ 6.],
                 [ 7.],
                 [ 8.]],

                [[ 4.],
                 [ 5.],
                 [ 6.],
                 [ 7.],
                 [ 8.],
                 [ 9.],
                 [10.],
                 [11.],
                 [12.],
                 [ 0.]],

                [[ 9.],
                 [10.],
                 [11.],
                 [12.],
                 [ 0.],
                 [ 0.],
                 [ 0.],
                 [ 0.],
                 [ 0.],
                 [ 0.]]])

        2D input (e.g. attention mask):

        >>> mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        >>> out = convert_input_to_chunked_for_offline(mask, lookback=2, chunk=5, lookahead=3)
        >>> out.shape
        torch.Size([2, 10])
        >>> out
        tensor([[0., 0., 1., 1., 1., 1., 1., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    is_2d = False
    if len(input.shape) == 2:
        input = input.unsqueeze(dim=-1)
        is_2d = True
    elif len(input.shape) != 3:
        raise ValueError(
            f"Input tensor must be 2D (batch, seq_len) or 3D (batch, seq_len, features), "
            f"but got {len(input.shape)}D tensor with shape {input.shape}. "
            "If your input is 1D, use `input.unsqueeze(0).unsqueeze(-1)` to add batch and feature dimensions."
        )

    batch, seq_len, features = input.shape
    streaming_len = lookback + chunk + lookahead

    if seq_len < chunk:
        raise ValueError(
            f"Sequence length ({seq_len}) must be at least chunk size ({chunk}). "
            f"Got seq_len={seq_len} < chunk={chunk}. "
            "Reduce `chunk`, or pad/truncate the input sequence to meet this requirement."
        )

    # padd input so we can resahpe it
    last_chunk_len = seq_len % chunk
    if last_chunk_len > 0:
        pad_len = chunk - last_chunk_len
        padded_input = nn.functional.pad(input, (0, 0, 0, pad_len), value=0).reshape(
            batch, -1, chunk, features
        )
    else:
        padded_input = input.view(batch, -1, chunk, features)

    num_chunks = padded_input.shape[1]
    output = torch.zeros(
        (batch, num_chunks, streaming_len, features),
        device=input.device,
        dtype=padded_input.dtype,
    )

    output[:, :, lookback : lookback + chunk, :] = padded_input  # chunk
    output[:, 1:, :lookback, :] = padded_input[
        :, :-1, chunk - lookback :, :
    ]  # lookback
    output[:, :-1, lookback + chunk :, :] = padded_input[
        :, 1:, :lookahead, :
    ]  # lookahead

    output = output.view(-1, streaming_len, features)
    if is_2d:
        output = output.squeeze(dim=-1)
    return output


class Wav2Vec2BertForRNNStreamingMultilevelCTC(Wav2Vec2BertPreTrainedModel):
    config_class = Wav2Vec2BertForRNNStreamingMultilevelCTCConfig

    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.level_to_vocab_size == {}:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2BertForCTC.from_pretrained(..., level_to_vocab_size=level_to_vocab_size)`. "
                "or define `level_to_vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        self.level_to_lm_head = nn.ModuleDict(
            {
                level: nn.Linear(output_hidden_size, vocab_size)
                for level, vocab_size in config.level_to_vocab_size.items()
            }
        )

        self.rnn = nn.LSTM(
            input_size=config.output_hidden_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=config.runn_dropout,
            bidirectional=False,
        )

        self.straming_len = (
            config.chunk_frames + config.lookback_frames + config.lookahead_frames
        )

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stream_inference: bool = False,
        labels: Optional[dict[str, torch.Tensor]] = None,
        labels_masks: Optional[dict[str, torch.Tensor]] = None,
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
        # Reshping input from batch, seq_len, feature_size to batch, chunk_batch_size,k
        if stream_inference:
            if input_features.shape[1] != self.steaming_len:
                raise ValueError(
                    f"The sequence len must be of shape: `chunk_frames + lookback_frames + lookahead_frames` = `{self.straming_len}`, got: `{input_features.shape[1]}`"
                )
        else:
            batch, seq_len, features = input_features.shape
            if seq_len < self.straming_len:
                raise ValueError(
                    f"Sequence len have to be >= `chunk_frames` i.e <= `{self.config.chunk_frames}` but got: `{seq_len}`"
                )
            batched_input_features = convert_input_to_chunked_for_offline(
                input_features,
                chunk=self.config.chunk_frames,
                lookback=self.config.lookback_frames,
                lookahead=self.config.lookahead_frames,
            ).reshape(-1, self.config.max_chunk_batch, self.steaming_len, features)
            batched_attenion_mask = convert_input_to_chunked_for_offline(
                attention_mask,
                chunk=self.config.chunk_frames,
                lookback=self.config.lookback_frames,
                lookahead=self.config.lookahead_frames,
            ).reshape(-1, self.config.max_chunk_batch, self.steaming_len)

        outputs = self.wav2vec2_bert(
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


__all__ = ["Wav2Vec2BertForMultilevelCTC"]
