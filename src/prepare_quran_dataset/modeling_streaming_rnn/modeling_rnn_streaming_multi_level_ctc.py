from typing import Optional, Union
from dataclasses import dataclass
from math import ceil

from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
    Wav2Vec2BertPreTrainedModel,
    Wav2Vec2BertModel,
    _HIDDEN_STATES_START_POSITION,
)
from transformers.utils import auto_docstring, ModelOutput
from transformers.modeling_outputs import CausalLMOutput
import torch
from torch import nn

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
# lookahead_frames= 5
# rnn_hidden_size = 128
# rnn_dropout = 0.1


def convert_input_to_chunked_for_offline(
    input: torch.Tensor,
    lookahead: int = 5,
    chunk: int = 25,
    lookback: int = 5,
    max_chunk_batch_size=1,
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
    if max_chunk_batch_size != 1 and max_chunk_batch_size % batch != 0:
        raise ValueError(
            f"`max_chunk_batch_size` has either to be 1 or be multiple of `batch_size`: You input `max_chunk_batch_size`={max_chunk_batch_size}, and the `batch_size` is {batch}"
        )
    streaming_len = lookback + chunk + lookahead

    if seq_len < chunk:
        raise ValueError(
            f"Sequence length ({seq_len}) must be at least chunk size ({chunk}). "
            f"Got seq_len={seq_len} < chunk={chunk}. "
            "Reduce `chunk`, or pad/truncate the input sequence to meet this requirement."
        )

    # Computing the padding to the  last input chunk such that every chunk has length of `chunk + lookback + lookahead`
    last_chunk_pad_len = 0
    last_chunk_len = seq_len % chunk
    if last_chunk_len > 0:
        last_chunk_pad_len = chunk - last_chunk_len

    # WARN: Needs rethinking about padding + concatenation we can get rid of padding I think
    padded_input = input
    if last_chunk_pad_len > 0:
        padded_input = nn.functional.pad(input, (0, 0, 0, last_chunk_pad_len), value=0)
    padded_input = padded_input.view(batch, -1, chunk, features)

    num_chunks = (seq_len + last_chunk_pad_len) // chunk  # num_chunks for every batch

    # Computing num_padded_chunks for max_chunk_batch_size such that: we can partiion the input to
    # (-1, max_chunk_batch_size, streaming_len, features
    # NOTE: num_chunks is diffrent from padded_num_chunks as the later may be padded with sparse chunks
    # to acomidiate max_chunk_batch_size
    padded_num_chunks = (
        ceil(num_chunks * batch / max_chunk_batch_size) * max_chunk_batch_size // batch
    )

    output = torch.zeros(
        (batch, padded_num_chunks, streaming_len, features),
        device=input.device,
        dtype=padded_input.dtype,
    )

    # chunk
    output[:, :num_chunks, lookback : lookback + chunk, :] = padded_input
    # lookback
    output[:, 1:num_chunks, :lookback, :] = padded_input[
        :, : num_chunks - 1, chunk - lookback :, :
    ]
    # lookahead
    output[:, : num_chunks - 1, lookback + chunk :, :] = padded_input[
        :, 1:num_chunks, :lookahead, :
    ]

    output = output.view(-1, streaming_len, features)
    if is_2d:
        output = output.squeeze(dim=-1)
    return output


@dataclass
class StreamingCTCOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    rnn_history: tuple[torch.FloatTensor, torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class Wav2Vec2BertForRNNStreamingMultilevelCTC(Wav2Vec2BertPreTrainedModel):
    config_class = Wav2Vec2BertForRNNStreamingMultilevelCTCConfig

    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.ssl_dropout = nn.Dropout(config.final_dropout)

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
                level: nn.Linear(
                    output_hidden_size + config.rnn_hidden_size, vocab_size
                )
                for level, vocab_size in config.level_to_vocab_size.items()
            }
        )

        self.rnn = nn.LSTM(
            input_size=config.output_hidden_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=config.rnn_dropout,
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
        rnn_history: Optional[tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stream_inference: bool = False,
        labels: Optional[dict[str, torch.Tensor]] = None,
        labels_mask: Optional[dict[str, torch.Tensor]] = None,
    ) -> Union[tuple, CausalLMOutput]:
        r"""
        labels (dict[`str`, `torch.LongTensor`] level_name to its labels of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # Level to Labels validation
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
        batch_size, seq_len, features = input_features.shape

        # LSTM history, (c0, h0) validation
        valid_rnn_history_shape = (1, batch_size, self.config.rnn_hidden_size)
        if rnn_history is not None:
            rnn_history_err_msg = f"Invlaid input shape for `rnn_history`. `rnn_history` has of 2d tuple of float tensor of shape: {valid_rnn_history_shape}"
            if len(rnn_history) != 2:
                raise ValueError(rnn_history_err_msg)
            else:
                if (
                    rnn_history[0].shape != valid_rnn_history_shape
                    or rnn_history[1].shape != valid_rnn_history_shape
                ):
                    raise ValueError(rnn_history_err_msg)
        else:
            rnn_history = (
                torch.zeros(valid_rnn_history_shape, device=input_features.device),
                torch.zeros(valid_rnn_history_shape, device=input_features.device),
            )

        # Reshping input from batch, seq_len, feature_size to batch, chunk_batch_size,k
        if stream_inference:
            if seq_len != self.steaming_len:
                raise ValueError(
                    f"The sequence len must be of shape: `chunk_frames + lookback_frames + lookahead_frames` = `{self.straming_len}`, got: `{seq_len}`"
                )
            batched_input_features = input_features.unsqueeze(dim=0)
            batched_attenion_mask = attention_mask.unsqueeze(dim=0)
        else:
            if seq_len < self.config.chunk_frames:
                raise ValueError(
                    f"Sequence len have to be >= `chunk_frames` i.e <= `{self.config.chunk_frames}` but got: `{seq_len}`"
                )
            batched_input_features = convert_input_to_chunked_for_offline(
                input_features,
                chunk=self.config.chunk_frames,
                lookback=self.config.lookback_frames,
                lookahead=self.config.lookahead_frames,
                max_chunk_batch_size=self.config.max_chunk_batch,
            ).reshape(-1, self.config.max_chunk_batch, self.steaming_len, features)
            batched_attenion_mask = convert_input_to_chunked_for_offline(
                attention_mask,
                chunk=self.config.chunk_frames,
                lookback=self.config.lookback_frames,
                lookahead=self.config.lookahead_frames,
                max_chunk_batch_size=self.config.max_chunk_batch,
            ).reshape(-1, self.config.max_chunk_batch, self.steaming_len)

        # Enabling larger EFFICTIVE batch size on small GPU by controling our own gpu max_chunk_batch
        # Utlizing the fact that we we have multiple small seq_lens each of `self.streaming_len` so we are
        # introducing a self.config.max_chunk_batch_size to making the most use of GPU and sequantilly inference over
        # the num_small_batches enabling training on relativly larger batch_size
        num_small_batches, small_batch_size, padded_seq_len = (
            batched_input_features.shape[0:3]
        )

        # Intializating hidden_states for future concatenation with the rnn output
        # hidden sates are the concatenation of the ssl output and the rnn outputs
        hidden_states = torch.zeros(
            (
                num_small_batches,
                small_batch_size,
                padded_seq_len,
                self.config.hidden_size + self.config.rnn_hidden_size,
            ),
            device=batched_input_features.device,
            dtype=batched_input_features.dtype,
        )
        for idx, (small_input_featrues, small_attention_mask) in enumerate(
            zip(batched_input_features, batched_attenion_mask)
        ):
            hidden_states[idx, :, :, : self.config.hidden_size] = self.wav2vec2_bert(
                small_input_featrues,
                attention_mask=small_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]

        # Back to the same original input shape as original_batch_size, num_of_chunks, padded_seq_len, hidden_size)
        hidden_states = hidden_states.view(
            batch_size, -1, padded_seq_len, self.config.hidden_size
        )
        batched_attenion_mask = batched_attenion_mask.view(
            batch_size, -1, padded_seq_len
        )
        # TODO: How to apply dorpout
        hidden_states = self.ssl_dropout(hidden_states)

        # RNN  Inference Looping for the chunks (dim=1) we have for every input
        # BUG: chunk_mask
        rnn_output, rnn_history = self.rnn(
            hidden_states[
                :, :, 0, : self.config.hidden_size
            ],  # taking the firt token(0) to to rnn as feedback
            rnn_history,
        )
        # Keeping only the chunk and drop the lookaback and lookahead
        hidden_states = hidden_states[
            :,
            :,
            :,
            self.config.lookback_frames : self.config.lookback_frames
            + self.config.chunk_frames,
        ]
        batched_attenion_mask = batched_attenion_mask[
            :,
            :,
            self.config.lookback_frames : self.config.lookback_frames
            + self.config.chunk_frames,
        ]

        # NOTE: that we are concatenating the lstm output with every hidden_sate
        # Concatenation the output for the rnn of shape (batch_size, num_chunks, rnn_hidden_size) to the hidden states
        hidden_states[:, :, :, self.config.hidden_size :] = rnn_output.unsqueeze(dim=2)

        # Flatting hidden sizes as exepcted without streaming
        hidden_states = hidden_states.view(
            batch_size, -1, self.config.hidden_size + self.config.rnn_hidden_size
        )
        attention_mask = batched_attenion_mask.view(batch_size, -1)

        # Appling CTC Linear Heads for every level
        level_to_logits = {}
        for level in self.level_to_lm_head:
            level_to_logits[level] = self.level_to_lm_head[level](hidden_states)

        loss = None
        if labels is not None:
            if labels_mask is None:
                raise ValueError(
                    "`labels_masks` has to be input to calculate CTC loss properly with a dict of tensors of shape of `batch_size, target_seq_len`"
                )
            else:
                for level in labels:
                    if labels_mask[level].shape != labels[level].shape:
                        raise ValueError(
                            f"`labels_masks` has to be input to calculate CTC loss properly with shape of `batch_size, target_seq_len` ({labels[level].shape}) for level: `{level}` got: `{labels_mask[level].shape}`"
                        )

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
                target_lengths = labels_mask[level].sum(-1)
                flattened_targets = labels[level].masked_select(labels_mask[level])

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
            # NOTE: Skipping returning attentions right now
            # output = (level_to_logits, rnn_history) + outputs[
            #     _HIDDEN_STATES_START_POSITION:
            # ]
            output = (level_to_logits, rnn_history, hidden_states)
            return ((loss,) + output) if loss is not None else output

        return StreamingCTCOutput(
            loss=loss,
            logits=level_to_logits,
            rnn_history=rnn_history,
            hidden_states=hidden_states,
            attentions=None,  # Make it `None` for now
        )


__all__ = ["Wav2Vec2BertForMultilevelCTC"]
