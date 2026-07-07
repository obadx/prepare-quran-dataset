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
        max_chunk_batch_size (int, optional):
            If > 1, extra zero-padded chunks are appended so the total number
            of chunks per batch is divisible by this value, enabling reshaping
            to `(-1, max_chunk_batch_size, ...)`. Must be `1` or a multiple of
            `batch`. Defaults to `1`.

    Returns:
        torch.Tensor: Chunked tensor of shape
        `(batch * num_chunks, lookback + chunk + lookahead, features)`.
        If the input was 2D, the feature dimension is squeezed away.

    Raises:
        ValueError:
            If `input` is not 2D or 3D.
        ValueError:
            If `seq_len < chunk`.
        ValueError:
            If `max_chunk_batch_size` is not `1` and not a multiple of `batch`.

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
    if batch % max_chunk_batch_size != 0 and max_chunk_batch_size % batch != 0:
        raise ValueError(
            f"`max_chunk_batch_size` has to be multiple of `batch_size`. Or `batch_size` is multiple of `max_chunk_batch_size`: You input `max_chunk_batch_size`={max_chunk_batch_size}, and the `batch_size` is {batch}"
        )
    streaming_len = lookback + chunk + lookahead

    if seq_len < chunk:
        raise ValueError(
            f"Sequence length ({seq_len}) must be at least chunk size ({chunk}). "
            f"Got seq_len={seq_len} < chunk={chunk}. "
            "Reduce `chunk`, or pad/truncate the input sequence to meet this requirement."
        )

    # Pad the last incomplete chunk so seq_len is divisible by chunk
    last_chunk_pad_len = 0
    last_chunk_len = seq_len % chunk
    if last_chunk_len > 0:
        last_chunk_pad_len = chunk - last_chunk_len
    padded_input = input
    if last_chunk_pad_len > 0:
        padded_input = nn.functional.pad(input, (0, 0, 0, last_chunk_pad_len), value=0)
    padded_input = padded_input.view(batch, -1, chunk, features)

    num_chunks = (seq_len + last_chunk_pad_len) // chunk  # num_chunks for every batch

    # Compute padded_num_chunks so the total chunks can be partitioned into
    # (-1, max_chunk_batch_size, streaming_len, features)
    # NOTE: num_chunks is different from padded_num_chunks; the latter may be padded
    # with sparse zero-chunks to accommodate max_chunk_batch_size
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
    """Wav2Vec2-BERT model with a streaming RNN and multi-level CTC heads.

    This model splits long audio sequences into overlapping chunks
    (lookback / chunk / lookahead), runs each chunk through the Wav2Vec2-BERT
    encoder, and uses a unidirectional LSTM to condition each chunk on the
    temporal history by taking the first frame only as summary for every chunk.
    A separate linear CTC head is applied per output level. the CTC lenaer heads
    input is: [chunk | lstm output]
    (e.g. phonemes, hams_or_jahr, …).

    During streaming inference (`stream_inference=True`) the caller supplies
    exactly one chunk at a time together with the previous LSTM state, enabling
    online / low-latency operation.
    """

    config_class = Wav2Vec2BertForRNNStreamingMultilevelCTCConfig

    def __init__(self, config):
        """Initialise the streaming RNN + multi-level CTC model.

        Args:
            config (`Wav2Vec2BertForRNNStreamingMultilevelCTCConfig`):
                Model configuration, including chunking parameters, RNN hidden size,
                and the per-level vocabulary sizes.
        """
        super().__init__(config)

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.ssl_dropout = nn.Dropout(config.final_dropout)
        self.rnn = nn.LSTM(
            input_size=config.output_hidden_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=config.rnn_dropout,
            bidirectional=False,
        )
        # as we have a single layer LSTM so the internal dropout is not applied
        self.rnn_dropout = nn.Dropout(config.rnn_dropout)

        if config.level_to_vocab_size == {}:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the CTC head. Please "
                "instantiate the model as follows: `Wav2Vec2BertForCTC.from_pretrained(..., level_to_vocab_size=level_to_vocab_size)`. "
                "or define `level_to_vocab_size` of your model's configuration."
            )
        # WARN: Ignoring Adapter logic for now
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

        self.streaming_len = (
            config.chunk_frames + config.lookback_frames + config.lookahead_frames
        )

        # Initialize weights and apply final processing
        self.post_init()

    # def initialize_weights(self):
    #     """
    #     Initialize all weights, then re-initialize nn.Embedding modules that
    #     HuggingFace's smart_apply skips inside child PreTrainedModel submodules.
    #
    #     The parent Wav2Vec2BertPreTrainedModel._init_weights does not handle
    #     nn.Embedding, so distance_embedding (nn.Embedding(73, head_size)) in each
    #     attention layer is left with PyTorch's default uniform(-1, 1) init.
    #
    #     When loaded with ignore_mismatched_sizes=True, the checkpoint's
    #     distance_embedding.weight shape [73, 64] does not match the model's
    #     [73, head_size], so the weight is not overwritten and the uniform init
    #     persists.  Vectors with norm ≈ √(head_size/3) ≈ 3.3 for head_size=32
    #     cause the relative position dot product (einsum Q·pos_emb) to exceed
    #     float32 range, producing NaN.
    #
    #     This override is called both during post_init() (model construction) and
    #     after checkpoint loading by _initialize_missing_keys, ensuring it catches
    #     the two-pass HuggingFace weight init pattern.
    #     The debugging was done using `tests/debug_explosion_modeling_streaming.py`
    #     """
    #     super().initialize_weights()
    #     for module in self.modules():
    #         if isinstance(module, nn.Embedding):
    #             nn.init.normal_(
    #                 module.weight, mean=0.0, std=self.config.initializer_range
    #             )

    def rnn_forward(
        self,
        rnn_input: torch.FloatTensor,  # (batch, num_chunks, input_size)
        rnn_history: tuple[
            torch.FloatTensor, torch.FloatTensor
        ],  # (h0, c0) each (num_layers, batch, hidden)
        num_chunks_mask: torch.Tensor,  # (batch, num_chunks) – 1 for real chunk, 0 for padding
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Run the LSTM over the chunk dimension, skipping padded chunks.

        Args:
            rnn_input: first frame of each chunk, shape (batch, num_chunks, input_size)
            rnn_history: initial hidden/cell states (h0, c0)
            num_chunks_mask: binary mask, shape (batch, num_chunks). 1 = real chunk, 0 = padding.

        Returns:
            rnn_output: LSTM output, same shape as rnn_input (padded positions are 0)
            (h_n, c_n): final hidden/cell states
        """
        # 1. Sequence lengths = number of real chunks per example
        lengths = num_chunks_mask.sum(dim=1)  # (batch,)

        # 2. Sort lengths in descending order (required by enforce_sorted=True)
        sorted_lengths, sort_indices = lengths.sort(descending=True)
        # Inverse permutation to restore order later
        _, unsort_indices = sort_indices.sort()

        # 3. Reorder the input and the hidden states to match the sorted batch
        sorted_input = rnn_input[sort_indices]  # (batch, num_chunks, input_size)
        # h0, c0 have shape (num_layers, batch, hidden); we sort along dim=1
        sorted_h0 = rnn_history[0][:, sort_indices, :]
        sorted_c0 = rnn_history[1][:, sort_indices, :]

        # 4. Pack sorted sequences (enforce_sorted=True is now safe) and for onnx export later
        packed_input = nn.utils.rnn.pack_padded_sequence(
            sorted_input,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )

        # 5. Run LSTM
        packed_output, (h_n_sorted, c_n_sorted) = self.rnn(
            packed_input, (sorted_h0, sorted_c0)
        )

        # 6. Unpack to a padded tensor of original shape
        rnn_output_sorted, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=rnn_input.size(1),  # number of chunks (including padding)
        )

        # 7. Restore original batch order
        rnn_output = rnn_output_sorted[unsort_indices]
        h_n = h_n_sorted[:, unsort_indices, :]
        c_n = c_n_sorted[:, unsort_indices, :]

        return rnn_output, (h_n, c_n)

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
        input_features (`torch.Tensor` of shape `(batch_size, seq_len, feature_size)`):
            Input features to the Wav2Vec2-BERT model.
            If `stream_inference` is `True`,
            the input is of shape (batch_size, (lookback + chunk + lookahead), feature_size)
            `lookback + chunk + lookahead`, and no chunking is applied.
        attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Attention mask indicating padded positions with 0.
            If `stream_inference` is `True`,
            the input is of shape (batch_size, (lookback + chunk + lookahead))
        rnn_history (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple of `(h_0, c_0)` LSTM states, each of shape `(1, batch_size, config.rnn_hidden_size)`.
            If `None`, zero states are initialised internally.
        output_attentions (`bool`, *optional*):
            Whether to return attention weights of the SSL encoder.
        output_hidden_states (`bool`, *optional*):
            Whether to return hidden states of the SSL encoder.
        return_dict (`bool`, *optional*):
            Whether to return a [`StreamingCTCOutput`] instead of a tuple.
        stream_inference (`bool`, *optional*, defaults to `False`):
            If `True`, the input is treated as a single streaming chunk of length
            `lookback + chunk + lookahead`, and no chunking is applied.
        labels (`dict[str, torch.LongTensor]`, *optional*):
            Dictionary mapping level names to their CTC targets of shape
            `(batch_size, target_length)`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            `-100` is ignored (masked), the loss is only computed for labels in `[0, ..., vocab_size - 1]`.
        labels_mask (`dict[str, torch.Tensor]`, *optional*):
            Dictionary mapping level names to their target masks of shape
            `(batch_size, target_length)`. 1 indicates a valid target, 0 indicates padding.

        Returns:
            `Union[tuple, StreamingCTCOutput]`:
                When `return_dict=False`, a tuple `(loss, logits, rnn_history, hidden_states)`.
                When `return_dict=True`, a [`StreamingCTCOutput`] with typed fields.
        """
        # Level to Labels validation
        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError(
                    f"Label has to be a dict for level to its target labels got `{type(labels)}`"
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

        if self.config.max_chunk_batch == 0 or (
            batch_size % self.config.max_chunk_batch != 0
            and self.config.max_chunk_batch % batch_size != 0
        ):
            raise ValueError(
                f"`config.max_chunk_batch` has to be multiple of `batch_size`. Or `batch_size` is multiple of `config.max_chunk_batch`: You input `config.max_chunk_batch`={self.config.max_chunk_batch}, and the `batch_size` is {batch_size}"
            )

        if labels is not None:
            if labels_mask is None:
                raise ValueError(
                    "`labels_mask` has to be input to calculate CTC loss properly with a dict of tensors of shape of `batch_size, target_seq_len`"
                )
            else:
                for level in labels:
                    if labels_mask[level].shape != labels[level].shape:
                        raise ValueError(
                            f"`labels_mask` has to be input to calculate CTC loss properly with shape of `batch_size, target_seq_len` ({labels[level].shape}) for level: `{level}` got: `{labels_mask[level].shape}`"
                        )

        # LSTM history, (c0, h0) validation
        valid_rnn_history_shape = (1, batch_size, self.config.rnn_hidden_size)
        if rnn_history is not None:
            rnn_history_err_msg = f"Invalid input shape for `rnn_history`. `rnn_history` has of 2d tuple of float tensor of shape: {valid_rnn_history_shape}"
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
                torch.zeros(
                    valid_rnn_history_shape,
                    device=input_features.device,
                    dtype=input_features.dtype,
                ),
                torch.zeros(
                    valid_rnn_history_shape,
                    device=input_features.device,
                    dtype=input_features.dtype,
                ),
            )

        # Reshaping input from (batch_size, seq_len), feature_size to (batch_size, num_chunks, streaming_len, feature_size)
        # For attention_mask the output shape: (batch_size, num_chunks, streaming_len)
        if stream_inference:
            if seq_len != self.streaming_len:
                raise ValueError(
                    f"The sequence len must be of shape: `chunk_frames + lookback_frames + lookahead_frames` = `{self.streaming_len}`, got: `{seq_len}`"
                )
            # Expanding dimension for the number of chunks
            batched_input_features = input_features.unsqueeze(dim=1)
            batched_attention_mask = attention_mask.unsqueeze(dim=1)
        else:
            if seq_len < self.config.chunk_frames:
                raise ValueError(
                    f"Sequence len have to be >= `chunk_frames` which is `{self.config.chunk_frames}` but got: `{seq_len}`"
                )
            batched_input_features = convert_input_to_chunked_for_offline(
                input_features,
                chunk=self.config.chunk_frames,
                lookback=self.config.lookback_frames,
                lookahead=self.config.lookahead_frames,
                max_chunk_batch_size=self.config.max_chunk_batch,
            ).reshape(-1, self.config.max_chunk_batch, self.streaming_len, features)
            batched_attention_mask = convert_input_to_chunked_for_offline(
                attention_mask,
                chunk=self.config.chunk_frames,
                lookback=self.config.lookback_frames,
                lookahead=self.config.lookahead_frames,
                max_chunk_batch_size=self.config.max_chunk_batch,
            ).reshape(-1, self.config.max_chunk_batch, self.streaming_len)

        # Enabling larger effective batch size on small GPU by controlling our own gpu max_chunk_batch
        # Utilizing the fact that we have multiple small seq_lens each of `self.streaming_len` so we are
        # introducing self.config.max_chunk_batch to make the most use of GPU and sequentially inference over
        # the num_small_batches enabling training on relatively larger batch_size

        # Initializing hidden_states for future concatenation with the rnn output
        # hidden states are the concatenation of the ssl output and the rnn outputs

        hidden_states = torch.zeros(
            (
                batched_input_features.shape[0],
                self.config.max_chunk_batch,
                self.streaming_len,
                self.config.hidden_size + self.config.rnn_hidden_size,
            ),
            device=batched_input_features.device,
            dtype=batched_input_features.dtype,
        )
        for idx, (small_input_features, small_attention_mask) in enumerate(
            zip(batched_input_features, batched_attention_mask)
        ):
            hidden_states[idx, :, :, : self.config.hidden_size] = self.wav2vec2_bert(
                small_input_features,
                attention_mask=small_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]

        # Back to the same original input shape as original_batch_size, num_of_chunks, self.streaming_len, hidden_size)
        hidden_states = hidden_states.view(
            batch_size,
            -1,
            self.streaming_len,
            self.config.hidden_size + self.config.rnn_hidden_size,
        )
        batched_attention_mask = batched_attention_mask.view(
            batch_size, -1, self.streaming_len
        )
        hidden_states = self.ssl_dropout(hidden_states)

        # RNN inference loop over the chunk dimension (dim=1)
        # Computing masks for chunks; not all chunks have valid input, some are all-zero padding
        num_chunks_mask = (batched_attention_mask.sum(dim=-1) != 0).to(torch.long)
        rnn_output, rnn_history = self.rnn_forward(
            hidden_states[
                :, :, 0, : self.config.hidden_size
            ],  # taking the first token(0) to the rnn as feedback
            rnn_history,
            num_chunks_mask=num_chunks_mask,
        )
        rnn_output = self.rnn_dropout(rnn_output)

        # Keeping only the chunk and drop the lookback and lookahead
        hidden_states = hidden_states[
            :,
            :,
            self.config.lookback_frames : self.config.lookback_frames
            + self.config.chunk_frames,
            :,
        ]
        batched_attention_mask = batched_attention_mask[
            :,
            :,
            self.config.lookback_frames : self.config.lookback_frames
            + self.config.chunk_frames,
        ]

        # NOTE: we are concatenating the lstm output with every hidden_state
        # Concatenating the output of the rnn of shape (batch_size, num_chunks, rnn_hidden_size) to the hidden states
        hidden_states[:, :, :, self.config.hidden_size :] = rnn_output.unsqueeze(dim=2)

        # Flattening hidden sizes as expected without streaming
        hidden_states = hidden_states.reshape(
            batch_size, -1, self.config.hidden_size + self.config.rnn_hidden_size
        )
        attention_mask = batched_attention_mask.reshape(batch_size, -1)

        # Applying CTC Linear Heads for every level
        level_to_logits = {}
        for level in self.level_to_lm_head:
            level_to_logits[level] = self.level_to_lm_head[level](hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum([-1])
            ).to(torch.long)

            loss = 0.0
            for level in labels:
                target_lengths = labels_mask[level].sum(-1)
                flattened_targets = labels[level].masked_select(
                    labels_mask[level].to(torch.bool)
                )

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
            # NOTE: Skipping attention return for now
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


__all__ = ["Wav2Vec2BertForRNNStreamingMultilevelCTC"]
