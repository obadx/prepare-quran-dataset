"""FastConformer cache-aware model with multi-level CTC heads.

Architecture overview::

    Raw audio (PCM)
        │
        ▼
    FastConformerMelProcessor .......... (mel-spectrogram extraction)
        │
        ▼
    ConformerEncoder ................... (NeMo, chunked_limited attention mask)
        │
        ▼
    per-level Linear CTC heads ........ (phonemes, sifat attributes, ...)
        │
        ▼
    logits  →  CTC loss (labels + labels_mask)

During **training** the full audio sequence is processed in a single forward
pass with a ``chunked_limited`` attention mask.  No cross-chunk caching is
used — set ``cache=None`` (the default).

During **streaming inference** pass a :class:`FastConformerCache` to enable
chunk-by-chunk processing with cache reuse, using NeMo's
``forward_internal()`` + ``streaming_post_process()`` two-stage pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Union

import torch
from nemo.collections.asr.modules import ConformerEncoder
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTCConfig,
)
from .processor import FastConformerMelProcessor


class MuaalemConformerEncoder(ConformerEncoder):
    def _create_masks(
        self,
        att_context_size: list[int],
        padding_length: torch.LongTensor,
        max_audio_length: int,
        offset: torch.LongTensor | None,
        device: torch.device,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor | None]:
        """Create the self-attention mask and padding mask.

        The attention mask applies the configured context style (``regular``
        or ``chunked_limited``) with the given left/right context sizes.

        The padding mask (``pad_mask``) marks which frames are **invalid**
        and must be excluded from *all* encoder sub-layers — both
        self-attention and convolution.  Two conditions produce masked
        positions:

        1. ``idx >= padding_length`` — frames beyond the valid signal
           length (standard sequence padding).
        2. ``idx < offset`` (only during streaming with cache) — zero-
           padded cache slots that contain no real data from a prior step.

        ``pad_mask`` is consumed in two ways:

        *   **Attention** — expanded to a 2D matrix
            (``pad_mask_2d[i,j] = pad_mask[i] OR pad_mask[j]``),
            then OR-ed into ``att_mask`` so padded positions are blocked
            as both queries and keys.
        *   **Convolution** — passed directly to each Conformer block's
            ``ConvLayer`` so the causal convolution state is only updated
            from valid (non-padding) frames, preventing boundary artifacts
            at chunk seams.

        Args:
            att_context_size:
                Left and right attention context as ``[left, right]`` in
                encoder frames.  ``-1`` means unlimited.
            padding_length:
                Number of valid frames per sample.
                Shape ``(batch_size,)``.
            max_audio_length:
                Maximum sequence length (padded), in frames.
            offset:
                Per-sample frame offset used during streaming with cache.
                Frames at ``idx < offset`` are zero-padded cache slots
                treated as padding in ``pad_mask``.  ``None`` in offline
                mode.  Shape ``(batch_size,)``.
            device:
                Target device for the returned tensors.

        Returns:
            ``(pad_mask, att_mask)`` — the padding mask has shape
            ``(batch_size, max_audio_length)`` with ``True`` = padded /
            masked-out and ``False`` = valid signal (allowed).

            The attention mask has shape ``(batch_size, max_audio_length,
            max_audio_length)`` where ``att_mask[i, j]`` is ``True`` =
            query position **i** cannot attend to key position **j**
            (blocked by padding or outside the context window) and
            ``False`` = allowed.  In row-major terms: row **i** lists
            which key positions are masked for query **i**; column **j**
            lists which queries cannot attend to key **j**.

            Returns ``None`` when using ``rel_pos_local_attn`` (the
            local-attention mechanism handles its own masking).

            .. note::
                ``True`` = **masked** in both masks (position is ignored
                in the attention computation).  This follows PyTorch's
                ``scaled_dot_product_attention`` / ``nn.functional.softmax``
                convention where ``True`` entries contribute ``-inf`` to
                the softmax, producing zero attention weight.

        Example — ``att_context_size=[6, 2]``, ``chunked_limited``::

            >>> pad_mask, att_mask = encoder._create_masks(
            ...     att_context_size=[6, 2],
            ...     padding_length=torch.tensor([20]),
            ...     max_audio_length=20,
            ...     offset=torch.tensor([3]),
            ...     device='cpu',
            ... )
            >>> pad_mask[0]
            tensor([ True,  True,  True, False, False, False, ...])

            Frame layout::

                idx range     content
                ─────────────────────────────────────────
                [0, offset)   zero-padded cache slots (masked)
                [offset, cache_len)    valid cached frames
                [cache_len, padding_length)  current input chunk
                [padding_length, max_audio_length)  beyond valid signal (masked)

            ``offset=3`` means frames 0-2 are zero-padded cache slots and are
            masked in both ``pad_mask`` and ``att_mask``.
            ``padding_length=20`` means all 20 frames are within the valid
            range for this example, so only the offset region is masked.

            With ``chunked_limited [6,2]`` the attention is further restricted:
            ``chunk_size = 2 + 1 = 3`` and ``left_chunks = 6 // 3 = 2``.
            Each chunk (3 frames) can attend to itself and the 2 preceding
            chunks (6 frames), so row 3 attends frames 0-5 (its chunk plus
            left context), but row 6 cannot attend frame 0 because that is
            2 chunks away and exceeds ``left_chunks=2``.

            Remember: ``True`` = masked (``X``), ``False`` = allowed (``.``)
            — row **i** shows what query **i** can and cannot see.
        """
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(
                1, max_audio_length, max_audio_length, dtype=torch.bool, device=device
            )

            if self.att_context_style == "regular":
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
                if att_context_size[1] >= 0:
                    att_mask = att_mask.tril(diagonal=att_context_size[1])
            elif self.att_context_style == "chunked_limited":
                # When right context is unlimited, just the left side of the masking need to get updated
                if att_context_size[1] == -1:
                    if att_context_size[0] >= 0:
                        att_mask = att_mask.triu(diagonal=-att_context_size[0])
                else:
                    chunk_size = att_context_size[1] + 1
                    # left_chunks_num specifies the number of chunks to be visible by each chunk on the left side
                    if att_context_size[0] >= 0:
                        left_chunks_num = att_context_size[0] // chunk_size
                    else:
                        left_chunks_num = 10000

                    chunk_idx = torch.arange(
                        0, max_audio_length, dtype=torch.int, device=att_mask.device
                    )
                    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                    chunked_limited_mask = torch.logical_and(
                        torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                    )
                    att_mask = torch.logical_and(
                        att_mask, chunked_limited_mask.unsqueeze(0)
                    )
        else:
            att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat(
                [1, max_audio_length, 1]
            )
            pad_mask_for_att_mask = torch.logical_and(
                pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2)
            )
            # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
            att_mask = torch.logical_and(
                pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device)
            )
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def forward_internal(
        self,
        audio_signal: torch.FloatTensor,
        length: torch.LongTensor | None,
        cache_last_channel: torch.FloatTensor | None = None,
        cache_last_time: torch.FloatTensor | None = None,
        cache_last_channel_len: torch.LongTensor | None = None,
    ) -> (
        tuple[torch.FloatTensor, torch.LongTensor]
        | tuple[
            torch.FloatTensor,
            torch.LongTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.LongTensor,
        ]
    ):
        """Run the encoder on (possibly chunked) audio with optional cache.

        When ``cache_last_channel`` is ``None``, runs the full encoder as a
        single forward pass — identical to the parent
        :class:`~nemo.collections.asr.modules.ConformerEncoder`.

        When a cache is provided, uses the caching-aware path:

        1.  Pre-encode (subsampling) — drops ``drop_extra_pre_encoded``
            frames from the start.
        2.  Prepends ``cache_len`` left-context frames for self-attention.
        3.  Loops through Conformer layers, reading/writing per-layer cache.
        4.  Applies final projection and reduction subsampling.
        5.  Returns updated cache tensors for the next streaming step.

        .. note::
            This method is called by the parent class's
            :meth:`streaming_post_process` which truncates the output to
            ``valid_out_len`` frames.

        Args:
            audio_signal:
                Mel-spectrogram features.
                Shape ``(batch_size, num_mel_bins, num_frames)``.
            length:
                Number of valid frames per sample.
                Shape ``(batch_size,)``.  If ``None``, inferred from
                ``audio_signal.size(-1)``.
            cache_last_channel:
                Left-context cache for self-attention layers (one entry per
                encoder layer).  Shape ``(n_layers, batch_size, cache_size,
                d_model)``.  ``None`` for offline (cache-free) mode.
            cache_last_time:
                Left-context cache for convolution layers.
                Shape ``(n_layers, batch_size, d_model, cache_size)``.
                ``None`` for offline mode.
            cache_last_channel_len:
                Number of valid cached frames per sample.
                Shape ``(batch_size,)``.  ``None`` for offline mode.

        Returns:
            *   **Offline mode** (``cache_last_channel`` is ``None``):
                ``(audio_signal, length)`` — encoder output with shape
                ``(batch_size, d_model, num_encoder_frames)`` and valid
                lengths ``(batch_size,)``.
            *   **Streaming mode** (cache provided):
                ``(audio_signal, length, cache_ch_next, cache_t_next,
                cache_len_next)`` — the first two elements are the encoder
                output for this chunk (including lookahead); the last three
                are the updated cache state for the next streaming step.

        Example — streaming step with 3-of-6 cache slots valid::

            >>> cache = model.get_initial_cache(batch_size=1)
            >>> cache.last_channel_len[:] = 3   # 3 valid, 3 zero-padded
            >>> audio = torch.randn(1, 80, 50)
            >>> length = torch.tensor([50])
            >>> out = encoder.forward_internal(audio, length,
            ...     cache_last_channel=cache.last_channel,
            ...     cache_last_time=cache.last_time,
            ...     cache_last_channel_len=cache.last_channel_len)

            Inside the method::

                # pre-encode: (1,80,50) → (1,14,64), length=[14]
                # cache_len=6 → max_audio_len=14+6=20, padding_length=[20]
                # offset = -3 + 6 = 3  → frames 0..2 are zero-padded cache
                # _create_masks masks frames 0..2 and applies chunked_limited
                # attention with left=6 / right=2 on frames 3..19
                # Returns: (1,64,T'), (1,), + updated cache tensors
        """
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(
                self.att_context_size_all, weights=self.att_context_probs
            )[0]
        else:
            cur_att_context_size = self.att_context_size

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            # self.streaming_cfg is set by setup_streaming_cfg(), called in the init
            if (
                self.streaming_cfg.drop_extra_pre_encoded > 0
                and cache_last_channel is not None
            ):
                audio_signal = audio_signal[
                    :, self.streaming_cfg.drop_extra_pre_encoded :, :
                ]
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(
                    min=0
                )

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(
            zip(self.layer_drop_probs, self.layers)
        ):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = (
                    audio_signal
                )
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (
                        1.0 - drop_prob
                    ) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(
                    x=audio_signal, lengths=length
                )
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            # saving tensors if required for interctc loss
            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get(
                        "interctc", {}
                    ).get("capture_layers", [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                    self.register_accessible_tensor(
                        name=f"interctc/layer_output_{lth}",
                        tensor=torch.transpose(lth_audio_signal, 1, 2),
                    )
                    self.register_accessible_tensor(
                        name=f"interctc/layer_length_{lth}", tensor=length
                    )

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(
                x=audio_signal, lengths=length
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length


@dataclass
class FastConformerCache:
    """Cache state for cache-aware streaming inference.

    Stores left-context from previous streaming steps for both
    self-attention layers (``last_channel``) and convolution layers
    (``last_time``), along with the number of valid cached frames.

    Use :meth:`FastConformerCacheAwareMultilevelCTC.get_initial_cache` to
    obtain the initial cache for a new stream, or set
    ``model.cache = model.get_initial_cache(batch_size)``.

    Args:
        last_channel:
            Cached left-context for self-attention layers.  Shape matches
            the output of ``encoder.get_initial_cache_state()``.
        last_time:
            Cached left-context for convolution layers.
        last_channel_len:
            Number of valid cached frames per sample.
            Shape ``(batch_size,)``.
    """

    last_channel: torch.FloatTensor
    last_time: torch.FloatTensor
    last_channel_len: torch.LongTensor


@dataclass
class FastConformerCTCWithCacheOutput(ModelOutput):
    """Output type for the cache-aware multi-level CTC model.

    Args:
        loss:
            Weighted sum of CTC losses across all output levels.
            Shape ``(1,)``.  ``None`` during pure inference (no labels).
        logits:
            Dictionary mapping each level name to its CTC logits tensor.
            Each tensor has shape ``(batch_size, num_encoder_frames,
            vocab_size)``.
        encoder_output:
            Raw encoder output (before CTC heads).  Shape
            ``(batch_size, num_encoder_frames, d_model)``.
        encoder_lengths:
            Number of valid encoder frames per sequence.
            Shape ``(batch_size,)``.
        cache:
            Updated cache state after this streaming step.
            ``None`` during offline training (when no cache was provided).
        attentions:
            Attention weights from the encoder, if requested via
            ``output_attentions=True``.  May be ``None``.
    """

    loss: torch.FloatTensor | None = None
    logits: dict[str, torch.FloatTensor] | None = None
    encoder_output: torch.FloatTensor | None = None
    encoder_lengths: torch.LongTensor | None = None
    cache: FastConformerCache | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class FastConformerCacheAwareMultilevelCTC(PreTrainedModel):
    """FastConformer model with cache-aware streaming and multi-level CTC heads.

    This model wraps NeMo's :class:`~nemo.collections.asr.modules.ConformerEncoder`
    inside a HuggingFace :class:`~transformers.PreTrainedModel`, adding a
    mel-spectrogram processor (:class:`~processor.FastConformerMelProcessor`)
    and multiple linear CTC heads (one per linguistic / phonetic level).

    A single :meth:`forward` handles both offline training and streaming
    inference:

    *   **Offline training** — set ``cache=None`` (default).  The full
        utterance is processed with a ``chunked_limited`` attention mask.
    *   **Streaming inference** — supply a :class:`FastConformerCache` via
        the ``cache`` argument.  The encoder splits into
        ``forward_internal()`` + ``streaming_post_process()`` for cache reuse
        across chunks.

    Args:
        config:
            Model configuration; see
            :class:`FastConformerCacheAwareMultilevelCTCConfig`.
    """

    config_class = FastConformerCacheAwareMultilevelCTCConfig

    def __init__(self, config: FastConformerCacheAwareMultilevelCTCConfig) -> None:
        super().__init__(config)

        # 1. Mel-spectrogram processor (raw audio → mel features)
        self.processor = FastConformerMelProcessor(**config.processor_kwargs)

        # 2. NeMo FastConformer encoder

        self.encoder = MuaalemConformerEncoder(
            feat_in=config.feat_in,
            feat_out=config.feat_out,
            n_layers=config.n_layers,
            d_model=config.d_model,
            use_bias=config.use_bias,
            subsampling=config.subsampling,
            subsampling_factor=config.subsampling_factor,
            subsampling_conv_channels=config.subsampling_conv_channels,
            causal_downsampling=config.causal_downsampling,
            ff_expansion_factor=config.ff_expansion_factor,
            self_attention_model=config.self_attention_model,
            n_heads=config.n_heads,
            att_context_size=config.att_context_size,
            att_context_style=config.att_context_style,
            xscaling=config.xscaling,
            pos_emb_max_len=config.pos_emb_max_len,
            conv_kernel_size=config.conv_kernel_size,
            conv_norm_type=config.conv_norm_type,
            conv_context_size=config.conv_context_size,
            dropout=config.dropout,
            dropout_pre_encoder=config.dropout_pre_encoder,
            dropout_emb=config.dropout_emb,
            dropout_att=config.dropout_att,
            stochastic_depth_drop_prob=config.stochastic_depth_drop_prob,
            stochastic_depth_mode=config.stochastic_depth_mode,
            stochastic_depth_start_layer=config.stochastic_depth_start_layer,
        )

        # 3. Multi-level CTC heads
        level_to_vocab_size = config.level_to_vocab_size
        if not level_to_vocab_size:
            raise ValueError(
                "At least one CTC level must be defined in "
                "`config.level_to_vocab_size`.  Received an empty dictionary."
            )
        self.level_to_lm_head = nn.ModuleDict(
            {
                level: nn.Linear(config.d_model, vocab_size)
                for level, vocab_size in level_to_vocab_size.items()
            }
        )

        # 4. Current cache state (``None`` until set by caller)
        self.cache: FastConformerCache | None = None

        # 5. Weight initialisation and tying
        # TODO: is this right after loading nemo checkpoint ?
        self.post_init()

    # ------------------------------------------------------------------
    # Setup streaming parameters
    # ------------------------------------------------------------------

    def setup_streaming_params(self) -> None:
        """Compute and configure the encoder's cache-aware streaming parameters.

        This is a convenience wrapper around
        ``self.encoder.setup_streaming_params()``.  It must be called **once**
        before the first streaming inference step.

        The computed configuration is stored in ``self.encoder.streaming_cfg``
        and includes ``chunk_size``, ``shift_size``, ``valid_out_len``,
        ``pre_encode_cache_size``, ``drop_extra_pre_encoded``,
        ``last_channel_cache_size``, and ``last_time_cache_size``.

        Example:
            >>> model = FastConformerCacheAwareMultilevelCTC(config)
            >>> model.setup_streaming_params()
            >>> print(model.encoder.streaming_cfg)
        """
        self.encoder.setup_streaming_params()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def get_initial_cache(self, batch_size: int = 1) -> FastConformerCache:
        """Create an empty initial cache for a new streaming session.

        Delegates to ``self.encoder.get_initial_cache_state()`` and wraps
        the result in a :class:`FastConformerCache` dataclass.

        Args:
            batch_size:
                Batch dimension for the cache tensors.  Defaults to ``1``.

        Returns:
            An empty :class:`FastConformerCache` ready for the first
            streaming step.
        """
        ch, t, l = self.encoder.get_initial_cache_state(batch_size=batch_size)
        return FastConformerCache(last_channel=ch, last_time=t, last_channel_len=l)

    # ------------------------------------------------------------------
    # Unified forward pass (offline training + streaming inference)
    # ------------------------------------------------------------------

    def forward(
        self,
        raw_audio: torch.FloatTensor,
        audio_length: torch.LongTensor,
        cache: FastConformerCache | None = None,
        keep_all_outputs: bool = True,
        drop_extra_pre_encoded: int | None = None,
        labels: dict[str, torch.LongTensor] | None = None,
        labels_mask: dict[str, torch.BoolTensor] | None = None,
        processed_signal: torch.FloatTensor | None = None,
        processed_length: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        selected_levels: set[str] | None = None,
    ) -> Union[tuple, FastConformerCTCWithCacheOutput]:
        r"""Forward pass — offline training or streaming inference.

        **Offline training** (``cache=None``, the default):

        Processes the full audio sequence in a single encoder pass with
        NeMo's ``chunked_limited`` attention mask.  No cache is involved.

        **Streaming inference** (``cache=FastConformerCache(...)``):

        Splits the encoder execution into ``forward_internal()`` +
        ``streaming_post_process()`` to reuse left-context cache across
        chunks.  The updated cache is returned in
        :attr:`FastConformerCTCWithCacheOutput.cache`.

        **Dataflow:**

        1.  ``raw_audio`` (PCM waveforms) → ``processor`` → mel spectrograms
            (or pass pre-processed mel via ``processed_signal``)
        2.  Mel spectrograms → ``encoder`` (offline or streaming)
        3.  Encoder output → per-level ``Linear`` heads → per-level logits
        4.  If ``labels`` and ``labels_mask`` are provided, compute the
            weighted CTC loss.

        Args:
            raw_audio:
                Raw PCM audio waveforms.  Shape ``(batch_size, num_samples)``.
                All sequences must be padded to the same length; use
                ``audio_length`` for the number of valid samples.
                Ignored when ``processed_signal`` is provided.
            audio_length:
                Number of valid samples per waveform.  Shape ``(batch_size,)``.
                Ignored when ``processed_signal`` is provided.
            cache:
                If ``None`` (default), run in **offline training** mode —
                the full utterance is processed in one pass.
                If a :class:`FastConformerCache`, run a **streaming step**
                with cache reuse via ``forward_internal()``.
            keep_all_outputs:
                Only meaningful when ``cache`` is provided.
                If ``True`` (offline) (typically the last streaming step), return
                **all** encoder output frames including the lookahead region.
                If ``False`` (streaming), drop the lookahead frames, keeping only the
                ``valid_out_len`` frames for the current chunk.
            drop_extra_pre_encoded:
                Only meaningful when ``cache`` is provided.
                Override for ``self.encoder.streaming_cfg.drop_extra_pre_encoded``.
                Set to ``0`` for the first step (no pre-encode cache to drop)
                and ``None`` for subsequent steps (uses config default).
            labels:
                Dictionary mapping each level name to its CTC target IDs of
                shape ``(batch_size, target_length)``.  Values ``-100`` (or
                ``self.config.pad_token_id``) are masked out.
            labels_mask:
                Dictionary mapping each level name to a boolean mask of shape
                ``(batch_size, target_length)``.  ``True`` = valid target,
                ``False`` = padding.  **Required** when ``labels`` is given.
            processed_signal:
                Pre-computed mel spectrograms.  Shape ``(batch_size,
                num_mel_bins, num_frames)``.  When provided, ``raw_audio``
                and ``audio_length`` are ignored.  Used by
                :class:`CacheAwareStreamingAudioBuffer` which yields
                already-processed chunks.
            processed_length:
                Number of valid frames per sequence in ``processed_signal``.
                Shape ``(batch_size,)``.  Required when ``processed_signal``
                is given.
            return_dict:
                If ``True``, return :class:`FastConformerCTCWithCacheOutput`.
                If ``False``, return a plain tuple.
                Defaults to ``self.config.use_return_dict``.
            selected_levels:
                If provided, only compute logits for the specified subset of
                levels.  ``None`` means all levels.

        Returns:
            :class:`FastConformerCTCWithCacheOutput` (when ``return_dict=True``)
            with ``loss``, ``logits``, ``encoder_output``, ``encoder_lengths``,
            ``cache`` (updated after a streaming step, ``None`` in offline
            mode), and ``attentions``.

            When ``return_dict=False``, a tuple ``(loss, logits,
            encoder_output, encoder_lengths)``.  ``loss`` is ``None`` during
            inference.

        Raises:
            ValueError:
                If ``labels`` is provided without ``labels_mask``.
            ValueError:
                If a level in ``labels`` is not in ``config.level_to_vocab_size``.

        Example — offline training::

            >>> out = model(raw_audio=audio, audio_length=lengths,
            ...             labels=labels, labels_mask=mask)
            >>> out.loss.backward()

        Example — streaming inference::

            >>> model.setup_streaming_params()
            >>> model.cache = model.get_initial_cache(batch_size=1)
            >>> streaming_buffer = CacheAwareStreamingAudioBuffer(model)
            >>> streaming_buffer.append_audio(audio_np)
            >>> for step, (chunk_proc, chunk_len) in enumerate(streaming_buffer):
            ...     drop = 0 if step == 0 else None
            ...     out = model(
            ...         raw_audio=None,
            ...         audio_length=None,
            ...         processed_signal=chunk_proc,
            ...         processed_length=chunk_len,
            ...         cache=model.cache,
            ...         keep_all_outputs=streaming_buffer.is_buffer_empty(),
            ...         drop_extra_pre_encoded=drop,
            ...     )
            ...     model.cache = out.cache
            ...     logits = model.level_to_lm_head["phonemes"](out.encoder_output)
            ...     preds = logits.argmax(dim=-1)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # ---- 1. Input validation ----
        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError(
                    f"`labels` must be a dictionary mapping level names to "
                    f"target tensors.  Got {type(labels)}."
                )
            for level in labels:
                if level not in self.config.level_to_vocab_size:
                    raise ValueError(
                        f"Level '{level}' in `labels` is not defined in "
                        f"`config.level_to_vocab_size`.  Available levels: "
                        f"{list(self.config.level_to_vocab_size.keys())}."
                    )
            if labels_mask is None:
                raise ValueError(
                    "`labels_mask` is required whenever `labels` is provided. "
                    "It must be a dictionary of the same structure as `labels`, "
                    "with boolean tensors indicating valid target positions."
                )
            for level in labels:
                if labels_mask[level].shape != labels[level].shape:
                    raise ValueError(
                        f"Shape mismatch for level '{level}': "
                        f"labels {labels[level].shape} vs "
                        f"labels_mask {labels_mask[level].shape}. "
                        "Both must have the same shape."
                    )

        # ---- 2. Mel spectrograms — from raw audio or pre-computed ----
        if processed_signal is not None:
            if processed_length is None:
                raise ValueError(
                    "`processed_length` is required when `processed_signal` is provided."
                )
        else:
            if raw_audio is None or audio_length is None:
                raise ValueError(
                    "Either provide `raw_audio`+`audio_length` (offline training) "
                    "or `processed_signal`+`processed_length` (streaming buffer)."
                )
            processed_signal, processed_length = self.processor(
                input_signal=raw_audio,
                length=audio_length,
            )

        # ---- 3. Encoder forward — offline or streaming ----
        new_cache: FastConformerCache | None = None

        if cache is None:
            encoder_output, encoder_lengths = self.encoder(
                audio_signal=processed_signal,
                length=processed_length,
                cache_last_channel=None,
                cache_last_time=None,
                cache_last_channel_len=None,
            )
        else:
            # --- Streaming inference (forward_internal + streaming_post_process) ---
            if drop_extra_pre_encoded is not None:
                self.encoder.streaming_cfg.drop_extra_pre_encoded = (
                    drop_extra_pre_encoded
                )

            rets = self.encoder.forward_internal(
                audio_signal=processed_signal,
                length=processed_length,
                cache_last_channel=cache.last_channel,
                cache_last_time=cache.last_time,
                cache_last_channel_len=cache.last_channel_len,
            )
            encoder_output, encoder_lengths, new_ch, new_t, new_len = (
                self.encoder.streaming_post_process(
                    rets, keep_all_outputs=keep_all_outputs
                )
            )

            new_cache = FastConformerCache(
                last_channel=new_ch,
                last_time=new_t,
                last_channel_len=new_len,
            )

        # NeMo returns (B, d_model, T); permute to (B, T, d_model) for linear heads
        encoder_output = encoder_output.transpose(1, 2)  # (B, T, d_model)

        # ---- 4. Per-level CTC heads ----
        level_to_logits: dict[str, torch.FloatTensor] = {}
        for level in self.level_to_lm_head:
            if selected_levels is not None and level not in selected_levels:
                continue
            level_to_logits[level] = self.level_to_lm_head[level](encoder_output)

        # ---- 5. CTC loss with labels_mask ----
        loss: torch.FloatTensor | None = None
        if labels is not None:
            loss = torch.tensor(0.0, device=raw_audio.device, dtype=torch.float32)
            input_lengths = encoder_lengths.to(torch.long)

            for level in labels:
                target_lengths = labels_mask[level].sum(dim=-1).to(torch.long)
                flattened_targets = labels[level].masked_select(
                    labels_mask[level].to(torch.bool)
                )

                # log_softmax over vocab, then transpose to (T, N, C) for CTC
                log_probs = nn.functional.log_softmax(
                    level_to_logits[level], dim=-1, dtype=torch.float32
                ).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    level_loss = nn.functional.ctc_loss(
                        log_probs=log_probs,
                        targets=flattened_targets,
                        input_lengths=input_lengths,
                        target_lengths=target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
                loss = loss + self.config.level_to_loss_weight[level] * level_loss

        # ---- 6. Return ----
        if not return_dict:
            output = (level_to_logits, encoder_output, encoder_lengths)
            return ((loss,) + output) if loss is not None else output

        return FastConformerCTCWithCacheOutput(
            loss=loss,
            logits=level_to_logits,
            encoder_output=encoder_output,
            encoder_lengths=encoder_lengths,
            cache=new_cache,
            attentions=None,
        )
