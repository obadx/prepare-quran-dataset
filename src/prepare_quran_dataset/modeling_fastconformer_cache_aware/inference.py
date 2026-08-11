"""Streaming inference utilities for the FastConformer cache-aware CTC model.

This module provides :func:`stream_inference` and
:class:`FastConformerCacheAwareMultilevelCTCInference` for running cache-aware
streaming inference on audio.

The chunk loop is driven by
:class:`~.streaming_buffer.HFCacheAwareStreamingAudioBuffer`, a thin adapter over
NeMo's ``CacheAwareStreamingAudioBuffer``, so each chunk is fed to the encoder
with the correct pre-encode left context rather than a zero boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from .modeling_fastconformer_cache_aware_ctc import (
    FastConformerCacheAwareMultilevelCTC,
)
from .streaming_buffer import HFCacheAwareStreamingAudioBuffer


@torch.inference_mode()
def stream_inference(
    model: FastConformerCacheAwareMultilevelCTC,
    audio: NDArray[np.float32] | None = None,
    *,
    processed_signal: torch.Tensor | None = None,
    processed_length: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    levels: set[str] | None = None,
) -> dict[str, list[int]]:
    """Run cache-aware streaming inference on a single audio stream.

    Feeds the audio to the encoder chunk by chunk, threading the
    :class:`~.modeling_fastconformer_cache_aware_ctc.FastConformerCache` between
    steps, and returns the per-frame argmax token ids for every requested output
    level.  The result is *not* CTC-collapsed — callers decide how to decode.

    Provide **either** ``audio`` (a raw waveform, mel is computed here) **or**
    ``processed_signal`` + ``processed_length`` (mel already computed, e.g. by a
    dataloader collator).

    Args:
        model:
            The model to run.  ``setup_streaming_params()`` is called
            automatically if the encoder has no streaming configuration yet.
        audio:
            Mono PCM waveform at the processor's sample rate, shape
            ``(num_samples,)``.  Ignored when ``processed_signal`` is given.
        processed_signal:
            Pre-computed log-mel spectrogram, shape ``(1, num_mel_bins,
            num_frames)``.
        processed_length:
            Number of valid mel frames, shape ``(1,)``.  Required with
            ``processed_signal``.
        device:
            Device to run on.  Defaults to the model's current device; the model
            is **not** moved.
        levels:
            Subset of CTC heads to decode.  ``None`` means every level in
            ``model.level_to_lm_head``.

    Returns:
        ``{level: [token_id, ...]}`` — one frame-aligned id sequence per level,
        concatenated across all streaming steps.  Use
        :class:`~.multi_level_tokenizer.MultiLevelTokenizer` (or the caller's own
        greedy CTC collapse) to turn these into text.

    Raises:
        ValueError:
            If neither ``audio`` nor ``processed_signal``/``processed_length`` is
            provided, or if ``levels`` names a head the model does not have.

    Example:
        >>> model.setup_streaming_params()
        >>> ids = stream_inference(model, waveform)
        >>> ids["phonemes"][:5]
        [0, 12, 12, 0, 31]

    .. note::
        Single stream only (batch size 1).  The chunk loop keeps one
        ``FastConformerCache``, and the buffer's per-stream tail handling would
        need per-sample bookkeeping to batch safely.
    """
    if levels is None:
        levels = set(model.level_to_lm_head.keys())
    else:
        levels = set(levels)
        unknown = levels - set(model.level_to_lm_head.keys())
        if unknown:
            raise ValueError(
                f"Unknown level(s) {sorted(unknown)}. Available levels: "
                f"{list(model.level_to_lm_head.keys())}."
            )

    if processed_signal is None and audio is None:
        raise ValueError(
            "Provide either `audio` (raw waveform) or `processed_signal`+"
            "`processed_length` (pre-computed mel)."
        )
    if processed_signal is not None and processed_length is None:
        raise ValueError(
            "`processed_length` is required when `processed_signal` is provided."
        )

    if getattr(model.encoder, "streaming_cfg", None) is None:
        model.setup_streaming_params()

    device = torch.device(device) if device is not None else model.device
    model.eval()

    # The streaming loop always passes `drop_extra_pre_encoded` explicitly, so
    # capture the configured value up front: `forward()` overwrites the field in
    # place and never restores it, which would otherwise make step 1 onwards
    # inherit step 0's zero.
    default_drop = model.encoder.streaming_cfg.drop_extra_pre_encoded

    buffer = HFCacheAwareStreamingAudioBuffer(model)
    if processed_signal is not None:
        valid = int(processed_length[0])
        buffer.append_processed_signal(
            processed_signal[..., :valid].to(device)
        )
    else:
        buffer.append_audio(np.asarray(audio, dtype=np.float32))

    cache = model.get_initial_cache(batch_size=1)
    level_to_ids: dict[str, list[int]] = {level: [] for level in levels}

    for step, (chunk, chunk_len) in enumerate(buffer):
        # The final chunk is normally short.  `_create_masks` asserts an exact
        # encoder input width during streaming (a deliberate guard), so pad the
        # mel out to the full width and let `chunk_len` mask the padding.
        expected = buffer.expected_chunk_width(step)
        if chunk.size(-1) < expected:
            chunk = torch.nn.functional.pad(chunk, (0, expected - chunk.size(-1)))

        out = model(
            processed_signal=chunk.to(device),
            processed_length=chunk_len.to(device),
            cache=cache,
            keep_all_outputs=buffer.is_buffer_empty(),
            drop_extra_pre_encoded=0 if step == 0 else default_drop,
            selected_levels=levels,
        )
        cache = out.cache

        # Trim the padded / lookahead tail so it is never emitted as tokens.
        valid_frames = int(out.encoder_lengths[0])
        if valid_frames <= 0:
            continue

        for level in levels:
            frame_ids = out.logits[level][0, :valid_frames].argmax(dim=-1)
            level_to_ids[level].extend(frame_ids.tolist())

    return level_to_ids


class FastConformerCacheAwareMultilevelCTCInference:
    """High-level wrapper for cache-aware streaming inference.

    Loads a trained model, configures streaming parameters once, and streams
    waveforms to produce per-level token-id predictions.

    Args:
        model_or_path:
            Either an instantiated
            :class:`~.modeling_fastconformer_cache_aware_ctc.FastConformerCacheAwareMultilevelCTC`
            or a path / Hub id to load with ``from_pretrained``.
        device:
            Device for inference.  Defaults to ``"cpu"``.
        dtype:
            Torch dtype for the model.  Cache-aware streaming requires
            ``torch.float32``; the mel front-end stays float32 regardless.
        levels:
            Subset of CTC heads to decode.  ``None`` means all of them.

    Example:
        >>> inference = FastConformerCacheAwareMultilevelCTCInference(
        ...     "/path/to/model", device="cuda",
        ... )
        >>> ids = inference(waveform)
        >>> sorted(ids)[:2]
        ['ghonna', 'hams_or_jahr']
    """

    def __init__(
        self,
        model_or_path: str | Path | FastConformerCacheAwareMultilevelCTC,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        levels: set[str] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.levels = levels

        if isinstance(model_or_path, FastConformerCacheAwareMultilevelCTC):
            self.model = model_or_path
        else:
            self.model = FastConformerCacheAwareMultilevelCTC.from_pretrained(
                str(model_or_path)
            )

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.model.setup_streaming_params()

    def __call__(self, audio: NDArray[np.float32]) -> dict[str, list[int]]:
        """Stream one waveform and return per-level frame-aligned token ids.

        Args:
            audio: Mono PCM waveform of shape ``(num_samples,)``.

        Returns:
            ``{level: [token_id, ...]}``.
        """
        return stream_inference(
            model=self.model,
            audio=audio,
            device=self.device,
            levels=self.levels,
        )
