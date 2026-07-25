"""Streaming inference utilities for the FastConformer cache-aware CTC model.

This module provides :class:`FastConformerCacheAwareMultilevelCTCInference`,
a high-level wrapper for running cache-aware streaming inference on audio files.

The wrapper integrates:

*   :class:`~processor.FastConformerMelProcessor` — raw audio → mel spectrograms.
*   NeMo's :class:`~nemo.collections.asr.parts.utils.streaming_utils.CacheAwareStreamingAudioBuffer`
    — manages the sliding window of mel frames across streaming steps.
*   :meth:`~modeling_fastconformer_cache_aware_ctc.FastConformerCacheAwareMultilevelCTC.cache_aware_stream_step`
    — per-step encoder execution with cache reuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from .modeling_fastconformer_cache_aware_ctc import (
    FastConformerCacheAwareMultilevelCTC,
)
from .processor import FastConformerMelProcessor


def _calc_drop_extra_pre_encoded(
    model: FastConformerCacheAwareMultilevelCTC,
    step_num: int,
    pad_and_drop_preencoded: bool = False,
) -> int:
    """Determine ``drop_extra_pre_encoded`` for a given streaming step.

    The first streaming step has no pre-encode cache, so no frames need to
    be dropped.  Subsequent steps use the encoder's configured default.

    Args:
        model: The FastConformer model instance.
        step_num: Zero-based streaming step index.
        pad_and_drop_preencoded: If ``True``, always use the configured
            drop value (enables ONNX-friendly uniform step behaviour).

    Returns:
        The number of frames to drop after the pre-encode subsampling.
    """
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return int(model.encoder.streaming_cfg.drop_extra_pre_encoded)


@torch.inference_mode()
def stream_inference(
    model: FastConformerCacheAwareMultilevelCTC,
    audio: NDArray[np.float32],
    sample_rate: int = 16000,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    level: str = "phonemes",
) -> list[int]:
    """Run cache-aware streaming inference on a single audio waveform.

    This convenience function loads the audio, creates a
    ``CacheAwareStreamingAudioBuffer``, iterates through the streaming steps,
    and returns the concatenated argmax token IDs for the requested output
    level.

    Args:
        model:
            The FastConformer model in evaluation mode.  Streaming parameters
            must already be set up via ``model.setup_streaming_params()``.
        audio:
            Mono PCM audio waveform at ``sample_rate`` Hz.
            1-D array of shape ``(num_samples,)``.
        sample_rate:
            Audio sampling rate in Hz.  Must match the model's processor
            configuration.  Defaults to ``16000``.
        device:
            Device to run inference on.  If ``None``, uses the model's
            current device.
        dtype:
            Torch dtype for the processed signal.  Defaults to
            ``torch.float32`` (required for cache-aware models).
        level:
            Output level name (must exist in ``model.level_to_lm_head``).
            Defaults to ``"phonemes"``.

    Returns:
        List of token IDs (argmax over the vocabulary) for the requested
        level, aggregated across all streaming steps.  The caller can then
        use :class:`~modeling_streaming_rnn.multi_level_tokenizer.MultiLevelTokenizer`
        to decode these into text.

    Raises:
        ValueError:
            If ``model.encoder.streaming_cfg`` has not been set up (call
            ``model.setup_streaming_params()`` first).
        ValueError:
            If the requested ``level`` is not among the model's CTC heads.

    Example:
        >>> import numpy as np
        >>> from .modeling_fastconformer_cache_aware_ctc import (
        ...     FastConformerCacheAwareMultilevelCTC,
        ... )
        >>> config = FastConformerCacheAwareMultilevelCTCConfig(
        ...     level_to_vocab_size={"phonemes": 44},
        ... )
        >>> model = FastConformerCacheAwareMultilevelCTC(config)
        >>> model.setup_streaming_params()
        >>> model.eval()
        >>> audio = np.random.randn(48000).astype(np.float32)  # 3 seconds
        >>> tokens = stream_inference(model, audio)
        >>> print(tokens[:5])
        [12, 7, 31, 0, 18]
    """
    if not hasattr(model.encoder, "streaming_cfg") or model.encoder.streaming_cfg is None:
        raise ValueError(
            "Model encoder has no streaming configuration.  "
            "Call `model.setup_streaming_params()` before running inference."
        )
    if level not in model.level_to_lm_head:
        raise ValueError(
            f"Level '{level}' not found in model's CTC heads. "
            f"Available levels: {list(model.level_to_lm_head.keys())}."
        )

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    # Move model to device if needed
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # --- Create streaming buffer ---
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    streaming_buffer = CacheAwareStreamingAudioBuffer(model)
    streaming_buffer.append_audio(audio)

    # --- Initialise cache ---
    model.cache = model.get_initial_cache(
        batch_size=len(streaming_buffer.streams_length)
    )

    # --- Streaming loop ---
    pred_out_prev: Optional[torch.FloatTensor] = None
    all_token_ids: list[int] = []

    for step_num, (chunk_proc, chunk_len) in enumerate(iter(streaming_buffer)):
        chunk_proc = chunk_proc.to(device=device, dtype=dtype)
        chunk_len = chunk_len.to(device=device)

        drop = _calc_drop_extra_pre_encoded(model, step_num)

        out = model(
            raw_audio=None,
            audio_length=None,
            processed_signal=chunk_proc,
            processed_length=chunk_len,
            cache=model.cache,
            keep_all_outputs=streaming_buffer.is_buffer_empty(),
            drop_extra_pre_encoded=drop,
        )
        model.cache = out.cache

        logits = model.level_to_lm_head[level](out.encoder_output)
        pred_ids = logits.argmax(dim=-1)

        # Concatenate predictions across steps
        if pred_out_prev is not None:
            valid_out_len = int(model.encoder.streaming_cfg.valid_out_len)
            pred_out_prev = torch.cat([pred_out_prev, pred_ids], dim=-1)
        else:
            pred_out_prev = pred_ids

    if pred_out_prev is not None:
        all_token_ids = pred_out_prev[0].tolist()

    return all_token_ids


class FastConformerCacheAwareMultilevelCTCInference:
    """High-level wrapper for cache-aware streaming inference.

    Provides a reusable object that loads a trained model, sets up streaming
    parameters, and streams audio files (or chunks) to produce token-ID
    predictions for a specified output level.

    Args:
        model_or_path:
            Either an instantiated
            :class:`~modeling_fastconformer_cache_aware_ctc.FastConformerCacheAwareMultilevelCTC`
            model, or a path to a directory with a saved HF model (via
            ``from_pretrained``).  The model must have its streaming
            parameters already configured in the config.
        device:
            Device for inference (``"cpu"``, ``"cuda"``, etc.).  Defaults to
            ``"cpu"``.
        dtype:
            Torch dtype.  Cache-aware models require ``torch.float32``.
            Defaults to ``torch.float32``.
        level:
            Output level name.  Defaults to ``"phonemes"``.

    Example:
        >>> inference = FastConformerCacheAwareMultilevelCTCInference(
        ...     "/path/to/model",
        ...     device="cuda",
        ... )
        >>> audio = np.random.randn(48000).astype(np.float32)
        >>> tokens = inference(audio)
        >>> print(tokens)
        [12, 7, 31, ...]
    """

    def __init__(
        self,
        model_or_path: str | Path | FastConformerCacheAwareMultilevelCTC,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        level: str = "phonemes",
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.level = level

        if isinstance(model_or_path, FastConformerCacheAwareMultilevelCTC):
            self.model = model_or_path
        else:
            self.model = FastConformerCacheAwareMultilevelCTC.from_pretrained(
                str(model_or_path)
            )

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.model.setup_streaming_params()

    def __call__(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> list[int]:
        """Run streaming inference on a single audio waveform.

        Args:
            audio:
                Mono PCM audio array of shape ``(num_samples,)``.
            sample_rate:
                Sampling rate in Hz.  Must match the model's processor.
                Defaults to 16000.

        Returns:
            List of argmax token IDs for the configured output level.
        """
        return stream_inference(
            model=self.model,
            audio=audio,
            sample_rate=sample_rate,
            device=self.device,
            dtype=self.dtype,
            level=self.level,
        )
