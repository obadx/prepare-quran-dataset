"""Streaming inference utilities for the FastConformer cache-aware CTC model.

This module provides :class:`FastConformerCacheAwareMultilevelCTCInference`,
a high-level wrapper for running cache-aware streaming inference on audio files.

The wrapper integrates:

*   :class:`~processor.FastConformerMelProcessor` — raw audio → mel spectrograms.
*   Manual chunking loop with zero-padded tail for constant lookahead delay.
*   :meth:`~modeling_fastconformer_cache_aware_ctc.FastConformerCacheAwareMultilevelCTC`
    — per-step encoder execution with cache reuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from .modeling_fastconformer_cache_aware_ctc import (
    FastConformerCacheAwareMultilevelCTC,
)


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

    This convenience function computes mel spectrograms from the audio,
    pads the tail with zeros for constant lookahead delay, iterates
    through the streaming steps, and returns the concatenated argmax
    token IDs for the requested output level.

    Unlike the NeMo ``CacheAwareStreamingAudioBuffer`` path, this
    implementation computes mel directly via ``model.processor`` and
    performs manual chunking — it does **not** require
    ``model._cfg`` (which the NeMo buffer needs) and therefore works
    with the HuggingFace-wrapped model.

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
    if (
        not hasattr(model.encoder, "streaming_cfg")
        or model.encoder.streaming_cfg is None
    ):
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

    # --- Compute mel spectrograms ---
    audio_tensor = torch.tensor(audio, dtype=dtype, device=device).unsqueeze(0)
    mel, mel_len = model.processor(
        input_signal=audio_tensor,
        length=torch.tensor([len(audio)], device=device),
    )

    # --- Zero-pad tail for constant lookahead delay ---
    # When att_context_size has a third entry (C), each emitted frame
    # needs C future frames as keys.  The last few output frames would
    # otherwise see truncated lookahead; padding the tail with zeros
    # gives them full C-frame lookahead (their predictions are noise,
    # but frames before them get correct constant-delay context).
    att_ctx = model.config.att_context_size
    C = att_ctx[2] if isinstance(att_ctx, list) and len(att_ctx) > 2 else 0
    if C > 0:
        pad_mel = C * model.encoder.subsampling_factor
        mel = torch.nn.functional.pad(mel, (0, pad_mel))
        mel_len = mel_len + pad_mel

    # --- Initialise cache ---
    cache = model.get_initial_cache(batch_size=1)

    # --- Streaming loop ---
    # Manual chunking (no NeMo buffer) — drop_extra_pre_encoded is always
    # 0 because we feed fresh mel chunks without a pre-encode cache.
    s = model.encoder.streaming_cfg
    all_tokens: Optional[torch.FloatTensor] = None
    buffer_idx = 0
    step_num = 0

    while buffer_idx < mel.size(-1):
        # Determine chunk size and shift for this step
        if step_num == 0:
            chunk_size = s.chunk_size[0]
            shift = s.shift_size[0]
        else:
            chunk_size = s.chunk_size[1]
            shift = s.shift_size[1]

        # Slice chunk (no pre-encode cache)
        end = min(buffer_idx + chunk_size, mel.size(-1))
        chunk = mel[:, :, buffer_idx:end]

        # Stop if chunk is too small to produce output
        if chunk.size(-1) < 1:
            break

        # No pre-encode cache → always drop 0 extra frames
        model.encoder.streaming_cfg.drop_extra_pre_encoded = 0

        # Determine if this is the last step (no more shifts fit)
        is_last = (buffer_idx + shift >= mel.size(-1))

        # Forward pass
        out = model(
            raw_audio=None,
            audio_length=None,
            processed_signal=chunk,
            processed_length=torch.tensor([chunk.size(-1)], device=device),
            cache=cache,
            keep_all_outputs=is_last,
            drop_extra_pre_encoded=0,
        )
        cache = out.cache

        # Collect predictions
        logits = model.level_to_lm_head[level](out.encoder_output)
        pred_ids = logits.argmax(dim=-1)

        if all_tokens is None:
            all_tokens = pred_ids
        else:
            all_tokens = torch.cat([all_tokens, pred_ids], dim=-1)

        buffer_idx += shift
        step_num += 1

    if all_tokens is not None:
        return all_tokens[0].tolist()
    return []


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
