"""Cache-aware streaming audio buffer for the HuggingFace-wrapped FastConformer.

NeMo's :class:`~nemo.collections.asr.parts.utils.streaming_utils.CacheAwareStreamingAudioBuffer`
already implements the two pieces of chunking that are easy to get wrong:

*   every chunk is prefixed with ``streaming_cfg.pre_encode_cache_size`` mel
    frames of **real** left context taken from the preceding audio (zeros only
    at the very start), so the causal subsampling convolutions do not see a
    zero boundary at each chunk seam;
*   the first step uses ``chunk_size[0]`` / ``shift_size[0]`` while every later
    step uses ``chunk_size[1]`` / ``shift_size[1]``.

It is written against NeMo ``ASRModel`` instances, though, and reaches for two
things a :class:`~transformers.PreTrainedModel` does not have: ``model._cfg``
(+ ``model.from_config_dict``) to rebuild the preprocessor, and a NeMo-flavoured
``model.device``.  This module supplies both.
"""

from __future__ import annotations

import torch
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)

from .processor import FastConformerMelProcessor


class HFCacheAwareStreamingAudioBuffer(CacheAwareStreamingAudioBuffer):
    """``CacheAwareStreamingAudioBuffer`` bound to a HF FastConformer model.

    Only the two NeMo-model-specific hooks are overridden; all of the chunking,
    pre-encode-cache handling and multi-stream bookkeeping is inherited.

    Args:
        model:
            A :class:`~.modeling_fastconformer_cache_aware_ctc.FastConformerCacheAwareMultilevelCTC`.
            Its ``encoder`` must already have ``streaming_cfg`` set up (the
            parent calls ``setup_streaming_params()`` if it is ``None``).
        online_normalization:
            Forwarded to the parent.  Leave ``None``/``False`` for models
            trained with ``normalize="NA"``, which is the case here.
        pad_and_drop_preencoded:
            Forwarded to the parent.  ``False`` (the default) makes step 0 use
            ``chunk_size[0]`` and prepend ``pre_encode_cache_size[0]`` zeros.

    Example:
        >>> buffer = HFCacheAwareStreamingAudioBuffer(model)
        >>> buffer.append_audio(waveform)          # 1-D float32 numpy array
        >>> for chunk, chunk_len in buffer:
        ...     ...                                 # chunk is (1, n_mels, T)
    """

    def extract_preprocessor(self):
        """Build the mel front-end without going through ``model._cfg``.

        Mirrors NeMo by disabling ``dither`` and ``pad_to`` so repeated
        evaluation runs are deterministic — the model's own processor keeps
        ``dither=1e-5`` for training.  Also sets ``self.model_normalize_type``,
        which the parent otherwise only assigns inside this method.
        """
        kwargs = dict(self.model.config.processor_kwargs)
        self.model_normalize_type = kwargs.get("normalize", "NA")
        kwargs["dither"] = 0.0
        kwargs["pad_to"] = 0
        if self.online_normalization:
            kwargs["normalize"] = "NA"
        return FastConformerMelProcessor(**kwargs).to(self.get_model_device())

    def get_model_device(self) -> torch.device:
        """Device of the wrapped model (``PreTrainedModel.device``)."""
        return self.model.device

    # ------------------------------------------------------------------
    # Chunk geometry — used to pad short tail chunks
    # ------------------------------------------------------------------

    def expected_chunk_width(self, step: int) -> int:
        """Full mel width a chunk has at ``step``, including the pre-encode cache.

        ``MuaalemConformerEncoder._create_masks`` asserts that the encoder input
        is exactly ``att_context_size[0] + chunk_size`` frames wide during
        streaming.  The buffer's final chunk is normally short, so callers must
        right-pad it to this width (keeping the true ``chunk_len``, which masks
        the padding out) instead of relaxing the assertion.

        Args:
            step: 0-based streaming step index.

        Returns:
            Expected number of mel frames for that step's chunk.
        """
        return self._geometry_at(self.streaming_cfg.chunk_size, step) + self._geometry_at(
            self.streaming_cfg.pre_encode_cache_size, step
        )

    @staticmethod
    def _geometry_at(value, step: int) -> int:
        """Pick the step-0 or steady-state entry of a ``streaming_cfg`` field."""
        if isinstance(value, (list, tuple)):
            return value[0] if step == 0 else value[1]
        return value
