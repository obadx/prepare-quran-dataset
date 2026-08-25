from collections.abc import Sequence

import torch
from librosa.core import load
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)

from .modeling_fastconformer_cache_aware_ctc import (
    FastConformerCacheAwareMultilevelCTC,
)
from .processor import FastConformerMelProcessor

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Cache-aware streaming inference for FastConformer multi-level CTC models.

Reference: https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py

Call chain (mirrors NeMo)::

    NeMo:  conformer_stream_step
               -> encoder.cache_aware_stream_step
                    -> encoder.forward + streaming_post_process
               -> decoder

    Ours:  model.streaming_step
               -> model.forward(processed_signal, cache)
                    -> encoder.forward_internal + streaming_post_process
                    -> per-level CTC heads
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def calc_drop_extra_pre_encoded(model, step_num, pad_and_drop_preencoded):
    """Number of pre-encoded frames to drop after subsampling.

    For the first step there is no pre-encode cache, so nothing is dropped.
    For subsequent steps the value comes from ``encoder.streaming_cfg``.
    """
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return model.encoder.streaming_cfg.drop_extra_pre_encoded


# ---------------------------------------------------------------------------
# Streaming audio buffer (adapts NeMo's CacheAwareStreamingAudioBuffer)
# ---------------------------------------------------------------------------


class MuaalemCacheAwareStreamingAudioBuffer(CacheAwareStreamingAudioBuffer):
    """Audio buffer that works with our HuggingFace-based model.

    NeMo's ``CacheAwareStreamingAudioBuffer.extract_preprocessor`` calls
    ``model._cfg`` and ``model.from_config_dict`` which do not exist on
    :class:`~transformers.PreTrainedModel`.  This subclass accepts a
    :class:`FastConformerMelProcessor` directly and skips that logic.

    Overrides ``__iter__`` to zero-pad the last chunk to the expected
    mel-frame count.  NeMo's parent ``__iter__`` slices the buffer
    directly, so the final chunk can be shorter than ``chunk_size``.
    With ``constant_lookahead_delay > 0``, the encoder's
    ``_create_masks`` requires an exact ``max_audio_length``; a short
    chunk triggers a ``ValueError``.  Padding restores the expected
    length while ``chunk_lengths`` (unchanged) keeps the valid-frame
    count correct for ``pad_mask``.
    """

    def __init__(
        self,
        model: FastConformerCacheAwareMultilevelCTC,
        preprocessor: FastConformerMelProcessor,
        online_normalization: bool | None = None,
        pad_and_drop_preencoded: bool = False,
    ):
        self.model = model
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None
        self.step = 0
        self.pad_and_drop_preencoded = pad_and_drop_preencoded
        self.online_normalization = online_normalization

        model.encoder.setup_streaming_params()
        self.streaming_cfg = model.encoder.streaming_cfg
        self.input_features = model.encoder._feat_in

        self.preprocessor = preprocessor

        if hasattr(model.encoder, "pre_encode") and hasattr(
            model.encoder.pre_encode, "get_sampling_frames"
        ):
            self.sampling_frames = model.encoder.pre_encode.get_sampling_frames()
        else:
            self.sampling_frames = None

    def extract_preprocessor(self):
        return self.preprocessor

    def __iter__(self):
        sc = self.streaming_cfg
        cs = (
            sc.chunk_size
            if isinstance(sc.chunk_size, list)
            else (sc.chunk_size, sc.chunk_size)
        )
        pc = (
            sc.pre_encode_cache_size
            if isinstance(sc.pre_encode_cache_size, list)
            else (sc.pre_encode_cache_size, sc.pre_encode_cache_size)
        )

        first_step = True
        for audio_chunk, chunk_lengths in super().__iter__():
            if first_step and not self.pad_and_drop_preencoded:
                expected = cs[0] + pc[0]
            else:
                expected = cs[1] + pc[1]
            first_step = False

            if audio_chunk.size(-1) < expected:
                audio_chunk = torch.nn.functional.pad(
                    audio_chunk, (0, expected - audio_chunk.size(-1))
                )
            yield audio_chunk, chunk_lengths

    def append_audio_file(self, audio_filepath, stream_id=-1):
        audio, _ = load(audio_filepath, mono=True, sr=16000)
        processed_signal, processed_signal_length, stream_id = self.append_audio(
            audio, stream_id
        )
        return processed_signal, processed_signal_length, stream_id


# ---------------------------------------------------------------------------
# Full streaming inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def infer_fastconformer_streaming(
    audio_sources: Sequence[str | bytes],
    device: str | torch.device,
    dtype: torch.dtype,
    model: FastConformerCacheAwareMultilevelCTC,
    processor: FastConformerMelProcessor,
    sampling_rate: int = 16000,
    pad_and_drop_preencoded: bool = False,
) -> dict[str, torch.FloatTensor]:
    """Stream audio through the model chunk-by-chunk.

    Follows NeMo's ``perform_streaming`` pattern from
    ``speech_to_text_cache_aware_streaming_infer.py``:

    1. Create a :class:`MuaalemCacheAwareStreamingAudioBuffer` and append all
       audio files (the buffer preprocesses raw audio to mel spectrograms).
    2. Obtain the initial cache via ``model.get_initial_cache(batch_size)``.
    3. Iterate the buffer — each iteration yields a mel-spectrogram chunk and
       its valid lengths.
    4. Call :meth:`model.streaming_step` for each chunk, threading the cache.
    5. Concatenate logits across steps and return them.

    The returned logits have the same format as
    ``model.forward().logits``: a dictionary mapping each level name to a
    tensor of shape ``(batch_size, T_total, vocab_size)``.

    Args:
        audio_sources:
            Paths (or bytes) to audio files.  ``len(audio_sources)`` is the
            batch size.  All files are loaded at the native sample rate
            expected by the processor (typically 16 kHz).
        device:
            Target device for inference (e.g. ``"cuda:0"``).
        dtype:
            Compute dtype (e.g. ``torch.float32`` or ``torch.bfloat16``).
        model:
            The FastConformer multi-level CTC model.  Must already have
            streaming parameters set up (``model.setup_streaming_params()``
            is called by the buffer automatically).
        processor:
            Mel-spectrogram processor used by the model.
        sampling_rate:
            Expected audio sample rate in Hz.  Passed to the buffer for
            audio loading.  Defaults to ``16000``.
        pad_and_drop_preencoded:
            If ``True``, pad the first audio chunk and always drop
            pre-encoded frames.  Eases ONNX export.  Defaults to
            ``False``.

    Returns:
        Dictionary mapping each level name to a logits tensor of shape
        ``(batch_size, T_total, vocab_size)`` — the concatenation of
        per-step logits along the time dimension.

    Example::

        logits = infer_fastconformer_streaming(
            audio_sources=["audio1.wav", "audio2.wav"],
            device="cuda",
            dtype=torch.float32,
            model=model,
            processor=model.processor,
        )
        pred_ids = {k: v.argmax(dim=-1) for k, v in logits.items()}
    """
    streaming_buffer = MuaalemCacheAwareStreamingAudioBuffer(
        model=model,
        preprocessor=processor,
        pad_and_drop_preencoded=pad_and_drop_preencoded,
    )

    # Append all audio files to the buffer (triggers mel preprocessing)
    batch_size = len(audio_sources)
    for source in audio_sources:
        streaming_buffer.append_audio_file(source, stream_id=-1)

    # Initial cache state (zeros)
    cache = model.get_initial_cache(batch_size=batch_size)

    # Accumulate logits across streaming steps per level
    all_logits: dict[str, list[torch.FloatTensor]] = {
        level: [] for level in model.level_to_lm_head
    }

    streaming_buffer_iter = iter(streaming_buffer)
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        chunk_audio = chunk_audio.to(dtype)
        step_out = model.streaming_step(
            processed_signal=chunk_audio,
            processed_length=chunk_lengths,
            cache=cache,
            # keep_all_outputs must be True for the last step, otherwise the
            # final frames (including the lookahead region) get dropped.
            keep_all_outputs=streaming_buffer.is_buffer_empty(),
            drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                model, step_num, pad_and_drop_preencoded
            ),
        )
        cache = step_out.cache
        for level, logits in step_out.logits.items():
            all_logits[level].append(logits)

    streaming_buffer.reset_buffer()

    # Concatenate logits across steps: list[(B, T_i, V)] -> (B, T_total, V)
    return {level: torch.cat(tensors, dim=1) for level, tensors in all_logits.items()}


__all__ = [
    "infer_fastconformer_streaming",
]
