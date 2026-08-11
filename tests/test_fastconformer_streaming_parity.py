"""Streaming-vs-offline parity for the FastConformer cache-aware model.

With plain ``chunked_limited`` attention (a two-element ``att_context_size``, so
``constant_lookahead_delay == 0``) a chunk's attention window ends at its own
boundary, so the receptive field does not grow with depth and cache-aware
streaming is *mathematically equivalent* to the offline forward.  The live
training config (``configs/train/streaming/train_config_fastconformer_v1.yml``)
uses ``att_context_size: [78, 12]``, i.e. exactly this regime.

A failure of :func:`test_streaming_matches_offline` therefore points at the
chunk plumbing — most likely the pre-encode left-context cache — rather than at
a model limitation.

Run with::

    uv run pytest tests/test_fastconformer_streaming_parity.py -v
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo.collections.asr")

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (  # noqa: E402
    FastConformerCacheAwareMultilevelCTC,
    FastConformerCacheAwareMultilevelCTCConfig,
    HFCacheAwareStreamingAudioBuffer,
    stream_inference,
)

SAMPLE_RATE = 16000
LEVELS = {"phonemes": 8, "ghonna": 4}

# Deliberately not a whole number of chunks: with att_context_size=[6, 2] the
# steady-state chunk is 4 + 4 * 2 = 12 mel frames, and ~1.13 s of audio gives
# ~114 mel frames -> the final chunk is a short tail that must be zero-padded
# up to the full width before it reaches the encoder.
NON_MULTIPLE_SAMPLES = 18_080


def _build_model(att_context_size):
    """Tiny CPU model with a deterministic (dither-free) mel front-end."""
    config = FastConformerCacheAwareMultilevelCTCConfig(
        level_to_vocab_size=dict(LEVELS),
        att_context_size=list(att_context_size),
        att_context_style="chunked_limited",
        n_layers=2,
        d_model=64,
        n_heads=2,
        subsampling_conv_channels=32,
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        dropout_att=0.0,
        # Dither must be off on both paths: the streaming buffer disables it for
        # determinism, so leaving it on offline would inject a mel difference
        # and make any parity comparison meaningless.
        processor_kwargs={
            "sample_rate": SAMPLE_RATE,
            "normalize": "NA",
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "features": 80,
            "n_fft": 512,
            "frame_splicing": 1,
            "dither": 0.0,
            "pad_to": 0,
        },
    )
    model = FastConformerCacheAwareMultilevelCTC(config)
    model.eval()
    model.setup_streaming_params()
    return model


def _random_audio(num_samples=NON_MULTIPLE_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(num_samples) * 0.1).astype(np.float32)


@torch.inference_mode()
def _offline_ids(model, audio):
    """Per-level frame-aligned argmax ids from a single full-utterance forward."""
    out = model(
        raw_audio=torch.from_numpy(audio).unsqueeze(0),
        audio_length=torch.tensor([len(audio)]),
    )
    valid = int(out.encoder_lengths[0])
    return {
        level: logits[0, :valid].argmax(dim=-1).tolist()
        for level, logits in out.logits.items()
    }


def test_streaming_handles_non_multiple_length():
    """A final partial chunk must not trip the strict `_create_masks` guard.

    ``MuaalemConformerEncoder._create_masks`` asserts an exact encoder input
    width during streaming. That guard is intentional, so the chunk loop is the
    thing that has to cope: it zero-pads the short tail chunk and lets
    ``processed_length`` mask the padding back out.
    """
    model = _build_model([6, 2])
    audio = _random_audio()

    ids = stream_inference(model, audio)

    assert set(ids) == set(LEVELS)
    assert len(ids["phonemes"]) > 0
    assert all(0 <= i < LEVELS["phonemes"] for i in ids["phonemes"])


def test_streaming_matches_offline():
    """Streaming must reproduce the offline forward when C == 0."""
    model = _build_model([6, 2])
    audio = _random_audio()

    streamed = stream_inference(model, audio)
    offline = _offline_ids(model, audio)

    for level in LEVELS:
        stream_ids, offline_ids = streamed[level], offline[level]
        assert abs(len(stream_ids) - len(offline_ids)) <= 2, (
            f"[{level}] frame count differs too much: "
            f"streaming={len(stream_ids)} offline={len(offline_ids)}"
        )

        n = min(len(stream_ids), len(offline_ids))
        assert n > 0
        agree = sum(a == b for a, b in zip(stream_ids[:n], offline_ids[:n])) / n
        assert agree >= 0.98, (
            f"[{level}] only {agree:.1%} of frames agree between streaming and "
            "offline. With att_context_size=[6, 2] these should be equivalent — "
            "suspect the pre-encode left-context cache in the chunk loop."
        )


def test_constant_lookahead_delay_runs():
    """`C > 0` is not expected to match offline — only to run and stay aligned.

    With a third ``att_context_size`` entry the offline mask lets lookahead
    compound across layers while the streaming path blanks the tail rows, so the
    two paths genuinely diverge. No config in the repo sets ``C > 0`` today; this
    guards future ones against a hard crash.
    """
    model = _build_model([6, 2, 2])
    audio = _random_audio()

    ids = stream_inference(model, audio)

    assert set(ids) == set(LEVELS)
    assert len(ids["phonemes"]) > 0
    assert len(ids["phonemes"]) == len(ids["ghonna"])


def test_buffer_pads_only_the_tail():
    """Every chunk but the last should already be full width."""
    model = _build_model([6, 2])
    buffer = HFCacheAwareStreamingAudioBuffer(model)
    buffer.append_audio(_random_audio())

    widths = [(step, chunk.size(-1)) for step, (chunk, _) in enumerate(buffer)]
    assert len(widths) > 1

    for step, width in widths[:-1]:
        assert width == buffer.expected_chunk_width(step), (
            f"step {step} is not full width; only the final chunk should be short"
        )

    last_step, last_width = widths[-1]
    assert last_width <= buffer.expected_chunk_width(last_step)
