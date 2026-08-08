"""Load the FastConformer encoder from a NVIDIA NeMo checkpoint on the HF Hub.

Downloads ``nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`` (a NeMo
``.nemo`` file) and initialises :class:`FastConformerCacheAwareMultilevelCTC`
from it via :meth:`FastConformerCacheAwareMultilevelCTC.from_nemo`, using the
project's streaming cache-aware configuration.  Only the encoder weights are
transferred; the rest of the checkpoint is ignored (with warnings).  The
script then verifies that every matched encoder weight equals the checkpoint
weight and runs a quick offline forward + streaming setup.
"""

from __future__ import annotations

import json
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTC,
    FastConformerCacheAwareMultilevelCTCConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_NAME = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"


def build_streaming_config() -> FastConformerCacheAwareMultilevelCTCConfig:
    """Streaming cache-aware config, mirroring ``test_modeling_fastconformer_cache_aware.py``."""
    vocab_path = REPO_ROOT / "vocab_streaming" / "vocab.json"
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}
    print(f"Level to vocab size: {level_to_vocab_size}")

    return FastConformerCacheAwareMultilevelCTCConfig(
        att_context_size=[78, 12, 5],
        level_to_vocab_size=level_to_vocab_size,
        level_to_loss_weight={"phonemes": 0.5, "hams_or_jahr": 0.2},
    )


def verify_encoder_weights(
    model: FastConformerCacheAwareMultilevelCTC,
) -> None:
    """Assert every matched encoder weight equals the NeMo checkpoint's."""
    nemo_model = nemo_asr.models.ASRModel.from_pretrained(
        PRETRAINED_NAME, map_location=torch.device("cpu")
    )
    nemo_sd = nemo_model.state_dict()
    del nemo_model

    our_sd = model.state_dict()
    encoder_keys = [k for k in nemo_sd if k.startswith("encoder.")]
    matched = [
        k for k in encoder_keys if k in our_sd and our_sd[k].shape == nemo_sd[k].shape
    ]
    mismatched = [
        k for k in encoder_keys if k in our_sd and our_sd[k].shape != nemo_sd[k].shape
    ]
    unmatched = [k for k in encoder_keys if k not in our_sd]

    assert matched, "No encoder weights were matched by from_nemo!"
    for k in matched:
        assert torch.equal(our_sd[k], nemo_sd[k]), (
            f"Encoder weight {k} differs from the NeMo checkpoint!"
        )

    print(
        f"\nVerified {len(matched)}/{len(encoder_keys)} encoder weights "
        f"loaded exactly from the NeMo checkpoint."
    )
    print(f"  not matched (different shape), left randomly initialised: {mismatched}")
    print(
        f"  not used (no counterpart in our model): {unmatched[:4]} "
        f"{'...' if len(unmatched) > 4 else ''} ({len(unmatched)} total)"
    )


def main() -> None:
    # ---- 1. Streaming config ----
    config = build_streaming_config()
    print(config)

    # ---- 2. Load the encoder weights from the NeMo checkpoint ----
    model = FastConformerCacheAwareMultilevelCTC.from_nemo(
        PRETRAINED_NAME, config, map_location="cpu"
    )
    model.eval()
    print(f"\nModel: {type(model).__name__}")
    print(f"  encoder layers: {model.encoder.n_layers}")
    print(f"  levels: {list(model.level_to_lm_head.keys())}")

    # ---- 3. Verify the transferred weights ----
    verify_encoder_weights(model)

    # ---- 4. Offline forward sanity check ----
    batch_size = 2
    wav_tensor = torch.randn(batch_size, 16000, dtype=torch.float32)
    wav_lengths = torch.full((batch_size,), 16000, dtype=torch.long)

    with torch.no_grad():
        out = model(raw_audio=wav_tensor, audio_length=wav_lengths)
    print("\nOffline inference (no labels):")
    print(f"  logits keys: {list(out.logits.keys())}")
    print(f"  encoder_output shape: {out.encoder_output.shape}")
    print(f"  encoder_lengths: {out.encoder_lengths}")

    # ---- 5. Streaming setup sanity check ----
    model.setup_streaming_params()
    cfg = model.encoder.streaming_cfg
    print("\nStreaming params:")
    print(f"  chunk_size: {cfg.chunk_size}")
    print(f"  shift_size: {cfg.shift_size}")
    print(f"  valid_out_len: {cfg.valid_out_len}")
    cache = model.get_initial_cache(batch_size=batch_size)
    print(f"  initial cache.last_channel: {cache.last_channel.shape}")
    print(f"  initial cache.last_time: {cache.last_time.shape}")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
