"""Debug the SeamlessM4T processor normalization mismatch.

Compare exact per-frame values to detect any shift between:
  A) processor(chunk_of_11440_samples)   → 35 features
  B) processor(full_audio)               → N features, slice first 35

With do_normalize_per_mel_bins=False, the raw log-mel features should be IDENTICAL.
"""

import numpy as np
import torch
from transformers import AutoFeatureExtractor
from librosa import load


def calc_frames(L, W=400, H=160, S=2) -> int:
    return max(0, int(1 + np.floor((L - W) / H)) // S)


def main():
    audio_path = "/home/abdullah/Downloads/alif-madd.ogg"
    audio, sr = load(audio_path, sr=16000)
    print(f"Audio: {len(audio)} samples, {len(audio) / 16000:.2f}s")

    seg_len = 11440  # 35 features
    print(f"Segment length: {seg_len} → {calc_frames(seg_len)} features\n")

    proc = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    # ── Test 1: WITH normalization (default) ──────────────────────────
    print("=" * 90)
    print("TEST 1: WITH normalization (do_normalize_per_mel_bins=True)")
    print("=" * 90)

    full = proc(audio, sampling_rate=16000, return_tensors="pt")
    full_feats = full["input_features"][0]

    chunk = proc(audio[:seg_len], sampling_rate=16000, return_tensors="pt")
    chunk_feats = chunk["input_features"][0]

    ref_feats = full_feats[:35]

    print(
        f"{'frame':>5} | {'chunk mean':>10} {'ref mean':>10} {'diff mean':>10} {'diff max':>10} | first 5 diff values"
    )
    print("-" * 90)
    for i in range(35):
        cf = chunk_feats[i]
        rf = ref_feats[i]
        d = cf - rf
        d5 = d[:5].tolist()
        d5_str = ", ".join(f"{v:+7.4f}" for v in d5)
        print(
            f"{i:>5} | {cf.mean():>10.4f} {rf.mean():>10.4f} {d.mean():>+10.4f} {d.abs().max():>10.4f} | {d5_str}"
        )

    print()
    print(f"Full stats:     mean={full_feats.mean():>8.4f} std={full_feats.std():.4f}")
    print(
        f"Chunk stats:    mean={chunk_feats.mean():>8.4f} std={chunk_feats.std():.4f}"
    )
    print(f"Ref [0:35]:     mean={ref_feats.mean():>8.4f} std={ref_feats.std():.4f}")
    print()

    # ── Test 2: WITHOUT normalization ─────────────────────────────────
    print("=" * 90)
    print("TEST 2: WITHOUT normalization (do_normalize_per_mel_bins=False)")
    print("=" * 90)

    full_raw = proc(
        audio, sampling_rate=16000, return_tensors="pt", do_normalize_per_mel_bins=False
    )
    full_raw_feats = full_raw["input_features"][0]

    chunk_raw = proc(
        audio[:seg_len],
        sampling_rate=16000,
        return_tensors="pt",
        do_normalize_per_mel_bins=False,
    )
    chunk_raw_feats = chunk_raw["input_features"][0]

    ref_raw_feats = full_raw_feats[:35]

    print(torch.allclose(ref_raw_feats, chunk_raw_feats))

    print(
        f"{'frame':>5} | {'chunk mean':>10} {'ref mean':>10} {'diff mean':>10} {'diff max':>10} | first 5 diff values"
    )
    print("-" * 90)
    for i in range(35):
        cf = chunk_raw_feats[i]
        rf = ref_raw_feats[i]
        d = cf - rf
        d5 = d[:5].tolist()
        d5_str = ", ".join(f"{v:+.2e}" for v in d5)
        print(
            f"{i:>5} | {cf.mean():>10.4f} {rf.mean():>10.4f} {d.mean():>+10.4f} {d.abs().max():>10.4f} | {d5_str}"
        )

    print()
    print(
        f"Full raw stats:     mean={full_raw_feats.mean():>8.4f} std={full_raw_feats.std():.4f}"
    )
    print(
        f"Chunk raw stats:    mean={chunk_raw_feats.mean():>8.4f} std={chunk_raw_feats.std():.4f}"
    )
    print(
        f"Ref raw [0:35]:     mean={ref_raw_feats.mean():>8.4f} std={ref_raw_feats.std():.4f}"
    )
    print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    d_norm = (chunk_feats - ref_feats).abs()
    d_raw = (chunk_raw_feats - ref_raw_feats).abs()
    print(f"Normalized: max_diff={d_norm.max():.6f} mean_diff={d_norm.mean():.6f}")
    print(f"Raw:        max_diff={d_raw.max():.2e} mean_diff={d_raw.mean():.2e}")
    print(
        "⇒ Normalization is the sole source of mismatch."
        if d_raw.max() < 1e-5
        else "⇒ Unexpected raw difference!"
    )


if __name__ == "__main__":
    main()
