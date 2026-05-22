"""
prune_w2v2_width.py
====================
Structured width pruning for Wav2Vec2-XLS-R-300M (omnilingual encoder).

Loads config from facebook/wav2vec2-large-xlsr-53 (with overrides from
test_load_omnilingual_encoder_300.py) and weights from
facebook/wav2vec2-xls-r-300m.

Prunes hidden_size, FFN intermediate_size, and num_attention_heads
from 1024->384, 4096->1536, 16->6 by default using weight-magnitude
importance scoring. No calibration data required.

Output is a HuggingFace-compatible model (config.json + model.safetensors)
loadable via Wav2Vec2ForPreTraining.from_pretrained().

Usage:
    uv run python prune_w2v2_width.py --output-dir ./pruned-model
    uv run python prune_w2v2_width.py --hf-token hf_... --push-to-hub user/repo
"""

import argparse
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
W2V_ORIG_HEADS = 16
W2V_ORIG_FFN = 4096
W2V_ORIG_HIDDEN = 1024
W2V_HEAD_DIM = 64  # hidden_size / num_heads = 1024 / 16

# Config overrides matching test_load_omnilingual_encoder_300.py
_CONFIG_OVERRIDES = {
    "attention_dropout": 0,
    "mask_time_prob": 0.0,
    "layerdrop": 0.0,
    "ctc_loss_reduction": "mean",
    "add_adapter": False,
    "adapter_stride": 1,
}

_LAYER_KEY_PREFIX = "wav2vec2.encoder.layers."
_LAYER_KEY_IDX_POS = 3

_PRETRAIN_HEAD_PREFIXES = ("quantizer.", "project_hid.", "project_q.")

_HIDDEN_ROW_TAGS = (
    "feature_projection.projection",
    "attention.out_proj",
    "feed_forward.output_dense",
    "layer_norm",
    "final_layer_norm",
    "encoder.layer_norm",
    "masked_spec_embed",
)

_HIDDEN_COL_TAGS = (
    "attention.q_proj",
    "attention.k_proj",
    "attention.v_proj",
    "feed_forward.intermediate_dense",
)

_HIDDEN_SKIP_TAGS = (
    "feature_extractor",
    "feature_projection.layer_norm",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _layer_prefixes(keys):
    seen, out = set(), []
    for k in keys:
        if k.startswith(_LAYER_KEY_PREFIX):
            p = ".".join(k.split(".")[:4])
            if p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out, key=lambda x: int(x.rsplit(".", 1)[-1]))


def _find_ffn_keys(state, layer_prefix):
    w1_key = f"{layer_prefix}.feed_forward.intermediate_dense.weight"
    w2_key = f"{layer_prefix}.feed_forward.output_dense.weight"
    if w1_key in state and w2_key in state:
        return w1_key, w2_key
    candidates_w1, candidates_w2 = [], []
    for k in state.keys():
        if not k.startswith(layer_prefix + "."):
            continue
        if "attention" in k or "layer_norm" in k:
            continue
        if not k.endswith(".weight"):
            continue
        w = state[k]
        if w.ndim == 2:
            if w.shape[0] > w.shape[1]:
                candidates_w1.append(k)
            elif w.shape[1] > w.shape[0]:
                candidates_w2.append(k)
    if not candidates_w1 or not candidates_w2:
        raise KeyError(f"Could not find FFN weight keys under '{layer_prefix}'.")
    w1_key = max(candidates_w1, key=lambda k: state[k].shape[0])
    w2_key = max(candidates_w2, key=lambda k: state[k].shape[1])
    return w1_key, w2_key


# ---------------------------------------------------------------------------
# Download & load
# ---------------------------------------------------------------------------
def _download_config(config_id: str, tmp_dir: Path):
    from huggingface_hub import hf_hub_download

    for fname in ("config.json", "preprocessor_config.json"):
        dest = tmp_dir / fname
        if not dest.exists():
            hf_hub_download(repo_id=config_id, filename=fname, local_dir=str(tmp_dir))


def _download_weights(model_id: str, tmp_dir: Path):
    from huggingface_hub import hf_hub_download

    safetensors_path = tmp_dir / "model.safetensors"
    pytorch_path = tmp_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        return "model.safetensors"
    if pytorch_path.exists():
        return "pytorch_model.bin"

    try:
        hf_hub_download(
            repo_id=model_id, filename="model.safetensors", local_dir=str(tmp_dir)
        )
        return "model.safetensors"
    except Exception:
        hf_hub_download(
            repo_id=model_id, filename="pytorch_model.bin", local_dir=str(tmp_dir)
        )
        return "pytorch_model.bin"


def _load_weights(tmp_dir: Path, weights_file: str, strip_pretrain: bool):
    weights_path = tmp_dir / weights_file

    if weights_file.endswith(".safetensors"):
        from safetensors.numpy import load_file

        state = load_file(str(weights_path))
    else:
        import torch

        sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        state = {k: v.numpy() for k, v in sd.items()}
        del sd
        gc.collect()

    if strip_pretrain:
        for k in list(state.keys()):
            if any(k.startswith(pfx) for pfx in _PRETRAIN_HEAD_PREFIXES):
                del state[k]

    return state


# ---------------------------------------------------------------------------
# Pruning: Attention heads
# ---------------------------------------------------------------------------
def prune_heads(state, target):
    prefixes = _layer_prefixes(state.keys())
    if not prefixes:
        return state

    q0 = state[f"{prefixes[0]}.attention.q_proj.weight"]
    head_dim = W2V_HEAD_DIM
    old_heads = q0.shape[0] // head_dim

    if target >= old_heads:
        print(f"  Heads {old_heads} <= target {target} -- skipping")
        return state

    print(f"  Pruning heads: {old_heads} -> {target}")

    for layer in prefixes:
        scores = sorted(
            [
                (
                    h,
                    float(
                        np.linalg.norm(
                            state[f"{layer}.attention.q_proj.weight"][
                                h * head_dim : (h + 1) * head_dim
                            ]
                        )
                        + np.linalg.norm(
                            state[f"{layer}.attention.k_proj.weight"][
                                h * head_dim : (h + 1) * head_dim
                            ]
                        )
                        + np.linalg.norm(
                            state[f"{layer}.attention.v_proj.weight"][
                                h * head_dim : (h + 1) * head_dim
                            ]
                        )
                    ),
                )
                for h in range(old_heads)
            ],
            key=lambda x: x[1],
        )
        keep = sorted(h for h, _ in scores[(old_heads - target) :])
        rows = [i for h in keep for i in range(h * head_dim, (h + 1) * head_dim)]

        for proj in ("q_proj", "k_proj", "v_proj"):
            w_key = f"{layer}.attention.{proj}.weight"
            b_key = f"{layer}.attention.{proj}.bias"
            state[w_key] = state[w_key][rows, :]
            if b_key in state:
                state[b_key] = state[b_key][rows]

        out_w = f"{layer}.attention.out_proj.weight"
        state[out_w] = state[out_w][:, rows]

    return state


# ---------------------------------------------------------------------------
# Pruning: FFN intermediate size
# ---------------------------------------------------------------------------
def prune_ffn(state, target):
    prefixes = _layer_prefixes(state.keys())
    if not prefixes:
        return state

    w1_key_0, w2_key_0 = _find_ffn_keys(state, prefixes[0])
    cur = state[w1_key_0].shape[0]

    if target >= cur:
        print(f"  FFN dim {cur} <= target {target} -- skipping")
        return state

    p0 = prefixes[0] + "."
    w1_subpath = w1_key_0[len(p0) :]
    w2_subpath = w2_key_0[len(p0) :]
    b1_subpath = w1_subpath.replace(".weight", ".bias")

    print(f"  Pruning FFN: {cur} -> {target}")

    for layer in prefixes:
        w1 = state[f"{layer}.{w1_subpath}"]
        w2 = state[f"{layer}.{w2_subpath}"]
        scores = np.linalg.norm(w1, axis=1) * np.linalg.norm(w2, axis=0)
        keep = np.sort(np.argsort(scores)[-target:])
        state[f"{layer}.{w1_subpath}"] = w1[keep, :]
        b1_key = f"{layer}.{b1_subpath}"
        if b1_key in state:
            state[b1_key] = state[b1_key][keep]
        state[f"{layer}.{w2_subpath}"] = w2[:, keep]

    return state


# ---------------------------------------------------------------------------
# Pruning: Hidden size
# ---------------------------------------------------------------------------
def _discover_hidden_size(state):
    for k, v in state.items():
        if "feature_projection.projection.weight" in k:
            return int(v.shape[0])
    for k, v in state.items():
        if k.startswith(_LAYER_KEY_PREFIX) and "attention.q_proj.weight" in k:
            return int(v.shape[1])
    raise RuntimeError("Could not infer hidden_size from state dict")


def _hidden_axis(key, v_shape, hidden_size):
    if any(t in key for t in _HIDDEN_SKIP_TAGS):
        return "none"
    is_row = any(t in key for t in _HIDDEN_ROW_TAGS)
    is_col = any(t in key for t in _HIDDEN_COL_TAGS)
    if is_row and len(v_shape) >= 1 and v_shape[0] == hidden_size:
        return "row"
    if is_col and len(v_shape) >= 2 and v_shape[-1] == hidden_size:
        return "col"
    return "none"


def _score_hidden_dims(state, hidden_size):
    scores = np.zeros(hidden_size, dtype=np.float64)
    for k, v in state.items():
        axis = _hidden_axis(k, v.shape, hidden_size)
        if axis == "row" and v.ndim >= 1 and v.shape[0] == hidden_size:
            scores += np.linalg.norm(v.reshape(hidden_size, -1), axis=1)
        elif axis == "col" and v.ndim >= 2 and v.shape[-1] == hidden_size:
            scores += np.linalg.norm(v.reshape(-1, hidden_size), axis=0)
    return scores.astype(np.float32)


def _balanced_keep_selection(scores, hidden_size, target, groups):
    per_group_old = hidden_size // groups
    per_group_new = target // groups
    keep = []
    for g in range(groups):
        start = g * per_group_old
        group_scores = scores[start : start + per_group_old]
        group_keep = np.argsort(group_scores)[-per_group_new:]
        keep.extend(start + np.sort(group_keep))
    return np.array(sorted(keep), dtype=np.intp)


def prune_hidden_size(state, target, num_conv_pos_groups=16):
    current = _discover_hidden_size(state)
    if target >= current:
        print(f"  Hidden size {current} <= target {target} -- skipping")
        return state

    scores = _score_hidden_dims(state, current)
    keep = _balanced_keep_selection(scores, current, target, num_conv_pos_groups)

    per_group_old = current // num_conv_pos_groups
    per_group_new = target // num_conv_pos_groups
    group_local_keeps = []
    for g in range(num_conv_pos_groups):
        start = g * per_group_old
        local = np.array(
            [k - start for k in keep if start <= k < start + per_group_old],
            dtype=np.intp,
        )
        group_local_keeps.append(local)

    print(f"  Pruning hidden size: {current} -> {target}")

    new_state = {}
    n_row = n_col = n_pos = 0

    for k, v in state.items():
        if "pos_conv_embed.conv" in k:
            if k.endswith(".weight_g") or k.endswith(
                "parametrizations.weight.original0"
            ):
                new_state[k] = v
                n_pos += 1
                continue

            if k.endswith(".weight_v") or k.endswith(
                "parametrizations.weight.original1"
            ):
                assert v.shape == (current, per_group_old, v.shape[2])
                out = np.empty((target, per_group_new, v.shape[2]), dtype=v.dtype)
                row_cursor = 0
                for g in range(num_conv_pos_groups):
                    out_start_old = g * per_group_old
                    out_keep_global = [
                        idx
                        for idx in keep
                        if out_start_old <= idx < out_start_old + per_group_old
                    ]
                    rows_sliced = v[out_keep_global, :, :]
                    cols_sliced = rows_sliced[:, group_local_keeps[g], :]
                    out[row_cursor : row_cursor + per_group_new] = cols_sliced
                    row_cursor += per_group_new
                new_state[k] = out
                n_pos += 1
                continue

            if k.endswith("conv.bias"):
                new_state[k] = v[keep]
                n_pos += 1
                continue

            if k.endswith("conv.weight"):
                assert v.shape == (current, per_group_old, v.shape[2])
                out = np.empty((target, per_group_new, v.shape[2]), dtype=v.dtype)
                row_cursor = 0
                for g in range(num_conv_pos_groups):
                    out_start_old = g * per_group_old
                    out_keep_global = [
                        idx
                        for idx in keep
                        if out_start_old <= idx < out_start_old + per_group_old
                    ]
                    rows_sliced = v[out_keep_global, :, :]
                    cols_sliced = rows_sliced[:, group_local_keeps[g], :]
                    out[row_cursor : row_cursor + per_group_new] = cols_sliced
                    row_cursor += per_group_new
                new_state[k] = out
                n_pos += 1
                continue

            new_state[k] = v
            continue

        axis = _hidden_axis(k, tuple(v.shape), current)
        if axis == "row":
            n_row += 1
            new_state[k] = v[keep] if v.ndim == 1 else v[keep, :]
        elif axis == "col":
            n_col += 1
            new_state[k] = v[:, keep]
        else:
            new_state[k] = v

    print(f"    sliced: {n_row} row, {n_col} col, {n_pos} pos_conv_embed tensors")
    return new_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_pruning(cfg: dict):
    from safetensors.numpy import save_file

    config_id = cfg["config_id"]
    model_id = cfg["model_id"]
    output_dir = cfg["output_dir"]
    tmp_dir = output_dir / "_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    head_target = cfg["num_attention_heads"]
    ffn_target = cfg["intermediate_size"]
    hidden_target = cfg["hidden_size"]

    print("=" * 60)
    print(f"  Pruning Wav2Vec2")
    print(f"  Config:  {config_id}")
    print(f"  Weights: {model_id}")
    print(f"  hidden_size: {W2V_ORIG_HIDDEN} -> {hidden_target}")
    print(f"  intermediate_size: {W2V_ORIG_FFN} -> {ffn_target}")
    print(f"  num_attention_heads: {W2V_ORIG_HEADS} -> {head_target}")
    print("=" * 60)

    # Download config + processor from config-id, weights from model-id
    print(f"\nDownloading config from {config_id}...")
    _download_config(config_id, tmp_dir)

    print(f"Downloading weights from {model_id}...")
    weights_file = _download_weights(model_id, tmp_dir)

    # Load
    print("Loading weights...")
    t0 = time.time()
    state = _load_weights(tmp_dir, weights_file, cfg["strip_pretrain_heads"])
    orig_params = sum(v.size for v in state.values())
    print(
        f"  Loaded {len(state)} tensors / {orig_params / 1e6:.1f}M params ({time.time() - t0:.1f}s)"
    )

    # Prune
    t0 = time.time()
    if hidden_target < W2V_ORIG_HIDDEN:
        state = prune_hidden_size(state, hidden_target)
    else:
        print("\n  Hidden size unchanged -- skipping")

    if ffn_target < W2V_ORIG_FFN:
        state = prune_ffn(state, ffn_target)
    else:
        print("  FFN unchanged -- skipping")

    if head_target < W2V_ORIG_HEADS:
        state = prune_heads(state, head_target)
    else:
        print("  Heads unchanged -- skipping")

    pruned_params = sum(v.size for v in state.values())
    reduction = (1 - pruned_params / orig_params) * 100
    print(
        f"\n  {orig_params / 1e6:.1f}M -> {pruned_params / 1e6:.1f}M params ({reduction:.1f}% reduction) [{time.time() - t0:.1f}s]"
    )

    # Update config
    with open(tmp_dir / "config.json") as f:
        cfg_json = json.load(f)

    cfg_json["_original_num_attention_heads"] = cfg_json.get(
        "num_attention_heads", W2V_ORIG_HEADS
    )
    cfg_json["num_attention_heads"] = head_target
    cfg_json["intermediate_size"] = ffn_target
    cfg_json["hidden_size"] = hidden_target
    cfg_json.update(_CONFIG_OVERRIDES)

    # Drop masked_spec_embed from state if masking is disabled (avoids UNEXPECTED warning)
    if not cfg_json.get("mask_time_prob") and not cfg_json.get("mask_feature_prob"):
        state.pop("wav2vec2.masked_spec_embed", None)

    (output_dir / "config.json").write_text(
        json.dumps(cfg_json, indent=2, ensure_ascii=False)
    )

    # Save weights
    print("\nSaving pruned model...")
    save_file(state, str(output_dir / "model.safetensors"), metadata={"format": "pt"})

    pp_cfg = tmp_dir / "preprocessor_config.json"
    if pp_cfg.exists():
        shutil.copy2(str(pp_cfg), str(output_dir / "preprocessor_config.json"))

    # Summary
    (output_dir / "pruning_summary.json").write_text(
        json.dumps(
            {
                "config_source": config_id,
                "weight_source": model_id,
                "original_params": int(orig_params),
                "pruned_params": int(pruned_params),
                "reduction_pct": round(reduction, 2),
                "hidden_size": hidden_target,
                "intermediate_size": ffn_target,
                "num_attention_heads": head_target,
                "original_hidden": W2V_ORIG_HIDDEN,
                "original_intermediate": W2V_ORIG_FFN,
                "original_heads": W2V_ORIG_HEADS,
            },
            indent=2,
        )
    )

    size_mb = (output_dir / "model.safetensors").stat().st_size / 1e6
    print(f"  Saved model.safetensors ({size_mb:.1f} MB)")
    print(f"  Output: {output_dir}")

    # Cleanup
    shutil.rmtree(str(tmp_dir))
    del state
    gc.collect()

    print(f"\nPruning complete!")
    print(
        f"  {orig_params / 1e6:.1f}M -> {pruned_params / 1e6:.1f}M params ({reduction:.1f}% reduction)"
    )

    return output_dir


def push_to_hub(output_dir: Path, repo_id: str, hf_token: str):
    from huggingface_hub import HfApi

    print(f"\nPushing to {repo_id}...")
    api = HfApi(token=hf_token)

    api.create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=False,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        token=hf_token,
        path_in_repo=".",
        delete_patterns=["*"],
    )
    print(f"  Pushed to https://huggingface.co/{repo_id}")


def main():
    p = argparse.ArgumentParser(
        description="Prune Wav2Vec2-XLS-R-300M width (hidden_size, FFN, heads)"
    )

    p.add_argument(
        "--config-id",
        default="facebook/wav2vec2-large-xlsr-53",
        help="Source for config.json + preprocessor_config.json",
    )
    p.add_argument(
        "--model-id",
        default="facebook/wav2vec2-xls-r-300m",
        help="Source for model weights",
    )
    p.add_argument(
        "--output-dir", default="./pruned_w2v2", type=Path, help="Output directory"
    )
    p.add_argument(
        "--hidden-size",
        type=int,
        default=384,
        help="Target hidden size (must be divisible by heads and 16)",
    )
    p.add_argument(
        "--intermediate-size",
        type=int,
        default=1536,
        help="Target FFN intermediate size",
    )
    p.add_argument(
        "--num-attention-heads", type=int, default=6, help="Target attention heads"
    )
    p.add_argument("--device", default="cpu", help="Device for tensor ops")
    p.add_argument(
        "--hf-token", default=None, help="HuggingFace token (for push to hub)"
    )
    p.add_argument(
        "--push-to-hub", default=None, help="Repository ID to push to (e.g. user/model)"
    )
    p.add_argument(
        "--strip-pretrain-heads",
        action="store_true",
        default=True,
        help="Remove quantizer/project heads (default: True)",
    )
    p.add_argument(
        "--no-strip-pretrain-heads", dest="strip_pretrain_heads", action="store_false"
    )

    args = p.parse_args()

    # Validate
    if args.hidden_size % args.num_attention_heads != 0:
        nearest = args.num_attention_heads * (
            args.hidden_size // args.num_attention_heads
        )
        p.error(
            f"--hidden-size {args.hidden_size} must be divisible by "
            f"--num-attention-heads {args.num_attention_heads}. Try {nearest}."
        )
    if args.hidden_size % 16 != 0:
        p.error(
            f"--hidden-size {args.hidden_size} must be divisible by 16 "
            f"(num_conv_pos_embedding_groups)."
        )
    if args.hidden_size > W2V_ORIG_HIDDEN:
        p.error(f"--hidden-size must be <= {W2V_ORIG_HIDDEN}")

    cfg = vars(args)
    output_dir = run_pruning(cfg)

    if args.push_to_hub:
        if not args.hf_token:
            p.error("--hf-token is required when using --push-to-hub")
        push_to_hub(output_dir, args.push_to_hub, args.hf_token)


if __name__ == "__main__":
    main()
