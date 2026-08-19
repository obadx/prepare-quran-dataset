"""Checks for ``FastConformerCacheAwareMultilevelCTCConfig.fp32_pre_encode``.

The flag makes the convolutional subsampling front-end (``encoder.pre_encode``)
compute in float32 even while the rest of the model runs under bf16 autocast —
the precision setup used by ``train_streaming.py`` (``TrainingArguments(bf16=True)``).
The front-end's output is cast back to the autocast dtype, so everything
downstream must behave exactly as it does with the flag off.

Run directly::

    python tests/test_fastconformer_fp32_pre_encode.py
"""

import tempfile
from pathlib import Path

import torch

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTC,
    FastConformerCacheAwareMultilevelCTCConfig,
)

LEVEL_TO_VOCAB_SIZE = {"phonemes": 20, "hams_or_jahr": 4}
BATCH_SIZE = 2
AUDIO_SAMPLES = 32000  # 2 seconds @ 16 kHz -> ~200 mel frames -> ~50 encoder frames
TARGET_LEN = 8


def build_model(fp32_pre_encode: bool, device: torch.device, seed: int = 0):
    """Build a tiny FastConformer model; identical weights for a given seed."""
    torch.manual_seed(seed)
    config = FastConformerCacheAwareMultilevelCTCConfig(
        level_to_vocab_size=LEVEL_TO_VOCAB_SIZE,
        level_to_loss_weight={"phonemes": 0.6},
        n_layers=2,
        d_model=64,
        n_heads=4,
        subsampling_conv_channels=32,
        att_context_size=[9, 2],
        fp32_pre_encode=fp32_pre_encode,
    )
    model = FastConformerCacheAwareMultilevelCTC(config).to(device)
    model.eval()
    return config, model


def hook_pre_encode(model):
    """Record the dtypes seen *inside* ``pre_encode`` (before the cast-back)."""
    seen = {}

    def _hook(module, args, kwargs, output):
        tensor = kwargs.get("x", args[0] if args else None)
        seen["input_dtype"] = tensor.dtype
        seen["output_dtype"] = (output[0] if isinstance(output, tuple) else output).dtype

    handle = model.encoder.pre_encode.register_forward_hook(_hook, with_kwargs=True)
    return seen, handle


def make_batch(device: torch.device):
    torch.manual_seed(1234)
    raw_audio = torch.randn(BATCH_SIZE, AUDIO_SAMPLES, device=device)
    audio_length = torch.full((BATCH_SIZE,), AUDIO_SAMPLES, dtype=torch.long, device=device)
    labels = {
        level: torch.randint(2, size, (BATCH_SIZE, TARGET_LEN), device=device)
        for level, size in LEVEL_TO_VOCAB_SIZE.items()
    }
    labels_mask = {
        level: torch.ones_like(ids, dtype=torch.bool) for level, ids in labels.items()
    }
    return raw_audio, audio_length, labels, labels_mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16
    print(f"device: {device}, autocast dtype: {autocast_dtype}")

    raw_audio, audio_length, labels, labels_mask = make_batch(device)

    # ---- 1. Flag off: pre_encode runs in the autocast dtype (today's behaviour) ----
    _, model_off = build_model(fp32_pre_encode=False, device=device)
    seen_off, handle = hook_pre_encode(model_off)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        with torch.no_grad():
            out_off = model_off(raw_audio=raw_audio, audio_length=audio_length)
    handle.remove()
    print("\n--- flag off (baseline) ---")
    print(f"  pre_encode output dtype: {seen_off['output_dtype']}")
    print(f"  encoder_output dtype:    {out_off.encoder_output.dtype}")
    assert seen_off["output_dtype"] == autocast_dtype, (
        f"Baseline pre_encode should run in {autocast_dtype}, got {seen_off['output_dtype']}"
    )

    # ---- 2. Flag on: pre_encode runs in fp32, output handed back in bf16 ----
    _, model_on = build_model(fp32_pre_encode=True, device=device)
    seen_on, handle = hook_pre_encode(model_on)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        with torch.no_grad():
            out_on = model_on(raw_audio=raw_audio, audio_length=audio_length)
    handle.remove()
    print("\n--- flag on ---")
    print(f"  pre_encode input dtype:  {seen_on['input_dtype']}")
    print(f"  pre_encode output dtype: {seen_on['output_dtype']}")
    print(f"  encoder_output dtype:    {out_on.encoder_output.dtype}")
    assert seen_on["input_dtype"] == torch.float32, "pre_encode input must be fp32"
    assert seen_on["output_dtype"] == torch.float32, "pre_encode must compute in fp32"
    assert out_on.encoder_output.dtype == out_off.encoder_output.dtype, (
        "The encoder output dtype must be unchanged by the flag: "
        f"{out_on.encoder_output.dtype} vs {out_off.encoder_output.dtype}"
    )

    # Same seed -> same weights, so the two runs differ only by the front-end precision.
    diff = (
        out_on.encoder_output.float() - out_off.encoder_output.float()
    ).abs().max().item()
    print(f"  max |encoder_output(on) - encoder_output(off)|: {diff:.6f}")

    # ---- 3. Outside autocast both paths stay in fp32 ----
    seen_on_fp32, handle = hook_pre_encode(model_on)
    with torch.no_grad():
        out_plain = model_on(raw_audio=raw_audio, audio_length=audio_length)
    handle.remove()
    print("\n--- flag on, no autocast ---")
    print(f"  pre_encode output dtype: {seen_on_fp32['output_dtype']}")
    print(f"  encoder_output dtype:    {out_plain.encoder_output.dtype}")
    assert seen_on_fp32["output_dtype"] == torch.float32
    assert out_plain.encoder_output.dtype == torch.float32, (
        "Without autocast nothing may be downcast"
    )

    # ---- 4. Training step: finite loss + gradients through the fp32 front-end ----
    model_on.train()
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        out_train = model_on(
            raw_audio=raw_audio,
            audio_length=audio_length,
            labels=labels,
            labels_mask=labels_mask,
        )
    print("\n--- training step (flag on) ---")
    print(f"  loss: {out_train.loss.item():.4f} (dtype {out_train.loss.dtype})")
    assert out_train.loss is not None and out_train.loss.isfinite(), "Loss must be finite"
    out_train.loss.backward()

    grad_params = [
        (name, p)
        for name, p in model_on.encoder.pre_encode.named_parameters()
        if p.requires_grad
    ]
    assert grad_params, "pre_encode has no trainable parameters?"
    for name, p in grad_params:
        assert p.grad is not None, f"No gradient for pre_encode.{name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for pre_encode.{name}"
        assert p.dtype == torch.float32, (
            f"pre_encode.{name} weights must stay fp32 masters, got {p.dtype}"
        )
    print(f"  gradients flow to all {len(grad_params)} pre_encode params: yes")
    model_on.zero_grad(set_to_none=True)
    model_on.eval()

    # ---- 5. Config round-trip through save_pretrained / from_pretrained ----
    with tempfile.TemporaryDirectory() as tmpdir:
        model_on.save_pretrained(tmpdir)
        reloaded = FastConformerCacheAwareMultilevelCTC.from_pretrained(tmpdir).to(device)
        print("\n--- config round-trip ---")
        print(f"  saved config.json has fp32_pre_encode: {reloaded.config.fp32_pre_encode}")
        assert reloaded.config.fp32_pre_encode is True
        assert reloaded.encoder.fp32_pre_encode is True, (
            "The flag must be re-propagated onto the encoder after from_pretrained"
        )
        assert Path(tmpdir, "config.json").exists()

    # ---- 6. Streaming step under autocast (guards the cache concatenations) ----
    model_on.setup_streaming_params()
    cfg = model_on.encoder.streaming_cfg
    chunk = cfg.chunk_size[1] if isinstance(cfg.chunk_size, list) else cfg.chunk_size
    cache = model_on.get_initial_cache(batch_size=BATCH_SIZE)
    processed_signal = torch.randn(
        BATCH_SIZE, model_on.config.feat_in, chunk, device=device
    )
    processed_length = torch.full((BATCH_SIZE,), chunk, dtype=torch.long, device=device)

    seen_stream, handle = hook_pre_encode(model_on)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        with torch.no_grad():
            out_stream = model_on(
                processed_signal=processed_signal,
                processed_length=processed_length,
                cache=cache,
                keep_all_outputs=False,
                drop_extra_pre_encoded=0,
            )
    handle.remove()
    print("\n--- streaming step (flag on) ---")
    print(f"  chunk frames: {chunk}")
    print(f"  pre_encode output dtype: {seen_stream['output_dtype']}")
    print(f"  encoder_output: {tuple(out_stream.encoder_output.shape)} "
          f"({out_stream.encoder_output.dtype})")
    assert seen_stream["output_dtype"] == torch.float32
    assert out_stream.cache is not None, "Streaming step must return an updated cache"

    print("\nAll fp32_pre_encode checks passed ✅")


if __name__ == "__main__":
    main()
