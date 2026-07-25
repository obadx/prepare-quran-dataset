import json
from pathlib import Path

import torch
from librosa.core import load
from quran_transcript import quran_phonetizer, Aya, MoshafAttributes

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTCConfig,
    FastConformerCacheAwareMultilevelCTC,
    FastConformerCache,
)
from prepare_quran_dataset.modeling_fastconformer_cache_aware.multi_level_tokenizer import (
    MultiLevelTokenizer,
)
from prepare_quran_dataset.modeling_fastconformer_cache_aware.vocab import (
    build_quran_phoneme_script_vocab,
)


def main():
    # ---- 1. Setup vocab ----
    vocab_path = "./vocab_streaming/vocab.json"
    if not Path(vocab_path).exists():
        build_quran_phoneme_script_vocab(vocab_path)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}
    print(f"Level to vocab size: {level_to_vocab_size}")

    # ---- 2. Config ----
    config = FastConformerCacheAwareMultilevelCTCConfig(
        level_to_vocab_size=level_to_vocab_size,
        level_to_loss_weight={"phonemes": 0.5, "hams_or_jahr": 0.2},
    )
    print(config)

    # ---- 3. Model ----
    model = FastConformerCacheAwareMultilevelCTC(config)
    model.eval()
    print(f"Model: {type(model).__name__}")
    print(f"  processor: {type(model.processor).__name__}")
    print(f"  encoder layers: {model.encoder.n_layers}")
    print(f"  levels: {list(model.level_to_lm_head.keys())}")

    batch_size = 2

    # ---- 4. Offline inference without labels ----
    wav, _ = load("./assets/audio-sampels/test.wav", sr=16000, mono=True)
    wav_tensor = torch.tensor([wav] * batch_size, dtype=torch.float32)
    wav_lengths = torch.full((batch_size,), len(wav), dtype=torch.long)

    with torch.no_grad():
        out = model(raw_audio=wav_tensor, audio_length=wav_lengths)
    print(f"\nOffline inference (no labels):")
    print(f"  logits keys: {list(out.logits.keys())}")
    for level, logits in out.logits.items():
        print(f"  {level} logits shape: {logits.shape}")
    print(f"  encoder_output shape: {out.encoder_output.shape}")
    print(f"  encoder_lengths: {out.encoder_lengths}")
    print(f"  loss: {out.loss}")

    # ---- 5. Offline inference with labels ----
    tokenizer = MultiLevelTokenizer("./")

    uth_strings = [Aya(2, i).get().uthmani for i in range(1, batch_size + 1)]
    moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=4,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=4,
        tasheel_or_madd="madd",
    )

    photenized_outs = [
        quran_phonetizer(
            uth_str,
            moshaf,
            remove_spaces=True,
        )
        for uth_str in uth_strings
    ]

    token_out = tokenizer.tokenize(
        [p.phonemes for p in photenized_outs],
        [p.sifat for p in photenized_outs],
        to_dict=True,
        return_tensors="pt",
        padding="longest",
    )
    print(f"\nTokenized input_ids['phonemes'] shape: {token_out['input_ids']['phonemes'].shape}")

    with torch.no_grad():
        out = model(
            raw_audio=wav_tensor,
            audio_length=wav_lengths,
            labels=token_out["input_ids"],
            labels_mask=token_out["attention_mask"],
        )
    print(f"\nOffline inference with labels:")
    print(f"  loss: {out.loss.item():.4f}")
    print(f"  logits keys: {list(out.logits.keys())}")

    # ---- 6. Decode predictions ----
    with torch.no_grad():
        out = model(raw_audio=wav_tensor, audio_length=wav_lengths)
    level_to_pred_ids = {
        k: torch.argmax(v, dim=-1) for k, v in out.logits.items()
    }
    decoded_outs = tokenizer.decode(
        level_to_pred_ids,
        place_zeros_in_between=False,
    )
    print(f"\nDecoded outputs:")
    print(json.dumps(decoded_outs, indent=1, ensure_ascii=False))

    # ---- 7. Loss with gradient flow ----
    model.train()
    out = model(
        raw_audio=wav_tensor,
        audio_length=wav_lengths,
        labels=token_out["input_ids"],
        labels_mask=token_out["attention_mask"],
    )
    print(f"\nLoss (training mode): {out.loss.item():.4f}")
    assert out.loss is not None and out.loss.isfinite(), "Loss must be finite"
    out.loss.backward()
    print(f"  gradient flows: yes")
    model.eval()

    # ---- 8. Cache dataclass + streaming step ----
    print(f"\n--- Cache dataclass + streaming ---")
    model.setup_streaming_params()
    cfg = model.encoder.streaming_cfg
    print(f"  chunk_size: {cfg.chunk_size}")
    print(f"  shift_size: {cfg.shift_size}")
    print(f"  valid_out_len: {cfg.valid_out_len}")

    cache = model.get_initial_cache(batch_size=batch_size)
    print(f"  initial cache:")
    print(f"    last_channel: {cache.last_channel.shape}")
    print(f"    last_time: {cache.last_time.shape}")
    print(f"    last_channel_len: {cache.last_channel_len.shape}")

    proc_signal = torch.randn(batch_size, config.feat_in, 128)
    proc_length = torch.full((batch_size,), 128, dtype=torch.long)

    with torch.no_grad():
        out = model(
            raw_audio=None,
            audio_length=None,
            processed_signal=proc_signal,
            processed_length=proc_length,
            cache=cache,
            keep_all_outputs=False,
            drop_extra_pre_encoded=0,
        )
    print(f"  streaming step 1:")
    print(f"    encoder_output shape: {out.encoder_output.shape}")
    print(f"    encoder_lengths: {out.encoder_lengths}")
    print(f"    cache present: {out.cache is not None}")
    if out.cache is not None:
        print(f"    new cache.last_channel: {out.cache.last_channel.shape}")
        print(f"    new cache.last_time: {out.cache.last_time.shape}")
        print(f"    new cache.last_channel_len: {out.cache.last_channel_len.shape}")

    # Step 2 with updated cache
    with torch.no_grad():
        out2 = model(
            raw_audio=None,
            audio_length=None,
            processed_signal=proc_signal,
            processed_length=proc_length,
            cache=out.cache,
            keep_all_outputs=False,
        )
    print(f"  streaming step 2:")
    print(f"    encoder_output shape: {out2.encoder_output.shape}")
    print(f"    encoder_lengths: {out2.encoder_lengths}")
    print(f"    cache present: {out2.cache is not None}")

    # ---- 9. selected_levels filtering ----
    with torch.no_grad():
        out_sel = model(
            raw_audio=wav_tensor,
            audio_length=wav_lengths,
            selected_levels={"phonemes"},
        )
    print(f"\nselected_levels={{'phonemes'}}: {list(out_sel.logits.keys())}")
    assert list(out_sel.logits.keys()) == ["phonemes"]

    # ---- 10. return_dict=False ----
    with torch.no_grad():
        tuple_out = model(
            raw_audio=wav_tensor,
            audio_length=wav_lengths,
            return_dict=False,
        )
    print(f"\nreturn_dict=False: type={type(tuple_out).__name__}, len={len(tuple_out)}")

    print(f"\nAll tests passed!")


if __name__ == "__main__":
    main()
