import json
from pathlib import Path

import torch

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTC,
    FastConformerCacheAwareMultilevelCTCConfig,
)
from prepare_quran_dataset.modeling_fastconformer_cache_aware.vocab import (
    build_quran_phoneme_script_vocab,
)

if __name__ == "__main__":
    # ---- 1. Setup vocab ----
    vocab_path = "./vocab_streaming/vocab.json"
    if not Path(vocab_path).exists():
        build_quran_phoneme_script_vocab(vocab_path)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}

    # ---- 2. Config ----
    config = FastConformerCacheAwareMultilevelCTCConfig(
        level_to_vocab_size=level_to_vocab_size,
        level_to_loss_weight={"phonemes": 0.5, "hams_or_jahr": 0.2},
    )

    model = FastConformerCacheAwareMultilevelCTC(config)

    # Offline
    print("Normal Offline Masking")
    pad_mask, att_mask = model.encoder._create_masks(
        att_context_size=[6, 2],
        padding_length=torch.tensor([20]),
        max_audio_length=20,
        offset=None,
        device="cpu",
    )
    print(att_mask.to(torch.long))
    print(pad_mask.to(torch.long))

    # Streaming Masking
    print("\nStreaming Masking step 0 (no cache yet)")
    # the len of the input is audio_input = chunk_size
    # padding_size = len(audio) + cach_len
    # max_audio_size = len(audio) + cach_len
    max_cache_len = 6
    input_len = 3
    last_cache_len = 0
    pad_mask, att_mask = model.encoder._create_masks(
        att_context_size=[max_cache_len, 2],
        padding_length=torch.tensor([input_len + max_cache_len]),
        max_audio_length=input_len + max_cache_len,
        offset=max_cache_len - torch.tensor([last_cache_len]),
        device="cpu",
    )
    print(att_mask.to(torch.long))
    print(pad_mask.to(torch.long))

    # Streaming Masking
    print("\nStreaming Masking step 1 (no cache yet)")
    # the len of the input is audio_input = chunk_size
    # padding_size = len(audio) + cach_len
    # max_audio_size = len(audio) + cach_len
    max_cache_len = 6
    input_len = 3
    last_cache_len = 3
    pad_mask, att_mask = model.encoder._create_masks(
        att_context_size=[max_cache_len, 2],
        padding_length=torch.tensor([input_len + max_cache_len]),
        max_audio_length=input_len + max_cache_len,
        offset=max_cache_len - torch.tensor([last_cache_len]),
        device="cpu",
    )
    print(att_mask.to(torch.long))
    print(pad_mask.to(torch.long))

    # Streaming Masking
    print("\nStreaming Masking step inf (no cache yet)")
    # the len of the input is audio_input = chunk_size
    # padding_size = len(audio) + cach_len
    # max_audio_size = len(audio) + cach_len
    max_cache_len = 6
    input_len = 3
    last_cache_len = 6
    pad_mask, att_mask = model.encoder._create_masks(
        att_context_size=[max_cache_len, 2],
        padding_length=torch.tensor([input_len + max_cache_len]),
        max_audio_length=input_len + max_cache_len,
        offset=max_cache_len - torch.tensor([last_cache_len]),
        device="cpu",
    )
    print(att_mask.to(torch.long))
    print(pad_mask.to(torch.long))

    # Offline
    print("-" * 60)
    print("-" * 60)
    print("Constant delay Offline Masking")
    pad_mask, att_mask = model.encoder._create_masks(
        att_context_size=[6, 2, 2],
        padding_length=torch.tensor([20]),
        max_audio_length=20,
        offset=None,
        device="cpu",
    )
    print(att_mask.to(torch.long))
    print(pad_mask.to(torch.long))

    # Constant Delay Streaming Masking step 0 (no cache yet)
    # chunk_size = lookahead + 1 + C = 2 + 1 + 2 = 5, C = 2
    # window = cache(6) + chunk(5) = 11 frames
    print("\nConstant Delay Streaming Masking step 0 (no cache yet)")
    max_cache_len = 6
    input_len = 5
    last_cache_len = 0
    pad_mask, att_mask = model.encoder._create_masks(
        att_context_size=[max_cache_len, 2, 2],
        padding_length=torch.tensor([input_len + max_cache_len]),
        max_audio_length=input_len + max_cache_len,
        offset=max_cache_len - torch.tensor([last_cache_len]),
        device="cpu",
    )
    # Apply streaming tail masking: C lookahead rows are keys-only.
    # In the full window, tail starts at cache_len + chunk_size.
    print(att_mask.to(torch.long))
    print(pad_mask.to(torch.long))

    # Streaming Masking
    print("\nConstat Streaming Masking step 1 (no cache yet)")
    # the len of the input is audio_input = chunk_size
    # padding_size = len(audio) + cach_len
    # max_audio_size = len(audio) + cach_len
    max_cache_len = 6
    input_len = 5
    last_cache_len = 3
    pad_mask, att_mask = model.encoder._create_masks(
        att_context_size=[max_cache_len, 2, 2],
        padding_length=torch.tensor([input_len + max_cache_len]),
        max_audio_length=input_len + max_cache_len,
        offset=max_cache_len - torch.tensor([last_cache_len]),
        device="cpu",
    )
    print(att_mask.to(torch.long))
    print(pad_mask.to(torch.long))

    # Streaming Masking
    print("\nStreaming Masking step inf (no cache yet)")
    # the len of the input is audio_input = chunk_size
    # padding_size = len(audio) + cach_len
    # max_audio_size = len(audio) + cach_len
    max_cache_len = 6
    input_len = 5
    last_cache_len = 6
    pad_mask, att_mask = model.encoder._create_masks(
        att_context_size=[max_cache_len, 2, 2],
        padding_length=torch.tensor([input_len + max_cache_len]),
        max_audio_length=input_len + max_cache_len,
        offset=max_cache_len - torch.tensor([last_cache_len]),
        device="cpu",
    )
    print(att_mask.to(torch.long))
    print(pad_mask.to(torch.long))
