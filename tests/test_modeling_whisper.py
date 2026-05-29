import json

from quran_transcript import quran_phonetizer, Aya, MoshafAttributes
from transformers import AutoFeatureExtractor
import torch

from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer
from prepare_quran_dataset.modeling_whisper.configuration_for_multi_level_ctc import (
    WhisperEncoderForMultilevelCTCConfig,
)
from prepare_quran_dataset.modeling_whisper.modeling_multi_level_ctc_whisper_encoder import (
    WhisperEncoderForMultilevelCTC,
)


if __name__ == "__main__":
    with open("./vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}

    config = WhisperEncoderForMultilevelCTCConfig(
        level_to_vocab_size=level_to_vocab_size,
        dropout=0,
        mask_time_prob=0.0,
        ctc_loss_reduction="mean",
    )
    print(config)

    model = WhisperEncoderForMultilevelCTC.from_pretrained(
        # "openai/whisper-small",
        "./whisper-small-encoder-only",
        config=config,
    )
    print(model)
    processor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    print(processor)
    batch_size = 2

    # Test Infernce without labels
    inputs = processor(
        [[0] * 16000] * batch_size,
        return_tensors="pt",
        padding=False,
        return_attention_mask=True,
    )
    model_output = model(**inputs)
    print(model_output)

    # Test Inference with labels
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
    print(token_out["input_ids"]["phonemes"].shape)

    model_output = model(**inputs, labels=token_out["input_ids"], return_dict=False)
    print(model_output[1].keys())
    print(model_output[1]["phonemes"].shape)
    print(model_output[1]["phonemes"].argmax(dim=-1))

    level_to_pred_ids = {k: torch.argmax(v, dim=-1) for k, v in model_output[1].items()}
    print(level_to_pred_ids)
    decoded_outs = tokenizer.decode(
        level_to_pred_ids,
        place_zeros_in_between=False,
    )
    print(json.dumps(decoded_outs, indent=1, ensure_ascii=False))

    print(model(**inputs, labels=token_out["input_ids"]).loss)
