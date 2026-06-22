import json

from quran_transcript import quran_phonetizer, Aya, MoshafAttributes
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch

from prepare_quran_dataset.modeling_streaming_rnn.multi_level_tokenizer import (
    MultiLevelTokenizer,
)
from prepare_quran_dataset.modeling_streaming_rnn.configuration_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCConfig,
)
from prepare_quran_dataset.modeling_streaming_rnn.modeling_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTC,
)


if __name__ == "__main__":
    with open("./vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}

    config = Wav2Vec2BertForRNNStreamingMultilevelCTCConfig(
        level_to_vocab_size=level_to_vocab_size,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
        num_hidden_layers=16,
        rnn_hidden_size=128,
        add_adapter=False,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        layerdrop=0.0,
        conv_depthwise_kernel_size=9,
    )
    print(config)

    model = Wav2Vec2BertForRNNStreamingMultilevelCTC.from_pretrained(
        "facebook/w2v-bert-2.0",
        config=config,
        ignore_mismatched_sizes=True,
    )

    print(model)
    processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    print(processor)
    batch_size = 2

    # Test Infernce without labels
    inputs = processor([[0] * 16000] * batch_size, return_tensors="pt")
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

    model_output = model(
        **inputs,
        labels=token_out["input_ids"],
        labels_mask=token_out["attention_mask"],
        return_dict=False,
    )
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

    print("Loss:")
    print(
        model(
            **inputs,
            labels=token_out["input_ids"],
            labels_mask=token_out["attention_mask"],
        ).loss
    )
