from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2BertForCTC,
    AutoConfig,
)
from quran_transcript import quran_phonetizer, Aya, MoshafAttributes

from prepare_quran_dataset.modeling.vocab import PAD_TOKEN, PAD_TOKEN_IDX


def add_zero_between(L, x=PAD_TOKEN_IDX):
    out = []
    for i, item in enumerate(L):
        out.append(item)
        if i < len(L) - 1:  # Don't add zero after the last element
            out.append(0)
    return out


if __name__ == "__main__":
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./", pad_token=PAD_TOKEN, target_lang="phonemes"
    )
    # tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
    #     "./",
    #     pad_token=PAD_TOKEN,
    #     phonemizer_lang="phonemes",
    #     do_phonemize=False,
    # )
    print(tokenizer)

    moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=4,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=4,
        tasheel_or_madd="madd",
    )

    uth_str = Aya(1, 1).get().uthmani
    ph_str = quran_phonetizer(uth_str, moshaf, remove_spaces=True).phonemes
    print(f"Len of phonemes: {len(ph_str)}")
    tok_text = tokenizer(ph_str)
    input_ids = tok_text["input_ids"]
    print(f"Len of ids: {len(input_ids)}")
    print(tok_text)

    decoded_str = tokenizer.decode(add_zero_between(input_ids))
    print(decoded_str)
    print(f"Len of output: {len(decoded_str)}")

    # model = Wav2Vec2BertForCTC.from_pretrained("facebook/w2v-bert-2.0", vocab_size=55)
    # print(model)
    # print(model.config)
