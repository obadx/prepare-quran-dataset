import json
from quran_transcript import quran_phonetizer, Aya, MoshafAttributes

from prepare_quran_dataset.modeling.multi_level_tokenizer import MutliLevelTokenizer

if __name__ == "__main__":
    tokenizer = MutliLevelTokenizer("./")

    uth_strings = [Aya(1, i).get().uthmani for i in range(1, 3)]
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
    )

    print(json.dumps(token_out["input_ids"], indent=1))

    decoded_outs = tokenizer.decode(
        token_out["input_ids"],
        place_zeros_in_between=True,
    )
    print(json.dumps(decoded_outs, indent=1, ensure_ascii=False))
