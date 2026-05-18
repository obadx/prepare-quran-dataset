import json
import string

from quran_transcript import quran_phonetizer, Aya, MoshafAttributes
from time import perf_counter
import numpy as np
from numpy.typing import NDArray
import Levenshtein

from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer


if __name__ == "__main__":
    tokenizer = MultiLevelTokenizer("./")

    uth_strings = [Aya(2, i).get().uthmani for i in range(1, 2)]
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

    start = perf_counter()
    token_out = tokenizer.tokenize(
        [p.phonemes for p in photenized_outs],
        [p.sifat for p in photenized_outs],
        to_dict=True,
        padding="longest",
        # return_tensors="np",
    )
    # for level in token_out["input_ids"]:
    #     mask = token_out["attention_mask"][level] == 0
    #     token_out["input_ids"][level][mask] = -100
    print(f"Total tokenization time: {perf_counter() - start} ")

    ph_level = [
        12,
        32,
        28,
        32,
        18,
        23,
        32,
        24,
        33,
        31,
        31,
        25,
        32,
        19,
        32,
        8,
        32,
        24,
        24,
        24,
        24,
        32,
        25,
        34,
        23,
        22,
        32,
        9,
        9,
        32,
        29,
        29,
        2,
        33,
        23,
        1,
        32,
        13,
        34,
        10,
    ]

    token_out["input_ids"]["phonemes"] = [ph_level]
    decoded_outs = tokenizer.decode(
        token_out["input_ids"],
        place_zeros_in_between=True,
    )
    print(json.dumps(decoded_outs, indent=1, ensure_ascii=False))
