import json
import string

from quran_transcript import quran_phonetizer, Aya, MoshafAttributes
from time import perf_counter
import numpy as np
from numpy.typing import NDArray
import Levenshtein

from prepare_quran_dataset.modeling_streaming_rnn.multi_level_tokenizer import (
    MultiLevelTokenizer,
)


if __name__ == "__main__":
    tokenizer = MultiLevelTokenizer("./vocab_streaming")

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
        [p.phonemes for p in photenized_outs] + [""],
        [p.sifat for p in photenized_outs] + [],
        add_eos=True,
        to_dict=True,
        padding="longest",
        # return_tensors="np",
    )
    print(f"Total tokenization time: {perf_counter() - start} ")
    print(token_out)

    ph_level = [
        13,
        33,
        29,
        33,
        19,
        24,
        33,
        25,
        34,
        32,
        32,
        26,
        33,
        20,
        33,
        9,
        33,
        25,
        25,
        25,
        25,
        33,
        26,
        35,
        24,
        23,
        33,
        10,
        10,
        33,
        30,
        30,
        3,
        34,
        24,
        2,
        33,
        14,
        35,
        11,
    ]

    token_out["input_ids"]["phonemes"] = [ph_level]
    decoded_outs = tokenizer.decode(
        token_out["input_ids"],
        place_zeros_in_between=True,
    )
