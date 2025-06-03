import pytest
import numpy as np

from prepare_quran_dataset.annotate.tarteel import (
    chunk_wave,
    merge_transcripts,
    SmallTarteelOverlap,
)


@pytest.mark.parametrize(
    "wave, chunk_overlap, max_len_samples, expected",
    [
        (
            np.arange(10),
            0,
            3,
            [np.arange(10)[:3], np.arange(10)[3:6], np.arange(10)[6:9], np.array([9])],
        ),
        (
            np.arange(10),
            1,
            3,
            [
                np.arange(10)[:3],
                np.arange(10)[2:5],
                np.arange(10)[4:7],
                np.arange(10)[6:9],
                np.arange(10)[8:],
            ],
        ),
        (
            np.arange(10),
            2,
            3,
            [
                np.arange(10)[:3],
                np.arange(10)[1:4],
                np.arange(10)[2:5],
                np.arange(10)[3:6],
                np.arange(10)[4:7],
                np.arange(10)[5:8],
                np.arange(10)[6:9],
                np.arange(10)[7:],
            ],
        ),
        (
            np.arange(100),
            0,
            200,
            [np.arange(100)],
        ),
        (
            np.arange(100),
            80,
            100,
            [np.arange(100)],
        ),
    ],
)
def test_chunk_wave(wave, chunk_overlap, max_len_samples, expected):
    chunks = chunk_wave(
        wave, chunk_overlap=chunk_overlap, max_len_samples=max_len_samples
    )
    for idx in range(len(expected)):
        np.testing.assert_equal(chunks[idx], expected[idx])


@pytest.mark.parametrize(
    "texts, expected, start, end",
    [
        (
            [
                "Ahmed Mohammed",
                "Mohammed Othman",
            ],
            "Ahmed Mohammed Othman",
            0,
            0,
        ),
        (
            [
                "Ahmed Mohammed",
                "Mohammed Othman",
                "Othman Alsayed",
            ],
            "Ahmed Mohammed Othman Alsayed",
            0,
            0,
        ),
        (
            [
                "Ahmed Mohammed asdkfh",
                "Mohammed Othman",
            ],
            "Ahmed Mohammed Othman",
            0,
            1,
        ),
        (
            [
                "Ahmed Mohammed",
                "saldkfj Mohammed Othman",
            ],
            "Ahmed Mohammed Othman",
            1,
            0,
        ),
        (
            [
                "Ahmed Mohammed asdkfh",
                "skldjfkj Mohammed Othman",
            ],
            "Ahmed Mohammed Othman",
            1,
            1,
        ),
        (
            [
                "Ahmed Mohammed",
                "sldkfj  Mohammed Othman",
                "sdlkfj Othman Alsayed",
            ],
            "Ahmed Mohammed Othman Alsayed",
            1,
            0,
        ),
        (
            [
                "Ahmed Mohammed sdkfj",
                "Mohammed Othman slkdfj",
                "Othman Alsayed",
            ],
            "Ahmed Mohammed Othman Alsayed",
            0,
            1,
        ),
        (
            [
                "Ahmed Mohammed sdlkfjs",
                "lkdfj Mohammed Othman askdjf",
                "sldkfj Othman Alsayed",
            ],
            "Ahmed Mohammed Othman Alsayed",
            1,
            1,
        ),
        (
            [
                "Ahmed Mohammed aslkdfj asdklfj",
                "Mohammed Othman skdjf sdkjf",
                "Othman Alsayed",
            ],
            "Ahmed Mohammed Othman Alsayed",
            0,
            2,
        ),
        (
            [
                "Ahmed Mohammed skldfj sldkfj",
                "sdkfj sdklfj sdlkjf Mohammed Othman sdlkfj sldkfj",
                "sldkfj sldkfj sldkfj Othman Alsayed",
            ],
            "Ahmed Mohammed Othman Alsayed",
            3,
            2,
        ),
    ],
)
def test_merge_text(texts, expected, start, end):
    assert expected == merge_transcripts(
        texts, min_merge_chars=5, start_truncate_words=start, end_truncate_words=end
    )


def test_merge_text_raise():
    texts = [
        "Mohammed Ahmed",
        "Mohammed Othman",
    ]
    with pytest.raises(SmallTarteelOverlap):
        merge_transcripts(
            texts, min_merge_chars=5, start_truncate_words=0, end_truncate_words=0
        )
