import pytest
import numpy as np

from prepare_quran_dataset.annotate.tarteel import chunk_wave


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
