import pytest
import torch
from prepare_quran_dataset.modeling_streaming_rnn.modeling_rnn_streaming_multi_level_ctc import (
    convert_input_to_chunked_for_offline,
)


# ======================================================================
# Static test cases (input, lookahead, chunk, lookback, expected_output)
# ======================================================================
test_cases = [
    # 0. Original test — basic case with padding needed
    (
        torch.tensor(
            [
                [[1.0], [2.0], [3.0], [4.0], [5.0]],
                [[6.0], [7.0], [8.0], [9.0], [10.0]],
            ]
        ),
        1,
        3,
        2,
        torch.tensor(
            [
                [[0.0], [0.0], [1.0], [2.0], [3.0], [4.0]],
                [[2.0], [3.0], [4.0], [5.0], [0.0], [0.0]],
                [[0.0], [0.0], [6.0], [7.0], [8.0], [9.0]],
                [[7.0], [8.0], [9.0], [10.0], [0.0], [0.0]],
            ]
        ),
    ),
    # 1. No lookback or lookahead (both zero)
    (
        torch.tensor([[[1.0], [2.0], [3.0]]]),
        0,
        2,
        0,
        torch.tensor([[[1.0], [2.0]], [[3.0], [0.0]]]),
    ),
    # 2. Lookback and lookahead larger than chunk, single batch
    (
        torch.tensor([[[1.0], [2.0]]]),
        2,
        2,
        2,
        torch.tensor([[[0.0], [0.0], [1.0], [2.0], [0.0], [0.0]]]),
    ),
    # 3. Multi-feature with padding
    (
        torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                ]
            ]
        ),
        1,
        3,
        1,
        torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ],
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]
        ),
    ),
    # 4. 2D input (attention_mask style)
    (
        torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]),
        1,
        4,
        2,
        torch.tensor(
            [
                [0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
    # 5. seq_len == chunk (exact, no padding)
    (
        torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]]),
        1,
        5,
        2,
        torch.tensor([[[0.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [0.0]]]),
    ),
    # 6. chunk=1 (minimal chunk size)
    (
        torch.tensor([[[1.0], [2.0], [3.0]]]),
        1,
        1,
        1,
        torch.tensor(
            [
                [[0.0], [1.0], [2.0]],
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [0.0]],
            ]
        ),
    ),
    # 7. lookback == chunk
    (
        torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        1,
        2,
        2,
        torch.tensor(
            [
                [[0.0], [0.0], [1.0], [2.0], [3.0]],
                [[1.0], [2.0], [3.0], [4.0], [0.0]],
            ]
        ),
    ),
    # 8. lookahead == chunk
    (
        torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        2,
        2,
        1,
        torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0], [4.0]],
                [[2.0], [3.0], [4.0], [0.0], [0.0]],
            ]
        ),
    ),
    # 9. Single element
    (
        torch.tensor([[[1.0]]]),
        0,
        1,
        0,
        torch.tensor([[[1.0]]]),
    ),
    # 10. Large batch multi-feature (shapes only)
    (
        torch.randn(4, 12, 2),
        2,
        5,
        2,
        None,  # expected checked via shape
    ),
    # 11. Default-parameter long sequence (shapes only)
    (
        torch.randn(1, 100, 1),
        5,
        25,
        5,
        None,  # expected checked via shape
    ),
    # 12. 2D input (attention_mask style + device)
    (
        torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
            device="cuda",
        ),
        1,
        4,
        2,
        torch.tensor(
            [
                [0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
            ],
            device="cuda",
        ),
    ),
]


# ======================================================================
# Parametrized tests
# ======================================================================
@pytest.mark.parametrize("input, lookahead, chunk, lookback, expected", test_cases)
def test_convert_static(input, lookahead, chunk, lookback, expected):
    """Test against hard-coded expected outputs or shape-only checks."""
    result = convert_input_to_chunked_for_offline(
        input, lookahead=lookahead, chunk=chunk, lookback=lookback
    )
    assert result.dtype == input.dtype
    assert result.device == input.device
    if expected is None:
        # shape-only test
        assert result.ndim == 3 or (result.ndim == 2 and input.ndim == 2)
        streaming_len = lookback + chunk + lookahead
        assert result.shape[1] == streaming_len
        return
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


# ======================================================================
# Error-case tests
# ======================================================================
def test_1d_input_raises_valueerror():
    with pytest.raises(ValueError, match="1D"):
        convert_input_to_chunked_for_offline(torch.tensor([1, 2, 3]))


def test_4d_input_raises_valueerror():
    with pytest.raises(ValueError, match="4D"):
        convert_input_to_chunked_for_offline(torch.randn(2, 3, 4, 5))


def test_seq_len_lt_chunk_raises_valueerror():
    with pytest.raises(ValueError, match="seq_len"):
        convert_input_to_chunked_for_offline(
            torch.tensor([[[1.0], [2.0], [3.0]]]), chunk=5
        )


# ----------------------------------------------------------------------
# Tests for max_chunk_batch_size
# ----------------------------------------------------------------------

# (lookback=1, chunk=3, lookahead=1) -> streaming_len=5
# Input: batch=2, seq=5, features=1
# Default output (max_chunk_batch_size=1) would have shape (4,5)
# When max_chunk_batch_size=6 -> need total chunks divisible by 6 -> pad to 6
seq5_input = torch.tensor(
    [
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
        [[6.0], [7.0], [8.0], [9.0], [10.0]],
    ]
)
expected_5_6 = torch.tensor(
    [
        [0.0, 1.0, 2.0, 3.0, 4.0],  # batch0 chunk0
        [3.0, 4.0, 5.0, 0.0, 0.0],  # batch0 chunk1
        [0.0, 0.0, 0.0, 0.0, 0.0],  # extra zero chunk
        [0.0, 6.0, 7.0, 8.0, 9.0],  # batch1 chunk0
        [8.0, 9.0, 10.0, 0.0, 0.0],  # batch1 chunk1
        [0.0, 0.0, 0.0, 0.0, 0.0],  # extra zero chunk
    ]
).unsqueeze(dim=-1)

# (lookback=0, chunk=2, lookahead=1) -> streaming_len=3
# Input: batch=2, seq=5 -> num_chunks per batch = ceil(5/2)=3, total=6
# max_chunk_batch_size=4, batch=2 -> total needed multiple of 4 -> pad to 8 (one extra per batch)
seq5_2d_input = torch.tensor(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ]
)  # 2D
expected_2d_5_4 = torch.tensor(
    [
        [1, 2, 3],
        [3, 4, 5],
        [5, 0, 0],
        [0, 0, 0],  # extra zero chunk
        [6, 7, 8],
        [8, 9, 10],
        [10, 0, 0],
        [0, 0, 0],  # extra zero chunk
    ]
)

# No extra padding needed: batch=2, seq=6, chunk=3 -> N=2 each, total=4
# max_chunk_batch_size=4 (4 % 4 == 0) -> unchanged
seq6_input = torch.tensor(
    [
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
        [[7.0], [8.0], [9.0], [10.0], [11.0], [12.0]],
    ]
)
# default output with lookback=0, chunk=3, lookahead=0
# padded: batch0: [1,2,3,4,5,6] -> chunks [[1,2,3],[4,5,6]]
# output shape (4,3): [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
expected_6_4 = torch.tensor(
    [
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
        [[7.0], [8.0], [9.0]],
        [[10.0], [11.0], [12.0]],
    ]
)


@pytest.mark.parametrize(
    "input,lookahead,chunk,lookback,max_chunk_batch_size,expected",
    [
        # padding adds extra zero-chunks
        (seq5_input, 1, 3, 1, 6, expected_5_6),
        # 2D input with extra padding
        (seq5_2d_input, 1, 2, 0, 4, expected_2d_5_4),
        # already aligned, no extra chunks
        (seq6_input, 0, 3, 0, 4, expected_6_4),
    ],
)
def test_max_chunk_batch_size_explicit(
    input, lookahead, chunk, lookback, max_chunk_batch_size, expected
):
    result = convert_input_to_chunked_for_offline(
        input,
        lookahead=lookahead,
        chunk=chunk,
        lookback=lookback,
        max_chunk_batch_size=max_chunk_batch_size,
    )
    assert result.dtype == input.dtype
    assert result.device == input.device
    torch.testing.assert_close(result, expected)


# ----------------------------------------------------------------------
# Tests for max_chunk_batch_size – shape and zero-padding validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch, seq, feat, lookahead, chunk, lookback, max_chunk_batch_size, expected_shape",
    [
        # (2,10,1) with M=6 → 6 chunks needed (2 original per batch → padded to 6)
        (2, 10, 1, 2, 5, 2, 6, (6, 9, 1)),
        # (3,9,2) with M=9 → exactly 9 chunks, no padding
        (3, 9, 2, 1, 4, 1, 9, (9, 6, 2)),
        # (4,7,1) with M=8 → 16 chunks needed (4 raw per batch → already 16, no padding)
        (4, 7, 1, 0, 2, 1, 8, (16, 3, 1)),
        # (1,5,3) with M=1 (default) → 2 chunks, no extra padding
        (1, 5, 3, 1, 3, 1, 1, (2, 5, 3)),
    ],
)
def test_max_chunk_batch_size_shape(
    batch, seq, feat, lookahead, chunk, lookback, max_chunk_batch_size, expected_shape
):
    """Verify output shape matches expected, and that extra chunks are all zeros."""
    x = torch.randn(batch, seq, feat)
    result = convert_input_to_chunked_for_offline(
        x,
        lookahead=lookahead,
        chunk=chunk,
        lookback=lookback,
        max_chunk_batch_size=max_chunk_batch_size,
    )

    # Check exact shape
    assert result.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {result.shape}"
    )
    # dtype and device preservation
    assert result.dtype == x.dtype
    assert result.device == x.device


def test_max_chunk_batch_size_invalid_raises():
    """max_chunk_batch_size must be 1 or a multiple of batch size."""
    x = torch.randn(2, 10, 1)
    with pytest.raises(ValueError, match="max_chunk_batch_size"):
        convert_input_to_chunked_for_offline(
            x, lookahead=1, chunk=5, lookback=1, max_chunk_batch_size=3
        )


def test_max_chunk_batch_size_device_cuda():
    """Check device stays on CUDA when specified."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(2, 10, 1, device="cuda")
    result = convert_input_to_chunked_for_offline(
        x, lookahead=2, chunk=5, lookback=2, max_chunk_batch_size=4
    )
    assert result.device == x.device
