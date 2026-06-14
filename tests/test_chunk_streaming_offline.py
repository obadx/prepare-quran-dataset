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
