import pytest
import torch
from prepare_quran_dataset.modeling_streaming_rnn.modeling_rnn_streaming_multi_level_ctc import (
    convert_input_to_chunked_for_offline,
)


# ======================================================================
# Static test cases (input, lookahead, chunk, lookback, expected_output)
# ======================================================================
test_cases = [
    # 1. Basic case with padding needed
    #   batch=2, seq_len=5, features=1, chunk=3, lookback=2, lookahead=1
    (
        torch.tensor(
            [
                [[1.0], [2.0], [3.0], [4.0], [5.0]],
                [[6.0], [7.0], [8.0], [9.0], [10.0]],
            ]
        ),  # input
        1,  # lookahead
        3,  # chunk
        2,  # lookback
        # expected output (4 chunks, streaming_len=6)
        torch.tensor(
            [
                [[0.0], [0.0], [1.0], [2.0], [3.0], [4.0]],  # chunk 0
                [[2.0], [3.0], [4.0], [5.0], [0.0], [0.0]],  # chunk 1
                [[0.0], [0.0], [6.0], [7.0], [8.0], [9.0]],  # chunk 2
                [[7.0], [8.0], [9.0], [10.0], [0.0], [0.0]],  # chunk 3
            ]
        ),
    ),
    # 3. No lookback or lookahead (both zero)
    #   batch=1, seq_len=3, features=1, chunk=2, lookback=0, lookahead=0
    (
        torch.tensor(
            [
                [[1.0], [2.0], [3.0]],
            ]
        ),
        0,
        2,
        0,
        # padded to length 4 -> 2 chunks, output (2,2,1)
        torch.tensor([[[1.0], [2.0]], [[3.0], [0.0]]]),
    ),
    # 4. Lookback and lookahead larger than chunk, single batch
    #   batch=1, seq_len=2, features=1, chunk=2, lookback=2, lookahead=2
    (
        torch.tensor([[[1.0], [2.0]]]),
        2,
        2,
        2,
        # seq_len%chunk=0 -> pad_len=2 -> padded length 4 -> 2 chunks
        # output (2, 2+2+2=6, 1)
        torch.tensor(
            [
                [[0.0], [0.0], [1.0], [2.0], [0.0], [0.0]],  # chunk 0
            ]
        ),
    ),
    # 5. Multi‑feature, exact fit without padding bug (seq_len % chunk != 0)
    #   batch=1, seq_len=5, features=3, chunk=3, lookback=1, lookahead=1
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
        # pad_len=1 -> padded len=6 -> 2 chunks, output (2,1+3+1=5,3)
        torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ],  # chunk 0
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],  # chunk 1
            ]
        ),
    ),
]


# ======================================================================
# Parametrized test
# ======================================================================
@pytest.mark.parametrize("input, lookahead, chunk, lookback, expected", test_cases)
def test_convert_static(input, lookahead, chunk, lookback, expected):
    """Test against hard‑coded expected outputs to avoid re‑implementing the logic."""
    result = convert_input_to_chunked_for_offline(
        input, lookahead=lookahead, chunk=chunk, lookback=lookback
    )
    print(f"Input:\n{input}\nOutput:\n{result}")
    # Exact comparison – all values are integer or .0 floats, so no tolerance needed
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
