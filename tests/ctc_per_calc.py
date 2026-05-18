import numpy as np
from numpy.typing import NDArray


def ctc_decode(batch_arr, blank_id=0, collapse_consecutive=True) -> list[NDArray]:
    decoded_list = []
    for seq in batch_arr:
        if collapse_consecutive:
            tokens = []
            prev = blank_id
            for current in seq:
                if current == blank_id:
                    prev = blank_id
                    continue
                if current == prev:
                    continue
                tokens.append(current)
                prev = current
            decoded_list.append(np.array(tokens, dtype=seq.dtype))
        else:
            tokens = seq[seq != blank_id]
            decoded_list.append(tokens)
    return decoded_list


# Example usage:
if __name__ == "__main__":
    x = np.array(
        [
            [1, 1, 0, 1, 2, 3, 0, 3, 0, 0, 0, 0],
            [1, 1, 0, 0, 2, 3, 0, 3, 0, 0, 0, 0],
        ]
    )

    target = [
        [1, 1, 2, 3, 3],
        [1, 2, 3, 3],
    ]

    out = ctc_decode(x)
    for t, o in zip(target, out):
        assert (np.array(t) == o).all()

    # without adding zeros
    x = np.array(
        [
            [1, 1, 0, 1, 2, 3, 0, 3, 0, 0, 0, 0],
        ]
    )
    target = np.array(
        [
            [1, 1, 1, 2, 3, 3],
        ],
    )
    out = ctc_decode(x, collapse_consecutive=False)
    for t, o in zip(target, out):
        assert (np.array(t) == o).all()
