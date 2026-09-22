"""Chunk-boundary CTC contracts for :mod:`ctc_decoding`.

These run without torch, nemo, or a model — the collapse rules are pure integer
manipulation, and they are exactly the part of the streaming path that is easy to get
subtly wrong.  The live mic app asserts the same equivalence at runtime ("per-chunk log
concatenates to transcript"); this file makes it a permanent guard.

Runnable two ways, so it works in CI and on a machine without the GPU stack::

    pytest tests/test_ctc_decoding.py
    python tests/test_ctc_decoding.py
"""

import random

from prepare_quran_dataset.ctc_decoding import (
    BLANK_ID,
    EOS_ID,
    collapse_carry,
    ctc_collapse,
)

# A blank, a plain phoneme, and noon/alef — the tokens that actually double in real
# output (shadda, madd), so the fuzz below exercises geminates rather than noise.
FUZZ_ALPHABET = [BLANK_ID, BLANK_ID, BLANK_ID, EOS_ID, 5, 26, 26, 30]


def _skip(reason: str) -> None:
    """Skip under pytest; print and continue under the __main__ driver."""
    try:
        import pytest
    except ImportError:
        print(f"  SKIP: {reason}")
        return
    pytest.skip(reason)


def test_known_vectors():
    """The textbook cases, including the two that distinguish geminates."""
    cases = [
        ([5, 5, 5], [5]),  # one held phoneme collapses to one token
        ([5, 0, 5], [5, 5]),  # a blank between repeats keeps both: a geminate
        ([5, 5, 0, 5, 5], [5, 5]),  # held, blank, held -> still two
        ([], []),
        ([0, 0, 0], []),
        ([26, 0, 26, 0, 26, 0, 26], [26, 26, 26, 26]),  # mushaddad noon + ghunna
    ]
    for ids, expected in cases:
        assert ctc_collapse(ids) == expected, ids


def test_eos_separates_rather_than_merges():
    """Specials are filtered after the collapse, so [EOS] acts as a separator.

    Filtering before collapsing would weld the pair into a single phoneme, silently
    destroying a shadda.
    """
    assert ctc_collapse([5, EOS_ID, 5]) == [5, 5]
    # Contrast: with no separator at all the two frames are one phoneme.
    assert ctc_collapse([5, 5]) == [5]


def test_geminate_survives_a_chunk_boundary():
    """The regression that motivated collapse_carry.

    Frames [26,26 | 26,0,26] are a shadda (two noon).  Collapsing each chunk from a
    fresh blank emits three, inventing a shadda that was never spoken.
    """
    chunks = [[26, 26], [26, 0, 26]]
    flat = [t for chunk in chunks for t in chunk]

    naive = [t for chunk in chunks for t in ctc_collapse(chunk)]
    assert naive == [26, 26, 26], "the old per-chunk behaviour should over-emit"

    prev, carried = BLANK_ID, []
    for chunk in chunks:
        tokens, prev = collapse_carry(chunk, prev)
        carried += tokens
    assert carried == ctc_collapse(flat) == [26, 26]


def test_collapse_carry_matches_global_collapse():
    """The property the whole streaming decoder rests on.

    For any sequence cut at any boundaries, decoding chunk-by-chunk with a carried
    ``prev`` must equal decoding the concatenated frames in one pass.
    """
    rng = random.Random(0)
    for _ in range(20_000):
        n = rng.randint(1, 60)
        ids = [rng.choice(FUZZ_ALPHABET) for _ in range(n)]

        cuts = sorted(rng.sample(range(1, n), rng.randint(0, min(4, n - 1)))) if n > 1 else []
        chunks, start = [], 0
        for cut in cuts + [n]:
            chunks.append(ids[start:cut])
            start = cut

        prev, carried = BLANK_ID, []
        for chunk in chunks:
            tokens, prev = collapse_carry(chunk, prev)
            carried += tokens
        assert carried == ctc_collapse(ids), (ids, chunks)


def test_single_frame_chunks_are_the_worst_case():
    """One frame per chunk is the most boundaries possible; it must still agree."""
    rng = random.Random(1)
    for _ in range(500):
        ids = [rng.choice(FUZZ_ALPHABET) for _ in range(rng.randint(1, 40))]
        prev, carried = BLANK_ID, []
        for t in ids:
            tokens, prev = collapse_carry([t], prev)
            carried += tokens
        assert carried == ctc_collapse(ids), ids


def _load_canonical_ctc_decode():
    """Lift ``ctc_decode`` out of train_streaming.py without importing the module.

    The obvious ``from train_streaming import ctc_decode`` skips everywhere — even on
    Colab — because that module imports Levenshtein, datasets and jax at the top.  The
    function itself is self-contained numpy, so extracting its AST keeps this a real
    drift guard instead of a permanently-skipped test.
    """
    import ast
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    path = Path(__file__).resolve().parents[1] / "train_streaming.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ctc_decode"),
        None,
    )
    if fn is None:
        return None
    # `ctc_decode` annotates its return as list[NDArray], evaluated at def time.
    namespace = {"np": np, "NDArray": NDArray}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ctc_decode"]


def test_matches_canonical_ctc_decode():
    """``ctc_collapse`` must stay equivalent to the repo's canonical decoder.

    Equivalence holds only once specials are stripped, which every ctc_decode caller
    does immediately afterwards — that composition is the contract being pinned.
    """
    try:
        import numpy as np
    except ImportError:
        _skip("numpy unavailable")
        return
    canonical_fn = _load_canonical_ctc_decode()
    if canonical_fn is None:
        _skip("ctc_decode not found in train_streaming.py")
        return

    rng = random.Random(2)
    for _ in range(2_000):
        ids = [rng.choice(FUZZ_ALPHABET) for _ in range(rng.randint(1, 60))]
        canonical = canonical_fn([np.array(ids, dtype=np.int64)], blank_id=BLANK_ID)[0]
        expected = [int(t) for t in canonical if int(t) not in (BLANK_ID, EOS_ID)]
        assert ctc_collapse(ids) == expected, ids


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    raise SystemExit(failures)
