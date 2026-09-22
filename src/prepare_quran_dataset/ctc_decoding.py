"""Greedy CTC decoding for the multi-level FastConformer heads.

Deliberately free of every third-party import — no torch, no numpy, no nemo — so the
collapse rules can be unit-tested without a model or a GPU.  It lives at the package
root rather than under ``modeling_fastconformer_cache_aware`` for that reason: that
subpackage's ``__init__`` imports transformers and nemo, so any module inside it drags
the whole stack in.  The scheme is shared with the RNN streaming model anyway.

:mod:`streaming_mic` imports from here; nothing here imports from it.

The collapse itself matches ``train_streaming.ctc_decode`` exactly: a blank resets
``prev``, and adjacent equal ids merge.  ``tests/test_ctc_decoding.py`` asserts that
equivalence directly whenever the heavy dependencies are importable.

Two contracts are load-bearing, and each has already been broken once in this repo:

1.  ``prev`` must carry across a chunk boundary.  Restarting it at blank re-emits a
    phoneme whose frames straddle the seam, and in this phonetic scheme a doubled
    token is a geminate (shadda) whose repeat count encodes duration in harakat — so a
    seam duplicate does not merely look wrong, it *fabricates a shadda*.  Use
    :func:`collapse_carry` for incremental work; :func:`ctc_collapse` is the
    whole-stream wrapper, and is only safe because the streamer accumulates every
    frame into one flat list before decoding.

2.  Special ids are filtered *after* collapsing, never before.  ``[EOS]`` therefore
    separates two identical phonemes instead of merging them: ``[5, 1, 5]`` decodes to
    two ``5``\\ s.  Filtering first would silently weld a geminate pair into one
    phoneme.
"""

from __future__ import annotations

# Mirrors vocab.PAD_TOKEN_IDX / vocab.EOS_TOKEN_IDX.  Hardcoded rather than imported
# because vocab pulls in quran_transcript, which would drag a dependency into this
# module and defeat its purpose.
BLANK_ID = 0
EOS_ID = 1
SPECIAL_IDS = (BLANK_ID, EOS_ID)


def collapse_carry(
    ids: list[int],
    prev: int = BLANK_ID,
    blank_id: int = BLANK_ID,
) -> tuple[list[int], int]:
    """Collapse one chunk of frame ids, threading ``prev`` in and out.

    Args:
        ids: Frame-level argmax ids for this chunk only.
        prev: The ``prev`` returned by the previous chunk's call.  Pass
            ``BLANK_ID`` for the first chunk — that is what a stream starts from.
        blank_id: The CTC blank, which is also ``[PAD]``.

    Returns:
        ``(tokens, prev)`` — the decoded ids for this chunk with specials removed,
        and the ``prev`` to feed into the next chunk.  Concatenating ``tokens``
        across chunks equals :func:`ctc_collapse` over the concatenated frames, for
        any chunking whatsoever.
    """
    out: list[int] = []
    for t in ids:
        if t == blank_id:
            prev = blank_id
            continue
        if t != prev:
            out.append(t)
        prev = t
    return [t for t in out if t not in SPECIAL_IDS], prev


def ctc_collapse(ids: list[int], blank_id: int = BLANK_ID) -> list[int]:
    """Collapse a complete frame-id sequence in one pass.

    Equivalent to a single :func:`collapse_carry` starting from blank.  Correct for
    streaming only because chunk boundaries have already been erased by the time it
    runs: the streamer extends one flat ``level_ids`` list per level, so the decoder
    never sees a seam.
    """
    return collapse_carry(ids, blank_id, blank_id)[0]


__all__ = ["BLANK_ID", "EOS_ID", "SPECIAL_IDS", "collapse_carry", "ctc_collapse"]
