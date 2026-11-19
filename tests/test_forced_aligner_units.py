"""Unit tests for the pure helpers of ``forced_aligner``.

None of the unit tests download a model.  The smoke test at the bottom runs
``force_align`` end to end with ``obadx/muaalem-model-v3_2`` (≈2.4 GB download)
and only runs when ``RUN_SLOW=1`` is set::

    uv run pytest tests/test_forced_aligner_units.py
    RUN_SLOW=1 uv run pytest tests/test_forced_aligner_units.py
"""

import json
import os
from pathlib import Path

import pytest
import torch

from prepare_quran_dataset.forced_aligner.forced_aligner import (
    Segment,
    align_tokens,
    count_audio_frames,
    count_required_frames,
    fastconformer_samples_per_frame,
    frames_to_time,
    groups_to_token_ids,
    load_phoneme_vocab,
    merge_repeats,
    segments_to_group_spans,
    split_phoneme_groups,
    w2v2_bert_samples_per_frame,
)

# ``quran_phonetizer`` output for 112:1 (قُلْ هُوَ اللَّهُ أَحَدٌ), hafs.
IKHLAS_1 = "قُل هُوَ للَااهُ ءَحَدڇ"
IKHLAS_1_GROUPS = ["قُ", "ل", "هُ", "وَ", "للَ", "اا", "هُ", "ءَ", "حَ", "دڇ"]

REPO_ROOT = Path(__file__).parent.parent
TEST_WAV = REPO_ROOT / "assets" / "audio-sampels" / "test.wav"


# ---------------------------------------------------------------------------
# Group chunking
# ---------------------------------------------------------------------------


def test_split_phoneme_groups_matches_chunck_phonemes():
    """Groups follow ``chunck_phonemes`` and word spaces are dropped."""
    assert split_phoneme_groups(IKHLAS_1) == IKHLAS_1_GROUPS


def test_split_phoneme_groups_covers_script():
    """Joining the groups gives the script back without spaces."""
    assert "".join(split_phoneme_groups(IKHLAS_1)) == IKHLAS_1.replace(" ", "")


def test_split_phoneme_groups_empty():
    """An empty script has no groups."""
    assert split_phoneme_groups("") == []


def test_split_phoneme_groups_rejects_uncovered_chars():
    """Characters ``chunck_phonemes`` cannot place raise instead of vanishing."""
    with pytest.raises(ValueError, match="do not cover"):
        split_phoneme_groups("قُ!")


# ---------------------------------------------------------------------------
# Char-range offsets
# ---------------------------------------------------------------------------


def test_groups_to_token_ids_ranges():
    """Each group maps to the half-open range of its characters."""
    vocab = {c: i + 1 for i, c in enumerate(sorted(set("".join(IKHLAS_1_GROUPS))))}
    token_ids, ranges = groups_to_token_ids(IKHLAS_1_GROUPS, vocab)

    assert len(token_ids) == 20
    assert ranges[0] == (0, 2)
    assert ranges[1] == (2, 3)
    assert ranges[4] == (7, 10)  # للَ
    assert ranges[-1] == (18, 20)
    for group, (start, end) in zip(IKHLAS_1_GROUPS, ranges):
        assert token_ids[start:end] == [vocab[c] for c in group]


def test_groups_to_token_ids_unknown_phoneme():
    """A phoneme missing from the vocabulary raises and names it."""
    with pytest.raises(ValueError, match="`ل`"):
        groups_to_token_ids(["قُ", "ل"], {"ق": 1, "ُ": 2})


@pytest.mark.parametrize(
    "token_ids, expected",
    [([], 0), ([1, 2, 3], 3), ([1, 1], 3), ([7, 7, 3, 9, 9, 9], 9)],
)
def test_count_required_frames(token_ids, expected):
    """Every consecutive repeat needs one extra blank frame."""
    assert count_required_frames(token_ids) == expected


@pytest.mark.parametrize(
    "vocab_module, vocab_file",
    [
        ("prepare_quran_dataset.modeling.vocab", "vocab.json"),
        (
            "prepare_quran_dataset.modeling_fastconformer_cache_aware.vocab",
            "vocab_streaming/vocab.json",
        ),
    ],
)
def test_load_phoneme_vocab_rebuilds_missing_vocab(vocab_module, vocab_file, tmp_path):
    """A checkpoint without ``vocab.json`` gets the vocab the model trained on."""
    import importlib

    module = importlib.import_module(vocab_module)
    with open(REPO_ROOT / vocab_file, encoding="utf-8") as f:
        expected = json.load(f)["phonemes"]

    def missing_tokenizer(path):
        raise OSError(f"no vocab.json in {path}")

    vocab = load_phoneme_vocab(
        str(tmp_path),
        missing_tokenizer,
        module.build_quran_phoneme_script_vocab,
        expected_size=len(expected),
    )
    assert vocab == expected


def test_load_phoneme_vocab_size_mismatch(tmp_path):
    """A vocab that does not fit the model head raises."""

    class Tokenizer:
        def __init__(self, path):
            self.vocab = {"phonemes": {"[PAD]": 0, "ق": 1}}

    with pytest.raises(ValueError, match="2 tokens"):
        load_phoneme_vocab(str(tmp_path), Tokenizer, lambda p: None, expected_size=43)


# ---------------------------------------------------------------------------
# Segments → spans
# ---------------------------------------------------------------------------


def test_merge_repeats():
    """Runs of identical labels become half-open segments."""
    assert merge_repeats([0, 0, 5, 5, 0, 7]) == [
        Segment(0, 0, 2),
        Segment(5, 2, 4),
        Segment(0, 4, 5),
        Segment(7, 5, 6),
    ]


def test_merge_repeats_empty():
    """An empty path has no segments."""
    assert merge_repeats([]) == []


def test_segments_to_group_spans_midpoints():
    """Blanks between groups split at the midpoint; edges reach 0 and the end."""
    #       _  _  _  ق  ُ  _  _  _  ل  _  _  ه  ُ  _  _  _  _  _
    path = [0, 0, 0, 1, 2, 0, 0, 0, 3, 0, 0, 4, 2, 0, 0, 0, 0, 0]
    spans = segments_to_group_spans(
        merge_repeats(path), [(0, 2), (2, 3), (3, 5)], blank_id=0, num_frames=18
    )
    assert spans == [(0, 6), (6, 10), (10, 18)]


def test_segments_to_group_spans_no_gap():
    """Adjacent groups with no blank between them share their boundary."""
    path = [1, 2, 3]
    spans = segments_to_group_spans(
        merge_repeats(path), [(0, 1), (1, 3)], blank_id=0, num_frames=3
    )
    assert spans == [(0, 1), (1, 3)]


def test_segments_to_group_spans_repeated_tokens():
    """Repeated tokens (e.g. ``اا``) separated by a blank stay in one group."""
    path = [0, 5, 0, 5, 0, 6, 0]
    spans = segments_to_group_spans(
        merge_repeats(path), [(0, 2), (2, 3)], blank_id=0, num_frames=7
    )
    assert spans == [(0, 4), (4, 7)]


def test_segments_to_group_spans_contiguous():
    """Spans tile ``[0, num_frames)`` with no gaps or overlaps."""
    path = [0, 1, 1, 0, 2, 0, 0, 3, 4, 0]
    spans = segments_to_group_spans(
        merge_repeats(path), [(0, 1), (1, 2), (2, 4)], blank_id=0, num_frames=10
    )
    assert spans[0][0] == 0
    assert spans[-1][1] == 10
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start


def test_segments_to_group_spans_token_count_mismatch():
    """A path with the wrong number of tokens raises."""
    with pytest.raises(ValueError, match="token segments"):
        segments_to_group_spans(merge_repeats([0, 1, 0]), [(0, 2)], 0, 3)


# ---------------------------------------------------------------------------
# Frame → time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_samples, expected", [(140320, 220), (640, 1), (641, 2), (0, 0)]
)
def test_count_audio_frames(num_samples, expected):
    """Only frames that start inside the audio are kept."""
    assert count_audio_frames(num_samples, 640) == expected


def test_frames_to_time_ms():
    """Milliseconds are ints; the last end is clipped to the audio length."""
    assert frames_to_time([(0, 6), (6, 10)], 640, 16000, 6000) == [
        (0, 240),
        (240, 375),
    ]


def test_frames_to_time_seconds():
    """Seconds are floats."""
    times = frames_to_time([(0, 6), (6, 10)], 640, 16000, 6400, return_seconds=True)
    assert times == [(0.0, 0.24), (0.24, 0.4)]
    assert all(isinstance(t, float) for span in times for t in span)


# ---------------------------------------------------------------------------
# Stride math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature_stride, add_adapter, adapter_stride, num_layers, expected",
    [
        (2, True, 2, 1, 640),  # obadx/muaalem-model-v3_2 → 40 ms
        (2, False, 2, 1, 320),  # no adapter → 20 ms
        (2, True, 2, 2, 1280),
    ],
)
def test_w2v2_bert_samples_per_frame(
    feature_stride, add_adapter, adapter_stride, num_layers, expected
):
    """Hop × feature stride × adapter downsampling."""
    assert (
        w2v2_bert_samples_per_frame(
            feature_stride, add_adapter, adapter_stride, num_layers
        )
        == expected
    )


@pytest.mark.parametrize(
    "window_stride, subsampling_factor, expected",
    [(0.01, 4, 640), (0.01, 8, 1280)],
)
def test_fastconformer_samples_per_frame(window_stride, subsampling_factor, expected):
    """Mel hop × encoder subsampling."""
    assert (
        fastconformer_samples_per_frame(window_stride, 16000, subsampling_factor)
        == expected
    )


# ---------------------------------------------------------------------------
# Viterbi alignment (torchaudio, no model)
# ---------------------------------------------------------------------------


def test_align_tokens_follows_peaky_emission():
    """With a near one-hot emission the path follows the argmax."""
    expected = [0, 1, 1, 0, 2, 0, 2, 0]
    emission = torch.full((len(expected), 3), -20.0)
    emission[torch.arange(len(expected)), torch.tensor(expected)] = 0.0
    assert align_tokens(emission, [1, 2, 2], blank_id=0) == expected


def test_align_tokens_too_short():
    """Too few frames for the targets raises a clear error."""
    emission = torch.log_softmax(torch.randn(2, 3), dim=-1)
    with pytest.raises(ValueError, match="too short"):
        align_tokens(emission, [1, 1], blank_id=0)


# ---------------------------------------------------------------------------
# Smoke test (downloads a model)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1")
def test_force_align_smoke_w2v2_bert():
    """End-to-end on ``test.wav``: one contiguous span per group, in range.

    ``test.wav`` has no reference transcript, so the model's own greedy CTC
    output is used as the phoneme script.
    """
    from prepare_quran_dataset.forced_aligner import force_align
    from prepare_quran_dataset.forced_aligner.forced_aligner import (
        compute_emissions,
        load_aligner_model,
        load_audio,
    )

    model_id = "obadx/muaalem-model-v3_2"
    aligner = load_aligner_model("w2v2-bert", model_id)
    wav = load_audio(TEST_WAV, aligner.sampling_rate)
    (emission,) = compute_emissions(aligner, [wav])
    id_to_token = {i: t for t, i in aligner.phoneme_vocab.items()}
    ids = torch.unique_consecutive(emission.argmax(-1)).tolist()
    phonemes = "".join(id_to_token[i] for i in ids if i != aligner.blank_id)
    groups = split_phoneme_groups(phonemes)
    assert groups

    (spans_ms,) = force_align([TEST_WAV], [phonemes], "w2v2-bert", model_id)
    (spans_s,) = force_align(
        [TEST_WAV.read_bytes()], [phonemes], "w2v2-bert", model_id, return_seconds=True
    )
    duration_ms = round(len(wav) * 1000 / aligner.sampling_rate)

    assert len(spans_ms) == len(groups) == len(spans_s)
    assert spans_ms[0][0] == 0
    assert spans_ms[-1][1] == duration_ms
    for start, end in spans_ms:
        assert isinstance(start, int) and 0 <= start < end <= duration_ms
    for (_, end), (start, _) in zip(spans_ms, spans_ms[1:]):
        assert end == start
    for (s_ms, e_ms), (s_s, e_s) in zip(spans_ms, spans_s):
        assert isinstance(s_s, float)
        assert abs(s_s * 1000 - s_ms) <= 1 and abs(e_s * 1000 - e_ms) <= 1
