"""Phoneme-group CTC forced aligner for the Muaalem multi-level CTC models.

Given audio files and their *transcribed phoneme* scripts, this module returns a
``(start, end)`` timestamp for every **phoneme group** (e.g. ``قَ``, ``اا``,
``لَ``) as produced by :func:`quran_transcript.chunck_phonemes`, not for every
individual phoneme.  Only the ``phonemes`` CTC level is used; Sifat levels are
ignored.

Reference: https://github.com/MahmoudAshraf97/ctc-forced-aligner (BSD-2)

Pipeline::

    phonemes ─► split_phoneme_groups ─► groups_to_token_ids ─┐
                                                              ├─► align_tokens
    audio ─► load_audio ─► compute_emissions (log-softmax) ──┘        │
                                                                      ▼
    frames_to_time ◄── segments_to_group_spans ◄── merge_repeats ◄── path

Group boundaries follow the reference "adjacent-blank midpoint" rule with one
fix: the reference counts the midpoint frame twice (once as the end of a group
and once as the start of the next), so consecutive groups overlap by one
frame.  Here every boundary is a single frame index shared by both groups,
so the spans are contiguous, non-overlapping and cover the whole audio.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
import torch
import torchaudio.functional as F
from quran_transcript import chunck_phonemes

ModelType = Literal["w2v2-bert", "fastconformer"]

PHONEMES_LEVEL = "phonemes"

# ``SeamlessM4TFeatureExtractor`` (the w2v2-bert front-end) hard-codes a
# 10 ms hop (160 samples at 16 kHz) when computing its fbank features.
SEAMLESS_M4T_HOP_LENGTH = 160


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A run of identical labels in a frame-level CTC path.

    Args:
        label: Token id repeated over the run.
        start: First frame of the run (inclusive).
        end: Frame after the last frame of the run (exclusive).

    Example:
        >>> Segment(label=5, start=3, end=5).length
        2
    """

    label: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class AlignerModel:
    """A loaded CTC model together with everything needed to align with it.

    Args:
        model_type: Which architecture ``model`` is.
        model: The multi-level CTC model, in ``eval`` mode on ``device``.
        feature_extractor: HF feature extractor for ``"w2v2-bert"``;
            ``None`` for ``"fastconformer"`` (which consumes raw audio).
        phoneme_vocab: ``phonemes``-level token to id mapping.
        blank_id: CTC blank id (``[PAD]``).
        samples_per_frame: Audio samples covered by one output (emission) frame.
        sampling_rate: Sampling rate the model expects, in Hz.
        device: Device the model runs on.
    """

    model_type: ModelType
    model: torch.nn.Module
    feature_extractor: Any | None
    phoneme_vocab: dict[str, int]
    blank_id: int
    samples_per_frame: int
    sampling_rate: int
    device: torch.device


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def split_phoneme_groups(phonemes: str) -> list[str]:
    """Split a phonetic script into phoneme groups, dropping word spaces.

    Spaces are not part of the CTC vocabulary, so they are removed before
    chunking.  Raises if the groups do not cover the script exactly, which
    means the script holds characters ``chunck_phonemes`` cannot place.

    Args:
        phonemes: Phonetic script, e.g. the ``phonemes`` field of
            :func:`quran_transcript.quran_phonetizer`.

    Returns:
        Phoneme groups in order; ``"".join(groups)`` equals ``phonemes``
        without spaces.

    Raises:
        ValueError: If some characters are not covered by any group.

    Example:
        >>> split_phoneme_groups("قُل هُوَ")
        ['قُ', 'ل', 'هُ', 'وَ']
    """
    script = phonemes.replace(" ", "")
    groups = chunck_phonemes(script)
    if "".join(groups) != script:
        raise ValueError(
            f"Phoneme groups do not cover the phonetic script exactly: "
            f"script=`{script}`, groups={groups}"
        )
    return groups


def groups_to_token_ids(
    groups: list[str], vocab: dict[str, int]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Convert phoneme groups to CTC token ids and per-group token ranges.

    Every phoneme character is one token.

    Args:
        groups: Phoneme groups from :func:`split_phoneme_groups`.
        vocab: ``phonemes``-level token to id mapping.

    Returns:
        ``(token_ids, group_ranges)`` where ``group_ranges[i] = (start, end)``
        is the half-open range of ``token_ids`` that belongs to ``groups[i]``.

    Raises:
        ValueError: If a character is not in ``vocab``.

    Example:
        >>> groups_to_token_ids(["قُ", "ل"], {"ق": 2, "ُ": 3, "ل": 4})
        ([2, 3, 4], [(0, 2), (2, 3)])
    """
    token_ids: list[int] = []
    group_ranges: list[tuple[int, int]] = []
    for group in groups:
        start = len(token_ids)
        for char in group:
            if char not in vocab:
                raise ValueError(
                    f"Phoneme `{char}` (U+{ord(char):04X}) in group `{group}` "
                    f"is not in the model vocabulary"
                )
            token_ids.append(vocab[char])
        group_ranges.append((start, len(token_ids)))
    return token_ids, group_ranges


def count_required_frames(token_ids: list[int]) -> int:
    """Minimum number of emission frames CTC needs to emit ``token_ids``.

    Consecutive identical tokens must be separated by a blank frame, so each
    repeat costs one extra frame.

    Args:
        token_ids: Target token ids.

    Returns:
        ``len(token_ids)`` plus the number of consecutive repeats.

    Example:
        >>> count_required_frames([7, 7, 3, 9, 9, 9])
        9
    """
    repeats = sum(1 for prev, cur in zip(token_ids, token_ids[1:]) if prev == cur)
    return len(token_ids) + repeats


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def load_audio(source: Path | str | bytes, sampling_rate: int) -> np.ndarray:
    """Load an audio file or in-memory audio as a mono float32 waveform.

    Args:
        source: Path to an audio file, or the raw bytes of an encoded file.
        sampling_rate: Target sampling rate; audio is resampled to it.

    Returns:
        1-D ``float32`` waveform.

    Example:
        >>> wav = load_audio(Path("assets/audio-sampels/test.wav"), 16000)
        >>> wav.ndim
        1
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    wav, _ = librosa.load(source, sr=sampling_rate, mono=True)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Frame / time math
# ---------------------------------------------------------------------------


def w2v2_bert_samples_per_frame(
    feature_stride: int,
    add_adapter: bool,
    adapter_stride: int,
    num_adapter_layers: int,
    hop_length: int = SEAMLESS_M4T_HOP_LENGTH,
) -> int:
    """Audio samples per output frame of a w2v2-bert model.

    The feature extractor emits one fbank frame per ``hop_length`` samples and
    stacks ``feature_stride`` of them; each adapter layer then downsamples by
    ``adapter_stride``.

    Args:
        feature_stride: ``stride`` of the feature extractor.
        add_adapter: Whether the model has an adapter.
        adapter_stride: Stride of each adapter layer.
        num_adapter_layers: Number of adapter layers.
        hop_length: Fbank hop in samples.

    Returns:
        Samples per emission frame.

    Example:
        >>> w2v2_bert_samples_per_frame(2, True, 2, 1)  # 40 ms at 16 kHz
        640
    """
    adapter_factor = adapter_stride**num_adapter_layers if add_adapter else 1
    return hop_length * feature_stride * adapter_factor


def fastconformer_samples_per_frame(
    window_stride: float, sampling_rate: int, subsampling_factor: int
) -> int:
    """Audio samples per output frame of a FastConformer model.

    Args:
        window_stride: Mel hop in seconds.
        sampling_rate: Sampling rate in Hz.
        subsampling_factor: Encoder temporal subsampling factor.

    Returns:
        Samples per emission frame.

    Example:
        >>> fastconformer_samples_per_frame(0.01, 16000, 4)  # 40 ms
        640
    """
    return round(window_stride * sampling_rate) * subsampling_factor


def count_audio_frames(num_samples: int, samples_per_frame: int) -> int:
    """Number of emission frames that start inside the audio.

    Models can emit a few extra frames past the end of the audio (padding in
    the front-end); aligning tokens into them would give groups that start
    after the audio ends.

    Args:
        num_samples: Audio length in samples.
        samples_per_frame: Audio samples per emission frame.

    Returns:
        ``ceil(num_samples / samples_per_frame)``.

    Example:
        >>> count_audio_frames(140320, 640)
        220
    """
    return math.ceil(num_samples / samples_per_frame)


def frames_to_time(
    spans: list[tuple[int, int]],
    samples_per_frame: int,
    sampling_rate: int,
    num_samples: int,
    return_seconds: bool = False,
) -> list[tuple[int, int]] | list[tuple[float, float]]:
    """Convert half-open frame spans to time, clipped to the audio length.

    Args:
        spans: ``(start_frame, end_frame)`` per group, end exclusive.
        samples_per_frame: Audio samples per emission frame.
        sampling_rate: Sampling rate in Hz.
        num_samples: Length of the audio in samples; ends are clipped to it.
        return_seconds: ``False`` → integer milliseconds, ``True`` → float
            seconds.

    Returns:
        ``(start, end)`` per group.

    Example:
        >>> frames_to_time([(0, 6), (6, 10)], 640, 16000, 6000)
        [(0, 240), (240, 375)]
        >>> frames_to_time([(0, 6)], 640, 16000, 6000, return_seconds=True)
        [(0.0, 0.24)]
    """
    times = []
    for start_frame, end_frame in spans:
        start = min(start_frame * samples_per_frame, num_samples)
        end = min(end_frame * samples_per_frame, num_samples)
        if return_seconds:
            times.append((start / sampling_rate, end / sampling_rate))
        else:
            times.append(
                (round(start * 1000 / sampling_rate), round(end * 1000 / sampling_rate))
            )
    return times


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_tokens(
    emission: torch.Tensor, token_ids: list[int], blank_id: int
) -> list[int]:
    """Viterbi CTC forced alignment of ``token_ids`` over one emission.

    Args:
        emission: Log-probabilities of shape ``(num_frames, vocab_size)``.
        token_ids: Target token ids (no blanks).
        blank_id: CTC blank id.

    Returns:
        Frame-level label path of length ``num_frames``.

    Raises:
        ValueError: If the emission is too short for the targets.

    Example:
        >>> emission = torch.log_softmax(torch.randn(10, 5), dim=-1)
        >>> len(align_tokens(emission, [1, 2, 2], blank_id=0))
        10
    """
    required = count_required_frames(token_ids)
    if emission.shape[0] < required:
        raise ValueError(
            f"Audio too short to align: {emission.shape[0]} frames, but the "
            f"phonemes need at least {required}"
        )
    path, _ = F.forced_align(
        emission.unsqueeze(0).float().cpu(),
        torch.tensor([token_ids], dtype=torch.int32),
        blank=blank_id,
    )
    return path[0].tolist()


def merge_repeats(path: list[int]) -> list[Segment]:
    """Collapse a frame-level path into runs of identical labels.

    Args:
        path: Label per frame.

    Returns:
        One :class:`Segment` per run, in order.

    Example:
        >>> merge_repeats([0, 0, 5, 5, 0, 7])
        [Segment(label=0, start=0, end=2), Segment(label=5, start=2, end=4), Segment(label=0, start=4, end=5), Segment(label=7, start=5, end=6)]
    """
    segments: list[Segment] = []
    start = 0
    for idx in range(1, len(path) + 1):
        if idx == len(path) or path[idx] != path[start]:
            segments.append(Segment(label=path[start], start=start, end=idx))
            start = idx
    return segments


def segments_to_group_spans(
    segments: list[Segment],
    group_ranges: list[tuple[int, int]],
    blank_id: int,
    num_frames: int,
) -> list[tuple[int, int]]:
    """Turn path segments into one contiguous frame span per phoneme group.

    Each non-blank segment is one target token (CTC separates repeated tokens
    by a blank, so no two tokens share a segment).  A group's core span runs
    from its first token's start to its last token's end.  The blank frames
    between two groups are split at their midpoint; the first group starts at
    frame ``0`` and the last group ends at ``num_frames``.

    Args:
        segments: Output of :func:`merge_repeats`.
        group_ranges: Token range of each group, from
            :func:`groups_to_token_ids`.
        blank_id: CTC blank id.
        num_frames: End frame of the last group, normally the number of
            frames the audio spans (see :func:`count_audio_frames`); may be
            larger than ``len(path)``.

    Returns:
        ``(start_frame, end_frame)`` per group, end exclusive.  Spans are
        contiguous: ``spans[i][1] == spans[i + 1][0]``.

    Raises:
        ValueError: If the number of token segments does not match the
            number of tokens.

    Example:
        >>> path = [0, 0, 0, 1, 2, 0, 0, 0, 3, 0, 0, 4, 2, 0, 0, 0, 0, 0]
        >>> segments_to_group_spans(
        ...     merge_repeats(path), [(0, 2), (2, 3), (3, 5)], 0, len(path)
        ... )
        [(0, 6), (6, 10), (10, 18)]
    """
    token_segments = [seg for seg in segments if seg.label != blank_id]
    num_tokens = group_ranges[-1][1] if group_ranges else 0
    if len(token_segments) != num_tokens:
        raise ValueError(
            f"Alignment path has {len(token_segments)} token segments, "
            f"expected {num_tokens}"
        )

    cores = [
        (token_segments[start].start, token_segments[end - 1].end)
        for start, end in group_ranges
    ]
    boundaries = [0]
    for (_, prev_end), (next_start, _) in zip(cores, cores[1:]):
        boundaries.append((prev_end + next_start) // 2)
    boundaries.append(num_frames)
    return list(zip(boundaries[:-1], boundaries[1:]))


# ---------------------------------------------------------------------------
# Model loading and emissions
# ---------------------------------------------------------------------------


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_phoneme_vocab(
    model_name_or_path: str,
    tokenizer_cls: Callable[[str], Any],
    build_vocab: Callable[[str], None],
    expected_size: int,
) -> dict[str, int]:
    """Load the ``phonemes`` vocab shipped with a model, or rebuild it.

    Hub repos ship a ``vocab.json``; local training checkpoints may not.  When
    the tokenizer cannot be loaded from ``model_name_or_path``, the vocab is
    rebuilt with the package's own ``build_quran_phoneme_script_vocab``.

    Args:
        model_name_or_path: Hub repo id or local checkpoint directory.
        tokenizer_cls: ``MultiLevelTokenizer`` of the model's package.
        build_vocab: ``build_quran_phoneme_script_vocab`` of the model's
            package (writes the multi-level vocab JSON to a path).
        expected_size: ``config.level_to_vocab_size["phonemes"]``.

    Returns:
        ``phonemes``-level token to id mapping.

    Raises:
        ValueError: If the vocab size does not match the model head.

    Example:
        >>> from prepare_quran_dataset.modeling.multi_level_tokenizer import (
        ...     MultiLevelTokenizer,
        ... )
        >>> from prepare_quran_dataset.modeling.vocab import (
        ...     build_quran_phoneme_script_vocab,
        ... )
        >>> vocab = load_phoneme_vocab(
        ...     "obadx/muaalem-model-v3_2",
        ...     MultiLevelTokenizer,
        ...     build_quran_phoneme_script_vocab,
        ...     expected_size=43,
        ... )
        >>> vocab["[PAD]"]
        0
    """
    try:
        vocab = tokenizer_cls(model_name_or_path).vocab[PHONEMES_LEVEL]
    except (OSError, TypeError, ValueError):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_path = str(Path(tmp_dir) / "vocab.json")
            build_vocab(vocab_path)
            with open(vocab_path, encoding="utf-8") as f:
                vocab = json.load(f)[PHONEMES_LEVEL]
    if len(vocab) != expected_size:
        raise ValueError(
            f"Phoneme vocab has {len(vocab)} tokens but the model's `phonemes` "
            f"head has {expected_size}"
        )
    return vocab


def _load_w2v2_bert(model_name_or_path: str, device: torch.device) -> AlignerModel:
    from transformers import AutoFeatureExtractor

    from ..modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
    from ..modeling.multi_level_tokenizer import MultiLevelTokenizer
    from ..modeling.vocab import build_quran_phoneme_script_vocab

    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_name_or_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    config = model.config
    return AlignerModel(
        model_type="w2v2-bert",
        model=model.to(device).eval(),
        feature_extractor=feature_extractor,
        phoneme_vocab=load_phoneme_vocab(
            model_name_or_path,
            MultiLevelTokenizer,
            build_quran_phoneme_script_vocab,
            config.level_to_vocab_size[PHONEMES_LEVEL],
        ),
        blank_id=config.pad_token_id,
        samples_per_frame=w2v2_bert_samples_per_frame(
            feature_stride=feature_extractor.stride,
            add_adapter=config.add_adapter,
            adapter_stride=config.adapter_stride,
            num_adapter_layers=config.num_adapter_layers,
        ),
        sampling_rate=feature_extractor.sampling_rate,
        device=device,
    )


def _load_fastconformer(
    model_name_or_path: str, device: torch.device
) -> AlignerModel:
    # Imported lazily: this package pulls in NeMo.
    from ..modeling_fastconformer_cache_aware import (
        FastConformerCacheAwareMultilevelCTC,
    )
    from ..modeling_fastconformer_cache_aware.multi_level_tokenizer import (
        MultiLevelTokenizer,
    )
    from ..modeling_fastconformer_cache_aware.vocab import (
        PAD_TOKEN_IDX,
        build_quran_phoneme_script_vocab,
    )

    model = FastConformerCacheAwareMultilevelCTC.from_pretrained(model_name_or_path)
    model.to(device).eval()
    model.processor.to(device)
    processor = model.processor
    return AlignerModel(
        model_type="fastconformer",
        model=model,
        feature_extractor=None,
        phoneme_vocab=load_phoneme_vocab(
            model_name_or_path,
            MultiLevelTokenizer,
            build_quran_phoneme_script_vocab,
            model.config.level_to_vocab_size[PHONEMES_LEVEL],
        ),
        blank_id=PAD_TOKEN_IDX,
        samples_per_frame=fastconformer_samples_per_frame(
            window_stride=processor.window_stride,
            sampling_rate=processor.sample_rate,
            subsampling_factor=model.config.subsampling_factor,
        ),
        sampling_rate=processor.sample_rate,
        device=device,
    )


@lru_cache(maxsize=2)
def load_aligner_model(
    model_type: ModelType, model_name_or_path: str, device: str | None = None
) -> AlignerModel:
    """Load (and cache) a model for alignment from the Hub or a local path.

    Args:
        model_type: ``"w2v2-bert"`` or ``"fastconformer"``.
        model_name_or_path: Hub repo id or local checkpoint directory.
        device: Torch device name; defaults to CUDA when available.

    Returns:
        The loaded :class:`AlignerModel`.

    Raises:
        ValueError: For an unknown ``model_type``.

    Example:
        >>> aligner = load_aligner_model("w2v2-bert", "obadx/muaalem-model-v3_2")
        >>> aligner.samples_per_frame
        640
    """
    torch_device = torch.device(device) if device else _default_device()
    if model_type == "w2v2-bert":
        return _load_w2v2_bert(model_name_or_path, torch_device)
    if model_type == "fastconformer":
        return _load_fastconformer(model_name_or_path, torch_device)
    raise ValueError(
        f"Unknown model_type `{model_type}`; expected `w2v2-bert` or `fastconformer`"
    )


def _w2v2_bert_batch_logits(
    aligner: AlignerModel, waves: list[np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = aligner.feature_extractor(
        waves,
        sampling_rate=aligner.sampling_rate,
        padding="longest",
        return_tensors="pt",
    ).to(aligner.device)
    out = aligner.model(
        input_features=inputs["input_features"],
        attention_mask=inputs["attention_mask"],
    )
    lengths = aligner.model._get_feat_extract_output_lengths(
        inputs["attention_mask"].sum(-1)
    )
    return out.logits[PHONEMES_LEVEL], lengths


def _fastconformer_batch_logits(
    aligner: AlignerModel, waves: list[np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(w) for w in waves)
    audio = torch.zeros(len(waves), max_len, dtype=torch.float32)
    for idx, wav in enumerate(waves):
        audio[idx, : len(wav)] = torch.from_numpy(wav)
    lengths = torch.tensor([len(w) for w in waves], dtype=torch.long)
    out = aligner.model(
        raw_audio=audio.to(aligner.device),
        audio_length=lengths.to(aligner.device),
        selected_levels={PHONEMES_LEVEL},
    )
    return out.logits[PHONEMES_LEVEL], out.encoder_lengths


@torch.no_grad()
def compute_emissions(
    aligner: AlignerModel, waves: list[np.ndarray]
) -> list[torch.Tensor]:
    """Run one padded batch and return per-sample ``phonemes`` log-probs.

    Args:
        aligner: Loaded model.
        waves: Mono waveforms at ``aligner.sampling_rate``.

    Returns:
        One CPU ``float32`` tensor of shape ``(num_frames_i, vocab_size)`` per
        waveform, trimmed to that waveform's valid frames (padding removed).
    """
    if aligner.model_type == "w2v2-bert":
        logits, lengths = _w2v2_bert_batch_logits(aligner, waves)
    else:
        logits, lengths = _fastconformer_batch_logits(aligner, waves)
    log_probs = torch.log_softmax(logits.float(), dim=-1).cpu()
    return [log_probs[idx, : int(n)] for idx, n in enumerate(lengths)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def force_align(
    audio_sources: list[Path | bytes],
    phonemse: list[str],
    model_type: ModelType,
    model_name_or_path: str,
    batch_size: int = 64,
    return_seconds: bool = False,
    samling_rate: int = 16000,
) -> list[list[tuple[int, int]]] | list[list[tuple[float, float]]]:
    """Timestamp every phoneme group of every audio by CTC forced alignment.

    Args:
        audio_sources: Audio file paths or raw encoded audio bytes.
        phonemse: Phonetic script per audio (spelling kept from the RFC).
        model_type: ``"w2v2-bert"`` or ``"fastconformer"``.
        model_name_or_path: Hub repo id or local checkpoint directory.
        batch_size: Number of audios per model forward pass.
        return_seconds: ``False`` → integer milliseconds, ``True`` → float
            seconds.
        samling_rate: Sampling rate audio is loaded at; must match the
            model's (spelling kept from the RFC).

    Returns:
        One list per audio with one ``(start, end)`` per phoneme group, in the
        order of :func:`quran_transcript.chunck_phonemes`.  An empty script
        gives an empty list.

    Raises:
        ValueError: On length mismatch, sampling-rate mismatch, unknown
            phonemes, or audio too short for its phonemes.

    Example:
        ``test.wav`` ends with 8:75 and starts 9:1 (… عَلِيمٌ بَرَاءَةٌ مِّنَ اللَّهِ
        وَرَسُولِهِ); its groups start ``ءِ``, ``ننننَ``, ``للَ``:

        >>> force_align(
        ...     [Path("assets/audio-sampels/test.wav")],
        ...     ["ءِننننَللَااهَبِكُللِشَيءِنعَلِۦۦمُ۾۾۾بَرَااااءَتُممممِنَللَااهِوَرَسُۥۥلِه"],
        ...     model_type="w2v2-bert",
        ...     model_name_or_path="obadx/muaalem-model-v3_2",
        ... )[0][:3]
        [(0, 440), (440, 1160), (1160, 1520)]
    """
    if len(audio_sources) != len(phonemse):
        raise ValueError(
            f"Got {len(audio_sources)} audio sources but {len(phonemse)} "
            f"phoneme scripts"
        )
    aligner = load_aligner_model(model_type, model_name_or_path)
    if samling_rate != aligner.sampling_rate:
        raise ValueError(
            f"samling_rate={samling_rate} but the model expects "
            f"{aligner.sampling_rate}"
        )

    targets = []
    for idx, script in enumerate(phonemse):
        try:
            groups = split_phoneme_groups(script)
            targets.append(groups_to_token_ids(groups, aligner.phoneme_vocab))
        except ValueError as e:
            raise ValueError(f"Sample {idx}: {e}") from e

    waves = [load_audio(src, samling_rate) for src in audio_sources]
    results: list[Any] = [[] for _ in waves]
    todo = [idx for idx, (token_ids, _) in enumerate(targets) if token_ids]
    # Sort by length so each batch carries as little padding as possible.
    todo.sort(key=lambda idx: len(waves[idx]))

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        emissions = compute_emissions(aligner, [waves[idx] for idx in batch])
        for idx, emission in zip(batch, emissions):
            token_ids, group_ranges = targets[idx]
            # Drop frames past the audio end, and stretch the last group to the
            # audio end when the model emits fewer frames than the audio spans.
            audio_frames = count_audio_frames(
                len(waves[idx]), aligner.samples_per_frame
            )
            try:
                path = align_tokens(
                    emission[:audio_frames], token_ids, aligner.blank_id
                )
            except ValueError as e:
                raise ValueError(f"Sample {idx}: {e}") from e
            spans = segments_to_group_spans(
                merge_repeats(path), group_ranges, aligner.blank_id, audio_frames
            )
            results[idx] = frames_to_time(
                spans,
                aligner.samples_per_frame,
                samling_rate,
                len(waves[idx]),
                return_seconds,
            )
    return results
