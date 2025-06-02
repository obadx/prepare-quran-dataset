from typing import Any
from dataclasses import dataclass

import numpy as np


@dataclass
class TruncateOutput:
    audio: list[dict[str, Any]]
    speech_intervals_sec: list[np.ndarray]
    speech_intervals_samples: list[np.ndarray]

    """
    audio: list({'array': np.ndarray, 'sampling_rate': int})
    """


def chunk_wave(
    wave: np.ndarray,
    chunk_overlap=16000,
    max_len_samples=480000,
) -> list[np.array]:
    """Moving winodw truncatation arlogrith where the window size is `max_len_samples`"""

    window = max_len_samples
    step = window - chunk_overlap
    num_items = int(np.ceil(max(0, len(wave) - window) / (window - chunk_overlap))) + 1

    wav_chunks = []
    start = 0
    for idx in range(num_items):
        end = start + window
        wav_chunks.append(wave[start:end])
        start += step

    return wav_chunks


def merge_transcripts(trans: list[str]) -> str: ...


async def tarteel_transcript(
    wave: np.ndarray,
    sample_rate=16000,
    max_len_sec: float = 30,
    chunck_overlap_sec: float = 10,
    vllm_endpoint="",
):
    assert chunck_overlap_sec < max_len_sec
    max_len_samples = int(max_len_sec * sample_rate)
    chunck_overlap_samples = int(chunck_overlap_sec * sample_rate)

    # chunk waves
    wav_chunks = chunk_wave(
        wave, max_len=max_len_samples, chunk_overlap=chunck_overlap_samples
    )

    transcripts: list[str] = []
    # run parallel async requests

    # merge transcripts
    merged_trans = merge_transcripts(transcripts)
