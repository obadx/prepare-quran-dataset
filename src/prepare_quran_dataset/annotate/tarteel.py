from typing import Any
from dataclasses import dataclass
import io
import asyncio

from openai import AsyncOpenAI, APIConnectionError
import soundfile as sf
import stamina

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


async def transcribe_chunk(
    client: AsyncOpenAI,
    chunk: np.ndarray,
    sample_rate=16000,
    model="tarteel-ai/whisper-base-ar-quran",
) -> str:
    buffer = io.BytesIO()

    # Handle different dtypes and ensure proper audio format
    if chunk.dtype != np.int16 and chunk.dtype != np.float32:
        if np.issubdtype(chunk.dtype, np.integer):
            chunk = chunk.astype(np.float32) / np.iinfo(chunk.dtype).max
        else:
            chunk = chunk.astype(np.float32)

    # Write audio to buffer
    sf.write(
        buffer,
        chunk,
        sample_rate,
        format="WAV",  # Explicitly specify WAV format
    )
    wav_bytes = buffer.getvalue()

    # Send request to ASR endpoint
    out = await client.audio.transcriptions.create(
        model=model,
        file=wav_bytes,
    )
    return out.text


def compute_prefix_function(pattern):
    pi = [0] * len(pattern)
    k = 0
    for q in range(1, len(pattern)):
        while k > 0 and pattern[k] != pattern[q]:
            k = pi[k - 1]
        if pattern[k] == pattern[q]:
            k += 1
        pi[q] = k
    return pi


class SmallTarteelOverlap(Exception): ...


def merge_transcripts(
    texts,
    min_merge_chars=5,
    end_truncate_words=1,
    start_truncate_words=1,
):
    """Merge overlaping transcripts using Knuth-Morris-Pratt (KMP) pi Algorithm
    https://cp-algorithms.com/string/prefix-function.html

    Args:
        text (list[str]):
        min_merge_chars (int): the minum overlap length else raises SmallTarteelOverlap
        end_truncate_words (int): removes the `end_truncate_words` form the end of every
            transcript to avoid partial or wrong tnrascript  at the end of the chunked word
        start_truncate_words (int): removes the `start_truncate_words` form the start of every
            transcript to avoid partial or wrong tnrascript  at the start of the chunked word
    """
    # Removing trailing words and starting words
    # to avoide (part of word wrong transcrips)
    if len(texts) > 1:
        # Not removing the start of the first transcript
        words = texts[0].split(" ")
        texts[0] = " ".join(words[: len(words) - end_truncate_words])
        for idx in range(1, len(texts) - 1):
            words = texts[idx].split(" ")
            texts[idx] = " ".join(
                words[start_truncate_words : len(words) - end_truncate_words]
            )
        # Not removing the tail of the last transcript
        texts[-1] = " ".join(texts[-1].split(" ")[start_truncate_words:])

    if not texts:
        return ""
    merged = texts[0]
    for i in range(1, len(texts)):
        next_text = texts[i]
        L = min(len(merged), len(next_text))
        s_tail = merged[-L:]
        pi = compute_prefix_function(next_text)
        state = 0  # number of matched chars
        for char in s_tail:
            # going batch to tha last matching prefix
            while state > 0 and next_text[state] != char:
                state = pi[state - 1]
            if next_text[state] == char:
                state += 1
        if state < min_merge_chars:
            raise SmallTarteelOverlap("Very Small overlap to merge on")
        merged += next_text[state:]
    return merged


@stamina.retry(on=APIConnectionError, attempts=3)
async def tarteel_transcribe(
    wave: np.ndarray,
    sample_rate=16000,
    max_len_sec: float = 30,
    chunck_overlap_sec: float = 10,
    vllm_endpoint="http://localhost:8000/v1",
    model="tarteel-ai/whisper-base-ar-quran",
    timeout_sec=300,
) -> list[str]:
    assert chunck_overlap_sec < max_len_sec
    max_len_samples = int(max_len_sec * sample_rate)
    chunck_overlap_samples = int(chunck_overlap_sec * sample_rate)

    # chunk waves
    wav_chunks = chunk_wave(
        wave, max_len_samples=max_len_samples, chunk_overlap=chunck_overlap_samples
    )

    transcripts: list[str] = []
    client = AsyncOpenAI(api_key="None", base_url=vllm_endpoint, timeout=timeout_sec)

    # Run concerent async requests
    tasks = [
        transcribe_chunk(client, chunk, model=model, sample_rate=sample_rate)
        for chunk in wav_chunks
    ]
    transcripts = await asyncio.gather(*tasks)

    return transcripts
