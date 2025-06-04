from pathlib import Path
import asyncio

import librosa
from datasets import Features, IterableDataset, load_dataset, Audio, Value, Sequence
import torch
from recitations_segmenter import (
    segment_recitations,
    clean_speech_intervals,
    W2vBSegmentationOutput,
)

from ..construct.data_classes import Moshaf, Reciter
from .tarteel import tarteel_transcribe


READ_FEATURES = Features(
    {
        "audio": Audio(decode=False),  # We will read auiofiles manyally
        "moshaf_id": Value(dtype="string"),
        "moshaf_name": Value(dtype="string"),
        "reciter_id": Value(dtype="int32"),
        "reciter_arabic_name": Value(dtype="string"),
        "reciter_english_name": Value(dtype="string"),
        "sura_or_aya_index": Value(dtype="string"),
        "index_type": Value(dtype="string"),
        "sample_rate": Value(dtype="int32"),
        "duration_minutes": Value(dtype="float32"),
    }
)

OUT_FEATURES = Features(
    {
        "audio": Audio(sampling_rate=16000),  # We will read auiofiles manyally
        "segment_index": Value(dtype="string"),
        "tarteel_transcript": Sequence(feature=Value(dtype="string")),
        "moshaf_id": Value(dtype="string"),
        "moshaf_name": Value(dtype="string"),
        "reciter_id": Value(dtype="int32"),
        "reciter_arabic_name": Value(dtype="string"),
        "reciter_english_name": Value(dtype="string"),
        "sura_or_aya_index": Value(dtype="string"),
        "index_type": Value(dtype="string"),
        "sample_rate": Value(dtype="int32"),
        "duration_seconds": Value(dtype="float32"),
    }
)

SEGMET_PARAMS = {
    "mujawad": {
        "min_silence_duration_ms": 650,
        "min_speech_duration_ms": 1000,
        "pad_duration_ms": 900,
    },
    "above_murattal": {
        "min_silence_duration_ms": 150,
        "min_speech_duration_ms": 900,
        "pad_duration_ms": 800,
    },
    "murattal": {
        "min_silence_duration_ms": 30,
        "min_speech_duration_ms": 700,
        "pad_duration_ms": 800,
    },
    "hadr": {
        "min_silence_duration_ms": 0,
        "min_speech_duration_ms": 0,
        "pad_duration_ms": 400,
    },
}


def librosa_mono_decoder(batch, sample_rate=16000, alias_start_sec: float = 0):
    """Loading aduio file and downsample it to 16000"""
    alias_samples = int(alias_start_sec * sample_rate)
    audio_data = []
    durations: list[float] = []
    for audio in batch["audio"]:
        audio_path = audio["path"]
        try:
            # Load as mono with original sample rate
            waveform, _ = librosa.core.load(
                audio_path,
                sr=sample_rate,
                mono=True,  # Force mono conversion
            )

            audio_data.append(
                {
                    "array": waveform[alias_samples:],
                    "sampling_rate": sample_rate,
                    "path": audio_path,
                    "bytes": None,  # solving bug for new dataset version 3.2.2
                }
            )
        except Exception as e:
            print(f"⚠️ Failed {audio_path}: {str(e)}")
            raise e

    return {"audio": audio_data, "sample_rate": [sample_rate] * len(audio_data)}


def segment_batch(
    batch: dict,
    batch_ids: list[int],
    moshaf: Moshaf,
    segment_model,
    segment_feature_extractor,
    batch_size=32,
    sample_rate=16000,
    dtype=torch.bfloat16,
    device="cuda",
    cache_dir: str | Path = ".segment_cache",
) -> dict:
    """Segmenting speech intervals using وقف"""

    # caching results for every moshaf in separate directory
    batch_cache_dir = Path(cache_dir) / moshaf.id / f"batch_{batch_ids[-1]:03d}"
    batch_cache_dir.mkdir(exist_ok=True, parents=True)

    waves = [
        torch.tensor(batch["audio"][idx]["array"], dtype=torch.float32)
        for idx in range(len(batch["audio"]))
    ]

    # Segmenting waves using وقف
    outs = segment_recitations(
        waves,
        segment_model,
        segment_feature_extractor,
        batch_size=batch_size,
        device=device,
        cache_dir=batch_cache_dir,
    )
    clean_outs: list[W2vBSegmentationOutput] = []
    for idx, out in enumerate(outs):
        try:
            clean_out = clean_speech_intervals(
                out.speech_intervals,
                out.is_complete,
                **SEGMET_PARAMS[moshaf.recitation_speed],
            )
            clean_outs.append(clean_out)
        except Exception as e:
            print(
                f"Error while cleaning speech intervals of moshaf: `{moshaf.id}` of file: `{batch['sura_or_aya_index'][idx]}`"
            )
            raise e

    # removing `duration_minutes` columns and rewriting `audio` column
    original_keys = set(batch.keys()) - {"audio", "duration_minutes"}
    new_batch = {"audio": [], "segment_index": [], "duration_seconds": []}
    for key in original_keys:
        new_batch[key] = []
    # Devising recitations into segments
    for idx, clean_out in enumerate(clean_outs):
        new_audios = []
        segment_ids = []
        durations_seconds = []
        for span_idx, span in enumerate(clean_out.clean_speech_intervals):
            new_audios.append(
                {
                    "array": batch["audio"][idx]["array"][span[0] : span[1]],
                    "sampling_rate": 16000,
                }
            )
            durations_seconds.append(len(new_audios[-1]["array"]) / sample_rate)
            segment_ids.append(f"{batch['sura_or_aya_index'][idx]}.{span_idx:04d}")

        # Organizing batch
        new_batch["audio"] += new_audios
        new_batch["duration_seconds"] += durations_seconds
        new_batch["segment_index"] += segment_ids
        for key in original_keys:
            new_batch[key] += [batch[key][idx]] * len(segment_ids)

    return new_batch


def tarteel_transcribe_batch(
    batch: dict,
    vllm_endpoint="http://localhost:8000/v1",
    timeout_sec=300,
    chunk_overlap_sec=10,
    max_len_sec=30,
    sample_rate=16000,
) -> dict:
    async def async_main(waves):
        tasks = [
            tarteel_transcribe(
                wave,
                vllm_endpoint=vllm_endpoint,
                timeout_sec=timeout_sec,
                chunck_overlap_sec=chunk_overlap_sec,
                max_len_sec=max_len_sec,
                sample_rate=sample_rate,
            )
            for wave in waves
        ]
        outs = await asyncio.gather(*tasks)
        return outs

    waves = [batch["audio"][idx]["array"] for idx in range(len(batch["audio"]))]
    transcritps = asyncio.run(async_main(waves))

    return {"tarteel_transcript": transcritps}


def process_moshaf_tracks(
    moshaf: Moshaf,
    dataset_path: str | Path,
    read_moshaf_features: Features = READ_FEATURES,
    out_moshaf_features: Features = READ_FEATURES,
    loop_batch_size=4,
    sample_rate=16000,
    tarteel_batch_size=64,
    segment_batch_size=64,
    segment_device="cuda",
    segment_model=None,
    segment_feature_extractor=None,
    segment_cache_dir=".segment_cache",
    tarteel_timeout_sec=300,
    tarteel_chunk_overlap_sec=10,
    tarteel_max_len_sec=30,
    tarteel_vllm_endpont="http://localhost:8000/v1",
) -> IterableDataset:
    """ """
    dataset_path = Path(dataset_path)
    ds = load_dataset(
        "audiofolder",
        data_dir=dataset_path / moshaf.path,
        split="train",
        streaming=True,
        features=read_moshaf_features,
    )

    # Loading audio files nad downsample it to 16000
    ds = ds.map(
        librosa_mono_decoder,
        batched=True,
        batch_size=10,
        features=read_moshaf_features,
        fn_kwargs={
            "sample_rate": sample_rate,
            "alias_start_sec": moshaf.alias_start_sec,
        },
    )

    # Segment Rcitations on puase وقف
    ds = ds.map(
        segment_batch,
        batched=True,
        batch_size=loop_batch_size,
        # features=out_moshaf_features,
        with_indices=True,
        remove_columns=ds.column_names,
        fn_kwargs={
            "moshaf": moshaf,
            "segment_model": segment_model,
            "segment_feature_extractor": segment_feature_extractor,
            "device": segment_device,
            "batch_size": segment_batch_size,
            "sample_rate": sample_rate,
            "cache_dir": segment_cache_dir,
        },
    )

    # Transcripe using Tarteel
    ds = ds.map(
        tarteel_transcribe_batch,
        batched=True,
        batch_size=tarteel_batch_size,
        # features=out_moshaf_features,
        fn_kwargs={
            "vllm_endpoint": tarteel_vllm_endpont,
            "timeout_sec": tarteel_timeout_sec,
            "chunk_overlap_sec": tarteel_chunk_overlap_sec,
            "max_len_sec": tarteel_max_len_sec,
            "sample_rate": sample_rate,
        },
    )

    return ds
