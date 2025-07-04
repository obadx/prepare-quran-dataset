from pathlib import Path
import asyncio
from typing import Optional
import logging

import librosa
from datasets import (
    Features,
    Dataset,
    IterableDataset,
    load_dataset,
    Audio,
    Value,
    Sequence,
)
import numpy as np
import torch
from recitations_segmenter import (
    segment_recitations,
    clean_speech_intervals,
    W2vBSegmentationOutput,
)

from ..construct.data_classes import Moshaf, Reciter
from .tarteel import tarteel_transcribe


def get_segment_format(sura_idx, aya_idx):
    return f"{sura_idx:03d}.{aya_idx:04d}"


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
        "timestamp_seconds": Sequence(feature=Value(dtype="float32")),
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

SEGMET_MOSHAF_PARAMS = {
    "19.0": {
        "min_silence_duration_ms": 30,
        "min_speech_duration_ms": 600,
        "pad_duration_ms": 380,
    },
    "4.0": {
        "min_silence_duration_ms": 300,
        "min_speech_duration_ms": 600,
        "pad_duration_ms": 700,
    },
    "3.0": {
        "min_silence_duration_ms": 300,
        "min_speech_duration_ms": 600,
        "pad_duration_ms": 440,
    },
    "6.0": {
        "min_silence_duration_ms": 300,
        "min_speech_duration_ms": 600,
        "pad_duration_ms": 560,
    },
    "30.0": {
        "min_silence_duration_ms": 300,
        "min_speech_duration_ms": 600,
        "pad_duration_ms": 800,
    },
}


# Moshaf Item thta loads the first channel only
FIRST_CHANNEL_ONLY_MOSHAF = {"4.0"}


def load_segment_intervals_from_cache(
    moshaf: Moshaf,
    moshaf_segment_cache_dir: str | Path,
    moshaf_metada_jsonl_path: str | Path,
) -> dict[str, np.array]:
    """return the intervals for each segment intervals for a single moshaf in seconds as dict:
    {'segment_index': np.array([start_seconds, end_seconsd])

    We are trying here to load intervals produces by the segmenter during data annotation

    """
    # the number of track used to feed segment index each iteration
    work_batch_size = 16
    ds = Dataset.from_json(str(moshaf_metada_jsonl_path))
    segment_dirs = sorted(
        Path(moshaf_segment_cache_dir).glob("*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    mode = segment_dirs[0].name.split("_")[0]  # either "batch" or "track"
    assert mode in {"batch", "track"}

    if moshaf.id in SEGMET_MOSHAF_PARAMS:
        segment_prams = SEGMET_MOSHAF_PARAMS[moshaf.id]
    else:
        segment_prams = SEGMET_PARAMS[moshaf.recitation_speed]

    logging.info(f"Segment Params of moshaf: {moshaf.id} are: {segment_prams}")
    segment_index_to_interval: dict[str, np.array] = {}

    if mode == "batch":
        # every item of the list is {'speech_intrvals': list[torch.tensor()], 'is_complete': list[bool]}
        idx_to_segment_loads: list[dict] = []
        for dir in segment_dirs:
            pt_file = list(dir.glob("*.pt"))[0]
            idx_to_segment_loads.append(torch.load(pt_file))

        for idx, sura_idx in enumerate(ds["sura_or_aya_index"]):
            sura_idx = int(sura_idx)
            batch_idx = idx // work_batch_size
            idx_in_batch = idx % work_batch_size

            speech_intervals = idx_to_segment_loads[batch_idx]["speech_intervals"][
                idx_in_batch
            ]
            is_complete = idx_to_segment_loads[batch_idx]["is_complete"][idx_in_batch]
            out = clean_speech_intervals(
                speech_intervals, is_complete, **segment_prams, return_seconds=True
            )
            for aya_idx, interval in enumerate(out.clean_speech_intervals):
                segment_index = get_segment_format(sura_idx, aya_idx)
                segment_index_to_interval[segment_index] = interval.numpy()

    elif mode == "track":
        # every item of the list is {'speech_intrvals': list[torch.tensor()], 'is_complete': list[bool]}
        idx_to_segment_loads: list[dict] = []
        # looping over every aya in this mode
        for dir in segment_dirs:
            pt_file = list(dir.glob("*.pt"))[0]
            idx_to_segment_loads.append(torch.load(pt_file))

        for idx, sura_idx in enumerate(ds["sura_or_aya_index"]):
            sura_idx = int(sura_idx)
            speech_intervals = idx_to_segment_loads[idx]["speech_intervals"][0]
            is_complete = idx_to_segment_loads[idx]["is_complete"][0]
            out = clean_speech_intervals(
                speech_intervals, is_complete, **segment_prams, return_seconds=True
            )
            for aya_idx, interval in enumerate(out.clean_speech_intervals):
                segment_index = get_segment_format(sura_idx, aya_idx)
                segment_index_to_interval[segment_index] = interval.numpy()

    return segment_index_to_interval


def librosa_mono_decoder(
    batch, sample_rate=16000, alias_start_sec: float = 0, first_channel_only=False
):
    """Loading aduio file and downsample it to 16000

    args:
        first_channel_only(bool): whether to load the first channel only or average both (default)
    """
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
                mono=not first_channel_only,  # Force mono conversion
            )

            # take only the first channel
            if first_channel_only:
                if len(waveform.shape) > 1:
                    waveform = waveform[0]

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

    if moshaf.id in SEGMET_MOSHAF_PARAMS:
        segment_prams = SEGMET_MOSHAF_PARAMS[moshaf.id]
    else:
        segment_prams = SEGMET_PARAMS[moshaf.recitation_speed]

    logging.info(f"Segment Params of moshaf: {moshaf.id} are: {segment_prams}")

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
                **segment_prams,
            )
            clean_outs.append(clean_out)
        except Exception as e:
            print(
                f"Error while cleaning speech intervals of moshaf: `{moshaf.id}` of file: `{batch['sura_or_aya_index'][idx]}`"
            )
            raise e

    # removing `duration_minutes` columns and rewriting `audio` column
    original_keys = set(batch.keys()) - {"audio", "duration_minutes"}
    new_batch = {
        "audio": [],
        "segment_index": [],
        "duration_seconds": [],
        "timestamp_seconds": [],
    }
    for key in original_keys:
        new_batch[key] = []
    # Devising recitations into segments
    for idx, clean_out in enumerate(clean_outs):
        new_audios = []
        segment_ids = []
        durations_seconds = []
        timestamp_list = []
        for span_idx, span in enumerate(clean_out.clean_speech_intervals):
            timestamp_list.append(span.numpy())
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
        new_batch["timestamp_seconds"] += timestamp_list
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
    annotated_segment_ids: Optional[set[str]] = None,
) -> dict:
    async def dummy_task():
        return ""

    async def async_main(waves, annotated_segment_ids):
        tasks = [
            tarteel_transcribe(
                wave,
                vllm_endpoint=vllm_endpoint,
                timeout_sec=timeout_sec,
                chunck_overlap_sec=chunk_overlap_sec,
                max_len_sec=max_len_sec,
                sample_rate=sample_rate,
            )
            if batch["segment_index"][idx] not in annotated_segment_ids
            else dummy_task()
            for idx, wave in enumerate(waves)
        ]
        outs = await asyncio.gather(*tasks)
        return outs

    waves = [batch["audio"][idx]["array"] for idx in range(len(batch["audio"]))]
    if annotated_segment_ids is None:
        annotated_segment_ids = set()
    transcritps = asyncio.run(async_main(waves, annotated_segment_ids))

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
    annotated_segment_ids: Optional[set[str]] = None,
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
            "first_channel_only": moshaf.id in FIRST_CHANNEL_ONLY_MOSHAF,
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
            "annotated_segment_ids": annotated_segment_ids,
        },
    )

    return ds
