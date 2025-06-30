from pathlib import Path
from dataclasses import dataclass
import asyncio

from librosa.core import load
import torch
from recitations_segmenter import segment_recitations, clean_speech_intervals
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
import soundfile as sf
from datasets import load_dataset, Dataset

from prepare_quran_dataset.construct.utils import get_audiofile_info
from prepare_quran_dataset.annotate.edit import EditConfig, MoshafEditConfig, Operation
from prepare_quran_dataset.annotate.tarteel import tarteel_transcribe


@dataclass
class SuraInfo:
    id: int
    path: Path
    duration_sec: float
    sample_rate: int


def get_sura_track_info(sura_id: int, base_path: Path) -> Path:
    base_path = Path(base_path)
    sura_str = f"{sura_id:03d}"
    path_list = list(base_path.glob(f"{sura_str}*"))

    assert len(path_list) == 1, f"Error parsing sura id: {sura_id}"
    track_path = path_list[0]

    info = get_audiofile_info(track_path)

    assert info is not None, f"Error reading track metedata for filepath: {track_path}"

    return SuraInfo(sura_id, track_path, info.duration_seconds, info.sample_rate)


def segment_and_save(
    model,
    processor,
    device,
    dtype,
    segment_params: dict,
    sura_info: SuraInfo,
    out_path: str,
    cache_dir: str,
    sample_rate=16000,
    chunk_sec=60.0,
    batch_size=8,
):
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    sura_str = f"{sura_info.id:03d}"
    sura_converted_tracks = list(out_path.glob(f"{sura_str}*"))

    if len(sura_converted_tracks) == 2:
        print(f"Skipping: {sura_str}")
        return

    wave = torch.tensor(
        load(
            sura_info.path,
            sr=sample_rate,
            mono=True,
            offset=sura_info.duration_sec - chunk_sec,
        )[0]
    )

    seg_outs = segment_recitations(
        [wave],
        model,
        processor,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        cache_dir=cache_dir / f"sura_{sura_str}",
    )

    clean_outs = clean_speech_intervals(
        seg_outs[0].speech_intervals, seg_outs[0].is_complete, **segment_params
    )
    for idx, span in enumerate(clean_outs.clean_speech_intervals[-2:]):
        out_track_path = out_path / f"{sura_str}_{idx}.wav"
        sf.write(out_track_path, wave[span[0] : span[1]], sample_rate)
        print(f"Saving {out_track_path}")


def get_last_sura_segment_ids(ds: Dataset) -> dict[int, str]:
    """sura int index: segment_index"""
    seg_to_idx = ds["segment_index"]
    sura_idx_to_last_seg_idx = {}
    for idx in range(len(seg_to_idx) - 1):
        if seg_to_idx[idx].split(".")[0] != seg_to_idx[idx + 1].split(".")[0]:
            sura_idx = int(seg_to_idx[idx].split(".")[0])
            sura_idx_to_last_seg_idx[sura_idx] = seg_to_idx[idx]

    # last element
    last_idx = len(seg_to_idx) - 1
    sura_idx = int(seg_to_idx[last_idx].split(".")[0])
    sura_idx_to_last_seg_idx[sura_idx] = seg_to_idx[last_idx]

    return sura_idx_to_last_seg_idx


def transcribe_and_save_operations(
    moshaf_id: str,
    sura_list: list[str],
    sura_idx_to_last_segment_index: dict[int, str],
    media_track_path: str,
    edit_config_path,
    chosen_idx=1,
    sample_rate=16000,
    vllm_endpoint="http://localhost:8000/v1",
):
    edit_config_path = Path(edit_config_path)
    media_track_path = Path(media_track_path)

    seg_to_ops = {}
    if edit_config_path.is_file():
        edit_config = EditConfig.from_yaml(edit_config_path)
        seg_to_ops = {op.segment_index: op for op in edit_config.configs[0].operations}

    for sura_idx in sura_list:
        track_path = media_track_path / f"{sura_idx:03d}_{chosen_idx}.wav"
        segment_index = sura_idx_to_last_segment_index[sura_idx]

        if segment_index not in seg_to_ops:
            print(f"Working on track path: {track_path}")
            wave = load(track_path, sr=sample_rate, mono=True)[0]
            tarteel_transcript = asyncio.run(
                tarteel_transcribe(
                    wave=wave, sample_rate=sample_rate, vllm_endpoint=vllm_endpoint
                )
            )
            new_op = Operation(
                type="update",
                segment_index=segment_index,
                new_tarteel_transcript=tarteel_transcript,
                new_audio_file=str(track_path.relative_to(media_track_path.parent)),
            )
            if not edit_config_path.is_file():
                edit_config = EditConfig(
                    configs=[MoshafEditConfig(id=moshaf_id, operations=[new_op])]
                )
            else:
                edit_config.configs[0].add_operation(new_op)
            # saving file
            edit_config.to_yaml(edit_config_path)
            seg_to_ops[segment_index] = new_op

        else:
            print(f"Track path: {track_path} is previously annotated")


def merge_edits(moshaf_id, edit_config_path, moshaf_edit_config_path):
    edit_config = EditConfig.from_yaml(edit_config_path)
    new_moshaf_edit_config = EditConfig.from_yaml(moshaf_edit_config_path).configs[0]
    for moshaf_edit_config in edit_config.configs:
        if moshaf_edit_config.id == moshaf_id:
            old_moshaf_edit_config = moshaf_edit_config
            break

    new_seg_to_ops = {op.segment_index: op for op in new_moshaf_edit_config.operations}
    to_del_ops = []
    for op in old_moshaf_edit_config.operations:
        if op.segment_index in new_seg_to_ops:
            to_del_ops.append(op)

    for op in to_del_ops:
        print(f"Deleting operations: {op}")
        old_moshaf_edit_config.delete_operation(op)

    # adding new_operations
    for op in new_moshaf_edit_config.operations:
        old_moshaf_edit_config.add_operation(op)

    edit_config.to_yaml(edit_config_path)


if __name__ == "__main__":
    sura_list = [
        6,
        8,
        12,
        13,
        14,
        15,
        17,
        23,
        25,
        30,
        33,
        34,
        35,
        36,
        37,
        39,
        40,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        55,
        56,
        59,
        77,
    ]
    download_path = "/home/abdullah/Downloads/4_0_missing/"
    ds_path = "/cluster/users/shams035u1/data/mualem-recitations-annotated"
    moshaf_id = "4.0"
    moshaf_edit_config_path = f"./edit_config_{moshaf_id}.yml"
    edit_config_path = "./edit_config.yml"
    fixes_path = Path(ds_path) / f"moshaf-fixes/{moshaf_id}"
    cache_dir = f".fix_{moshaf_id}_cache"
    batch_size = 8
    sample_rate = 16000
    chunk_sec = 80
    vllm_endpoint = "http://localhost:8000/v1"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16
    processor = AutoFeatureExtractor.from_pretrained("obadx/recitation-segmenter-v2")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "obadx/recitation-segmenter-v2",
    )
    segment_params = {
        "min_silence_duration_ms": 300,
        "min_speech_duration_ms": 600,
        "pad_duration_ms": 700,
    }

    model.to(device, dtype=dtype)

    fixes_path = Path(fixes_path)
    for sura_idx in sura_list:
        sura_str = f"{sura_idx:03d}"
        sura_converted_tracks = list(fixes_path.glob(f"{sura_str}*"))
        if len(sura_converted_tracks) == 2:
            print(f"Skipping Loading: {sura_str}")
            continue

        sura_info = get_sura_track_info(sura_idx, download_path)
        segment_and_save(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            sura_info=sura_info,
            out_path=fixes_path,
            cache_dir=cache_dir,
            chunk_sec=chunk_sec,
            segment_params=segment_params,
        )

    ds = load_dataset(ds_path, name=f"moshaf_{moshaf_id}", split="train")
    sura_idx_to_last_seg_idx = get_last_sura_segment_ids(ds)
    print(sura_idx_to_last_seg_idx)
    transcribe_and_save_operations(
        moshaf_id=moshaf_id,
        sura_list=sura_list,
        sura_idx_to_last_segment_index=sura_idx_to_last_seg_idx,
        edit_config_path=moshaf_edit_config_path,
        media_track_path=fixes_path,
        chosen_idx=1,
        sample_rate=16000,
        vllm_endpoint=vllm_endpoint,
    )

    merge_edits(
        moshaf_id=moshaf_id,
        edit_config_path=edit_config_path,
        moshaf_edit_config_path=moshaf_edit_config_path,
    )
