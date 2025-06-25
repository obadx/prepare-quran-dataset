import argparse
from pathlib import Path
import yaml
import logging
import gc
from typing import Literal
import numpy as np

from pydantic import BaseModel
import submitit
from datasets import Dataset

from prepare_quran_dataset.annotate.utils import load_segment_ids
from prepare_quran_dataset.annotate.main import load_segment_intervals_from_cache
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool
from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf


# Setup logging configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),  # Print to console
        ],
    )


class MoshafTruncConfig(BaseModel):
    id: str
    turnc_ms: float


class TruncationConfig(BaseModel):
    items: list[MoshafTruncConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TruncationConfig":
        # Load YAML data from file
        path = Path(path)
        yaml_data = path.read_text(encoding="utf-8")
        raw_data = yaml.safe_load(yaml_data)

        # Handle different YAML structures
        if isinstance(raw_data, list):
            # YAML contains direct list of items
            return cls(items=raw_data)
        elif isinstance(raw_data, dict):
            # YAML has a dictionary structure
            if "items" in raw_data:
                # Standard case: top-level 'items' key
                return cls(**raw_data)
            else:
                # Handle flat dictionary (single item) by wrapping in a list
                return cls(items=[raw_data])
        else:
            raise TypeError(
                f"Invalid YAML structure. Expected list/dict, got {type(raw_data)}"
            )


def decode_segment_index(
    segment_index: str,
) -> tuple[int, int]:
    """
    Args:
        segment_index (str): {sura absolute index}.{part index} Example: "001.0003"

    Returns:
        tuple[sura integer index, part integer index]
    """
    parts = segment_index.split(".")
    return int(parts[0]), int(parts[1])


def get_turncataion_strategy(
    segment_index: str,
    idx_to_segment: list[str],
    segment_to_idx: dict[str, int],
) -> Literal["start", "middle", "end"]:
    sura_index, part_index = decode_segment_index(segment_index)

    if part_index == 0:
        return "start"

    # Last item
    if (segment_to_idx[segment_index] + 1) == len(segment_to_idx):
        return "end"

    next_segment_index = idx_to_segment[segment_to_idx[segment_index] + 1]
    next_sura_index, next_part_index = decode_segment_index(next_segment_index)

    if next_sura_index != sura_index:
        return "end"

    return "middle"


def add_timestamp_map(
    example,
    trunc_samples,
    idx_to_segment: list[str],
    segment_to_idx: dict[str, int],
    segment_index_to_interval: dict[str, np.array],
    sample_rate=16000,
):
    startegy = get_turncataion_strategy(
        example["segment_index"], idx_to_segment, segment_to_idx
    )
    interval = segment_index_to_interval[example["segment_index"]]
    logging.debug(
        f"Interval for {example['segment_index']} before processing: {interval}"
    )

    match startegy:
        case "start":
            logging.debug(f"Start Truncation strategy for: {example['segment_index']}")
            interval[1] -= trunc_samples / sample_rate

        case "middle":
            interval[0] += trunc_samples / sample_rate
            interval[1] -= trunc_samples / sample_rate

        case "end":
            logging.debug(f"End Truncation Strategy for: {example['segment_index']}")
            interval[0] += trunc_samples / sample_rate

    logging.debug(
        f"Interval for {example['segment_index']} after processing: {interval}"
    )
    example["timestamp_seconds"] = interval

    return example


def add_timestamp_column(
    moshaf: Moshaf,
    path_to_parquets: Path,
    metadata_jsonl_path: Path,
    moshaf_segment_cache_dir: Path,
    moshaf_trunc_config: MoshafTruncConfig = None,
    num_proc=16,
    sample_rate=16000,
):
    """
    Args:
        ds_path: path to parquet files
    """
    idx_to_segment = sorted(load_segment_ids(path_to_parquets))
    segment_to_idx = {seg: idx for idx, seg in enumerate(idx_to_segment)}
    segment_index_to_interval = load_segment_intervals_from_cache(
        moshaf,
        moshaf_segment_cache_dir,
        metadata_jsonl_path,
    )
    logging.info(
        f"{len(segment_index_to_interval)} intervals loaded for moshaf: {moshaf.id}"
    )

    for parquet_path in sorted(path_to_parquets.glob("*.parquet")):
        logging.info(f"Woring in shard: {parquet_path}")
        ds_shard = Dataset.from_parquet(str(parquet_path))

        # Truncate
        if moshaf_trunc_config is not None:
            trunc_samples = int(moshaf_trunc_config.turnc_ms * sample_rate / 1000)
        else:
            trunc_samples = 0
        logging.debug(f"Truncation samples: {trunc_samples}")

        ds_shard = ds_shard.map(
            add_timestamp_map,
            fn_kwargs={
                "trunc_samples": trunc_samples,
                "idx_to_segment": idx_to_segment,
                "segment_to_idx": segment_to_idx,
                "segment_index_to_interval": segment_index_to_interval,
            },
            num_proc=num_proc,
        )

        ds_shard = ds_shard.sort("segment_index")
        ds_shard.to_parquet(parquet_path)
        logging.info(f"Done! Saving shard: {parquet_path}")

        del ds_shard
        gc.collect()


def main(args):
    reciter_pool = ReciterPool(args.dataset_dir / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, args.dataset_dir)

    trunc_conifg = TruncationConfig.from_yaml("./truncation_config.yml")
    moshaf_id_to_moshaf_trunc_conifg = {m.id: m for m in trunc_conifg.items}

    # Configure Slurm
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_account="shams035",
        slurm_partition="cpu",
        slurm_time="0-18:00:00",
        slurm_ntasks_per_node=1,
        cpus_per_task=16,
    )

    for moshaf in moshaf_pool:
        executor.update_parameters(
            slurm_job_name=f"stamp-{moshaf.id}",
            slurm_additional_parameters={
                # "output": f"QVADcpu_{split}_%j.out"  # %j = Slurm job ID
            },
        )
        if moshaf.id in moshaf_id_to_moshaf_trunc_conifg:
            moshaf_trunc_config = moshaf_id_to_moshaf_trunc_conifg[moshaf.id]
        else:
            moshaf_trunc_config = None
        job = executor.submit(
            add_timestamp_column,
            moshaf=moshaf,
            path_to_parquets=args.dataset_dir / f"dataset/{moshaf.id}/train",
            metadata_jsonl_path=(
                args.original_dataset_dir / f"dataset/{moshaf.id}/metadata.jsonl"
            ),
            moshaf_segment_cache_dir=args.dataset_dir / f".segment_cache/{moshaf.id}",
            moshaf_trunc_config=moshaf_trunc_config,
        )
        print(job.job_id)
        # add_timestamp_column(
        #     moshaf=moshaf,
        #     path_to_parquets=args.dataset_dir / f"dataset/{moshaf.id}/train",
        #     metadata_jsonl_path=(
        #         args.original_dataset_dir / f"dataset/{moshaf.id}/metadata.jsonl"
        #     ),
        #     moshaf_segment_cache_dir=args.dataset_dir / f".segment_cache/{moshaf.id}",
        #     moshaf_trunc_config=moshaf_trunc_config,
        # )


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        "Building Recitations Dataset by spliting tracks using وقف and trancripe using Tarteel"
    )
    parser.add_argument(
        "--original-dataset-dir",
        type=Path,
        required=True,
        help="""The path to the quran dataset that has the following directory structure
        ../quran-dataset/
        ├── dataset
        │   └── 0.1
        │   └──---- 001.mp3
        │   └──---- 002.mp3
        ├── moshaf_pool.jsonl
        └── reciter_pool.jsonl
        """,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="""The path to the annotated quran dataset that has the following directory structure
        ../quran-dataset/
        ├── dataset
        │   └── 0.0
        │       └── train
        │           ├── shard_00016.parquet
        │           └── shard_00017.parquet
        ├── moshaf_pool.jsonl
        ├── moshaf_pool.parquet
        ├── README.md
        ├── reciter_pool.jsonl
        └── reciter_pool.parquet
        """,
    )

    args = parser.parse_args()

    main(args)
