import argparse
from pathlib import Path
import yaml
import logging
import gc
from typing import Literal

from pydantic import BaseModel
import submitit
from datasets import Dataset

from prepare_quran_dataset.annotate.utils import load_segment_ids


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


def truncate_example(
    example,
    trunc_samples,
    idx_to_segment: list[str],
    segment_to_idx: dict[str, int],
    sample_rate=16000,
):
    startegy = get_turncataion_strategy(
        example["segment_index"], idx_to_segment, segment_to_idx
    )

    match startegy:
        case "start":
            logging.debug(f"Start Truncation strategy for: {example['segment_index']}")
            example["audio"]["array"] = example["audio"]["array"][:-trunc_samples]

        case "middle":
            example["audio"]["array"] = example["audio"]["array"][
                trunc_samples:-trunc_samples
            ]

        case "end":
            logging.debug(f"End Truncation Strategy for: {example['segment_index']}")
            example["audio"]["array"] = example["audio"]["array"][trunc_samples:]

    example["duration_seconds"] = len(example["audio"]["array"]) / sample_rate

    return example


def truncate_moshaf(
    moshaf_trunc_config: MoshafTruncConfig,
    ds_path: Path,
    num_proc=16,
    sample_rate=16000,
):
    idx_to_segment = sorted(load_segment_ids(ds_path))
    segment_to_idx = {seg: idx for idx, seg in enumerate(idx_to_segment)}

    for parquet_path in ds_path.glob("*.parquet"):
        logging.info(f"Woring in shard: {parquet_path}")
        ds_shard = Dataset.from_parquet(str(parquet_path))

        # Truncate
        trunc_samples = int(moshaf_trunc_config.turnc_ms * sample_rate / 1000)
        ds_shard.map(
            truncate_example,
            fn_kwargs={
                "trunc_samples": trunc_samples,
                "idx_to_segment": idx_to_segment,
                "segment_to_idx": segment_to_idx,
            },
            num_proc=num_proc,
        )

        # caching
        cache = [item for item in ds_shard]
        del ds_shard
        gc.collect()
        to_save_ds = Dataset.from_list(cache)
        logging.info(f"Done! Saving shard: {parquet_path}")
        to_save_ds.to_parquet(parquet_path)
        del to_save_ds
        gc.collect()


def main(args):
    out_path = Path(args.dataset_dir) / "dataset"
    out_path.mkdir(exist_ok=True, parents=True)

    trunc_conifg = TruncationConfig.from_yaml("./truncation_config.yml")

    # Configure Slurm
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_account="shams035",
        slurm_partition="cpu",
        slurm_time="0-18:00:00",
        slurm_ntasks_per_node=1,
        cpus_per_task=16,
    )

    for moshaf_trunc_config in trunc_conifg.items:
        executor.update_parameters(
            slurm_job_name=moshaf_trunc_config.id,
            slurm_additional_parameters={
                # "output": f"QVADcpu_{split}_%j.out"  # %j = Slurm job ID
            },
        )
        job = executor.submit(
            truncate_moshaf,
            moshaf_trunc_config,
            out_path / moshaf_trunc_config.id / "train",
        )
        print(job.job_id)
        # truncate_moshaf(
        #     moshaf_trunc_config, out_path / moshaf_trunc_config.id / "train"
        # )


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        "Building Recitations Dataset by spliting tracks using وقف and trancripe using Tarteel"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="""The path to the quran dataset that has the following directory structure
        ../quran-dataset/
        ├── dataset
        │   └── 0.1
        ├── moshaf_pool.jsonl
        └── reciter_pool.jsonl
        """,
    )

    args = parser.parse_args()

    main(args)
