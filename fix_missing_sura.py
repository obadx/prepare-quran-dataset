import argparse
from pathlib import Path
import logging

import submitit
from datasets import Dataset, load_dataset, concatenate_datasets
import numpy as np

from prepare_quran_dataset.annotate.utils import save_to_disk_split
from prepare_quran_dataset.annotate.main import OUT_FEATURES


# Setup logging configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),  # Print to console
        ],
    )


def add_missing_suar(moshaf_id, ds_path: Path, added_shard_path: Path):
    """Adding missing suar into the dataset"""

    added_shard = Dataset.from_parquet(str(added_shard_path))
    added_shard = added_shard.cast(OUT_FEATURES)

    added_suar_set = set(added_shard["sura_or_aya_index"])
    print(f"Len of added shard: {len(added_shard)}")

    ds = load_dataset(str(ds_path), name=f"moshaf_{moshaf_id}", split="train")
    print(f"Len of ds: {len(ds)}")

    # clean dataset from addes suar
    ds = ds.filter(
        lambda ex: ex["sura_or_aya_index"] not in added_suar_set, num_proc=16
    )
    print(f"Len of ds after removing to be addes suar: {len(ds)}")

    ds = concatenate_datasets([ds, added_shard])
    ds = ds.sort("segment_index")
    print(f"Len of ds after adding new suar: {len(ds)}")

    save_to_disk_split(ds, moshaf_id, ds_path / "dataset", samples_per_shard=512)


def main(args):
    out_path = Path(args.dataset_dir) / "dataset"
    out_path.mkdir(exist_ok=True, parents=True)

    # Configure Slurm
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_account="shams035",
        slurm_partition="cpu",
        slurm_time="0-4:00:00",
        slurm_ntasks_per_node=1,
        cpus_per_task=16,
    )

    # for moshaf_id in {"0.2", "3.0", "6.0", "30.0"}:
    for moshaf_id in {"2.0"}:
        executor.update_parameters(
            slurm_job_name=f"ADD{moshaf_id}",
            slurm_additional_parameters={
                # "output": f"QVADcpu_{split}_%j.out"  # %j = Slurm job ID
            },
        )
        job = executor.submit(
            add_missing_suar,
            moshaf_id,
            ds_path=args.dataset_dir,
            added_shard_path=args.added_dataset_dir
            / f"dataset/{moshaf_id}/train/shard_00000.parquet",
        )
        print(job.job_id)
        # add_missing_suar(
        #     moshaf_id,
        #     ds_path=args.dataset_dir,
        #     added_shard_path=args.added_dataset_dir
        #     / f"dataset/{moshaf_id}/train/shard_00000.parquet",
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
    parser.add_argument(
        "--added-dataset-dir",
        type=Path,
        required=True,
        help="""The path to the added sura that has the following directory structure
        ../quran-dataset/
        ├── dataset
        │   └── 0.1
        ├── moshaf_pool.jsonl
        └── reciter_pool.jsonl
        """,
    )

    args = parser.parse_args()

    main(args)
