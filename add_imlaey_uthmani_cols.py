import argparse
from pathlib import Path
import logging
import json

import submitit
from datasets import Dataset, load_dataset, concatenate_datasets
import numpy as np

from prepare_quran_dataset.annotate.utils import save_to_disk_split
from prepare_quran_dataset.annotate.main import OUT_FEATURES
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool


# Setup logging configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),  # Print to console
        ],
    )


def load_tasmeea_info(tasmeea_dir: Path) -> dict[str, dict]:
    with open(tasmeea_dir / "tasmeea.json", encoding="utf-8") as f:
        sura_to_tasmeea = json.load(f)

    seg_idx_to_tasmeea = {}
    for sura_tasmeea in sura_to_tasmeea.values():
        for tasmeea_info in sura_tasmeea:
            seg_idx_to_tasmeea[tasmeea_info["segment_index"]] = tasmeea_info
    return seg_idx_to_tasmeea


def add_quran_columns(moshaf_id: str, dataset_dir: Path):
    """Adding Quran columns including imlaey, uthmani and spans"""

    seg_idx_to_tasmeea_info = load_tasmeea_info(dataset_dir / f"tasmeea/{moshaf_id}")

    ds = load_dataset(str(dataset_dir), name=f"moshaf_{moshaf_id}", split="train")

    ds = ds.map(
        lambda ex: {
            "imlaey": seg_idx_to_tasmeea_info["imlaey"],
            "uthmani": seg_idx_to_tasmeea_info["uthmani"],
            "has_quran": seg_idx_to_tasmeea_info["has_quran"],
            "has_istiaatha": seg_idx_to_tasmeea_info["has_istiaatha"],
            "has_bismillah": seg_idx_to_tasmeea_info["has_bismillah"],
            "has_sadaka": seg_idx_to_tasmeea_info["has_sadaka"],
            "start_span": seg_idx_to_tasmeea_info["start_span"],
            "end_span": seg_idx_to_tasmeea_info["end_span"],
            "match_ratio": seg_idx_to_tasmeea_info["ratio"],
        },
        # num_proc=16,
    )
    ds = ds.cast(OUT_FEATURES)

    save_to_disk_split(ds, moshaf_id, dataset_dir / "dataset", samples_per_shard=512)


def main(args):
    reciter_pool = ReciterPool(args.dataset_dir / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, args.dataset_dir)

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
    for moshaf in moshaf_pool:
        if args.slurm:
            executor.update_parameters(
                slurm_job_name=f"UTH{moshaf_id}",
                slurm_additional_parameters={
                    # "output": f"QVADcpu_{split}_%j.out"  # %j = Slurm job ID
                },
            )
            job = executor.submit(
                add_quran_columns,
                moshaf_id=moshaf.id,
                dataset_dir=args.dataset_dir,
            )
            print(job.job_id)
        else:
            add_quran_columns(
                moshaf_id=moshaf.id,
                dataset_dir=args.dataset_dir,
            )


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
    parser.add_argument("--slurm", action="store_true")

    args = parser.parse_args()

    main(args)
