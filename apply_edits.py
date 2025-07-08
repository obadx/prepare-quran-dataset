import argparse
from pathlib import Path
import logging
import json

import submitit

from prepare_quran_dataset.annotate.edit import (
    EditConfig,
    MoshafEditConfig,
    load_moshaf_dataset_info,
    plan_moshaf_edits,
    apply_ds_shard_edits,
)


# Setup logging configuration
def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),  # Print to console
        ],
    )


def apply_moshaf_edits(
    moshaf_edit_config: MoshafEditConfig,
    ds_path: Path,
    moshaf_media_files_path: Path,
    new_audiofile_path: Path,
    num_proc=4,
):
    ds_path = Path(ds_path)
    new_audiofile_path = Path(new_audiofile_path)
    ds_info = load_moshaf_dataset_info(ds_path)

    to_apply_shrads = plan_moshaf_edits(ds_info, moshaf_edit_config)

    for shards_ops in to_apply_shrads:
        logging.info(
            f"Planned operatations for shard: {shards_ops.path}\n{json.dumps(shards_ops.model_dump(), ensure_ascii=False, indent=4)}"
        )

    for shards_ops in to_apply_shrads:
        apply_ds_shard_edits(
            shards_ops,
            new_audiofile_path=new_audiofile_path,
            moshaf_media_files_path=moshaf_media_files_path,
            num_proc=num_proc,
        )


def main(args):
    out_path = Path(args.dataset_dir) / "dataset"
    out_path.mkdir(exist_ok=True, parents=True)

    edit_cofig = EditConfig.from_yaml("./edit_config.yml")

    # Configure Slurm
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_account="shams035",
        slurm_partition="cpu",
        slurm_time="0-18:00:00",
        slurm_ntasks_per_node=1,
        cpus_per_task=16,
    )

    for moshaf_edit_config in edit_cofig.configs:
        executor.update_parameters(
            slurm_job_name=f"Edit_{moshaf_edit_config.id}",
            slurm_additional_parameters={
                # "output": f"QVADcpu_{split}_%j.out"  # %j = Slurm job ID
            },
        )
        # job = executor.submit(
        #     apply_moshaf_edits,
        #     moshaf_edit_config,
        #     ds_path=out_path / moshaf_edit_config.id / "train",
        #     moshaf_media_files_path=(
        #         args.original_dataset_dir / f"dataset/{moshaf_edit_config.id}"
        #     ),
        #     new_audiofile_path=args.new_audiofile_base_path,
        #     num_proc=16,
        # )
        # print(job.job_id)
        apply_moshaf_edits(
            moshaf_edit_config,
            ds_path=out_path / moshaf_edit_config.id / "train",
            moshaf_media_files_path=(
                args.original_dataset_dir / f"dataset/{moshaf_edit_config.id}"
            ),
            new_audiofile_path=args.new_audiofile_base_path,
            num_proc=4,
        )
        break


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        "Building Recitations Dataset by spliting tracks using وقف and trancripe using Tarteel"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="""The path to the annoteted quran dataset that has the following directory structure
            .
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
    parser.add_argument(
        "--original-dataset-dir",
        type=Path,
        required=True,
        help="""The path to the original quran dataset which has original media files (before annotation)
            .
            ├── dataset
            │   ├── 0.0
            │   │   ├── 050.mp3
            │   │   ├── 114.mp3
            │   │   └── metadata.jsonl
            ├── moshaf_pool.jsonl
            ├── moshaf_pool.parquet
            ├── README.md
            ├── reciter_pool.jsonl
            └── reciter_pool.parquet
        """,
    )

    parser.add_argument(
        "--new-audiofile-base-path",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    main(args)
