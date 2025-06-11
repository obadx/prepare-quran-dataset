import argparse
from pathlib import Path
import shutil

from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
import torch


from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool
from prepare_quran_dataset.construct.utils import overwrite_readme_yaml
from prepare_quran_dataset.hf_dataset_config import (
    HFDatasetBuilder,
    HFDatasetConfig,
    HFDatasetSplit,
)
from prepare_quran_dataset.annotate.main import OUT_FEATURES, process_moshaf_tracks
from prepare_quran_dataset.annotate.utils import save_to_disk_split


def write_redmme_splits(
    dataset_path: str | Path,
    moshaf_excluded_fields: list[str],
    moshaf_pool: MoshafPool,
    recitation_features=OUT_FEATURES,
):
    """Write metadata to yaml section of readme:
    EX:

    ---
    configs:
    - config_name: default
    data_files:
    - path: data/recitation_6/train/*.parquet
        split: recitation_6
    ---
    """
    pass
    dataset_path = Path(dataset_path)
    dataset_path.mkdir(exist_ok=True)

    moshaf_features, _ = Moshaf.extract_huggingface_features(
        exclueded_fields=moshaf_excluded_fields
    )
    reciter_features, _ = Reciter.extract_huggingface_features()

    configs: list[HFDatasetConfig] = []
    configs.append(
        HFDatasetConfig(
            config_name="moshaf_metadata",
            features=moshaf_features,
            data_files=[
                HFDatasetSplit(
                    split="train",
                    path=(dataset_path / "moshaf_pool.parquet").relative_to(
                        dataset_path
                    ),
                )
            ],
        ),
    )
    configs.append(
        HFDatasetConfig(
            config_name="reciters_metadata",
            features=reciter_features,
            data_files=[
                HFDatasetSplit(
                    split="train",
                    path=(dataset_path / "reciter_pool.parquet").relative_to(
                        dataset_path
                    ),
                )
            ],
        ),
    )

    # Adding recitation tracks as splits
    splits: list[HFDatasetSplit] = []
    for moshaf in moshaf_pool:
        splits.append(
            HFDatasetSplit(
                split=f"moshaf_{moshaf.id}",
                path=f"dataset/{moshaf.id}/train/*.parquet",
            )
        )
    configs.append(
        HFDatasetConfig(
            config_name="moshaf_tracks",
            features=recitation_features,
            data_files=splits,
        )
    )

    # building the dataset info
    builder = HFDatasetBuilder(
        configs=configs,
        # dataset_info={'configs': [{
        #     'name': 'moshaf_tracks', 'num_examples': num_tracks}]}
    )
    builder.to_readme_yaml(dataset_path / "README.md")


def write_redmme_configs(
    dataset_path: str | Path,
    moshaf_excluded_fields: list[str],
    moshaf_pool: MoshafPool,
    recitation_features=OUT_FEATURES,
):
    """Write metadata to yaml section of readme:
    EX:

    ---
    configs:
    - config_name: default
    data_files:
    - path: data/recitation_6/train/*.parquet
        split: recitation_6
    ---
    """
    pass
    dataset_path = Path(dataset_path)
    dataset_path.mkdir(exist_ok=True)

    moshaf_features, _ = Moshaf.extract_huggingface_features(
        exclueded_fields=moshaf_excluded_fields
    )
    reciter_features, _ = Reciter.extract_huggingface_features()

    configs: list[HFDatasetConfig] = []
    configs.append(
        HFDatasetConfig(
            config_name="moshaf_metadata",
            features=moshaf_features,
            data_files=[
                HFDatasetSplit(
                    split="train",
                    path=(dataset_path / "moshaf_pool.parquet").relative_to(
                        dataset_path
                    ),
                )
            ],
        ),
    )
    configs.append(
        HFDatasetConfig(
            config_name="reciters_metadata",
            features=reciter_features,
            data_files=[
                HFDatasetSplit(
                    split="train",
                    path=(dataset_path / "reciter_pool.parquet").relative_to(
                        dataset_path
                    ),
                )
            ],
        ),
    )

    # Adding recitation tracks as separeate dataset configs
    for moshaf in moshaf_pool:
        configs.append(
            HFDatasetConfig(
                config_name=f"moshaf_{moshaf.id}",
                features=recitation_features,
                data_files=[
                    HFDatasetSplit(
                        split="train",
                        path=f"dataset/{moshaf.id}/train/*.parquet",
                    ),
                ],
            )
        )

    # building the dataset info
    builder = HFDatasetBuilder(
        configs=configs,
        # dataset_info={'configs': [{
        #     'name': 'moshaf_tracks', 'num_examples': num_tracks}]}
    )
    builder.to_readme_yaml(dataset_path / "README.md")


def main(args):
    moshaf_excluded_fields = set(
        [
            "downloaded_sources",
            "recitation_files",
        ]
    )

    out_path = Path(args.out_dataset_dir) / "dataset"
    out_path.mkdir(exist_ok=True, parents=True)

    reciter_pool = ReciterPool(args.dataset_dir / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, args.dataset_dir)
    metadata_files = []
    for ext in ["jsonl", "parquet"]:
        metadata_files += list(args.dataset_dir.glob(f"*.{ext}"))
    for f in metadata_files:
        shutil.copy(f, args.out_dataset_dir)

    if not args.not_process_data:
        processor = AutoFeatureExtractor.from_pretrained(
            "obadx/recitation-segmenter-v2"
        )
        model = AutoModelForAudioFrameClassification.from_pretrained(
            "obadx/recitation-segmenter-v2",
        )
        model.to(args.device, dtype=torch.bfloat16)

        for moshaf in moshaf_pool:
            if (out_path / moshaf.id).is_dir():
                continue
            if moshaf.id not in ["4.0", "19.0"]:
                break
            ds = process_moshaf_tracks(
                moshaf,
                args.dataset_dir,
                loop_batch_size=16,
                sample_rate=16000,
                tarteel_batch_size=16,
                segment_batch_size=16,
                segment_device="cuda",
                segment_model=model,
                segment_feature_extractor=processor,
                segment_cache_dir=".segment_cache",
                tarteel_timeout_sec=300,
                tarteel_chunk_overlap_sec=10,
                tarteel_max_len_sec=30,
                tarteel_vllm_endpont="http://localhost:8000/v1",
            )

            # saves every path under outpath / "{moshaf_id}/train/shard.parquest
            save_to_disk_split(
                ds,
                moshaf.id,
                out_path=out_path,
                samples_per_shard=32,
            )

    write_redmme_configs(
        args.out_dataset_dir,
        moshaf_excluded_fields=moshaf_excluded_fields,
        moshaf_pool=moshaf_pool,
        recitation_features=OUT_FEATURES,
    )


if __name__ == "__main__":
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
        "--out-dataset-dir",
        type=Path,
        required=True,
        help="""The output path to the dataset dir""",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--not-process-data",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
