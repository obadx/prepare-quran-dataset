import argparse
from pathlib import Path
import os

from datasets import Features, Audio, Value


from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool
from prepare_quran_dataset.construct.utils import overwrite_readme_yaml, save_jsonl
from prepare_quran_dataset.hf_dataset_config import (
    HFDatasetBuilder,
    HFDatasetConfig,
    HFDatasetSplit,
)


RECITATION_FEATURES = Features(
    {
        "audio": Audio(),
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


def write_redmme(
    dataset_path: str | Path,
    moshaf_excluded_fields: list[str],
    moshaf_pool: MoshafPool,
    recitation_features: Features,
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
    dataset_path = Path(dataset_path)
    dataset_path.mkdir(exist_ok=True)

    # metadata_items = []
    # for split in splits:
    #     metadata_items.append(
    #         {'split': split,
    #          'path': f'data/{split}/train/*.parquet'
    #          }
    #     )
    moshaf_features, _ = Moshaf.extract_huggingface_features(
        exclueded_fields=moshaf_excluded_fields
    )
    reciter_features, _ = Reciter.extract_huggingface_features()

    # # use one of them not both
    # metadata = {
    #     # 'dataset_info': {'featrues': features._to_yaml_list()},
    #     'configs': [
    #         {
    #             'config_name': 'recitations_metadata',
    #             'features': moshaf_features,
    #             'data_files':
    #                 [
    #                     {
    #                         'split': 'train',
    #                         'path': (dataset_path / 'moshaf_pool.parquet').relative_to(dataset_path),
    #                     }
    #                 ],
    #
    #         },
    #         {
    #             'config_name': 'reciters_metadata',
    #             'features': reciter_features,
    #             'data_files':
    #                 [
    #                     {
    #                         'split': 'train',
    #                         'path': (dataset_path / 'reciter_pool.parquet').relative_to(dataset_path),
    #                     }
    #                 ],
    #
    #
    #         },
    #
    #     ]
    # }
    # builder = HFDatasetBuilder(**metadata)

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

    # # Adding recitation tracks as splits
    # splits: list[HFDatasetSplit] = []
    # num_tracks = 0
    # for moshaf in moshaf_pool:
    #     splits.append(HFDatasetSplit(
    #         split=f'moshaf_{moshaf.id}',
    #         path=moshaf.path,
    #     ))
    #     num_tracks += moshaf.num_recitations
    # configs.append(HFDatasetConfig(
    #     config_name='moshaf_tracks',
    #     features=recitation_features,
    #     data_files=splits,
    # ))

    # building the dataset info
    builder = HFDatasetBuilder(
        configs=configs,
        # dataset_info={'configs': [{
        #     'name': 'moshaf_tracks', 'num_examples': num_tracks}]}
    )
    builder.to_readme_yaml(dataset_path / "README.md")


def write_tracks_metadata(
    moshaf_pool: MoshafPool,
    dataset_dir: str | Path,
) -> None:
    """Writes the tracks metadata for every track to be a huggingface dataset format"""
    dataset_dir = Path(dataset_dir)
    for moshaf in moshaf_pool:
        (dataset_dir / moshaf.path).mkdir(exist_ok=True)
        metadata = []
        for track_info in moshaf.recitation_files:
            metadata.append(
                {
                    "file_name": Path(track_info.path).name,
                    "moshaf_id": moshaf.id,
                    "moshaf_name": moshaf.name,
                    "reciter_id": moshaf.reciter_id,
                    "reciter_arabic_name": moshaf.reciter_arabic_name,
                    "reciter_english_name": moshaf.reciter_english_name,
                    "sura_or_aya_index": track_info.name.split(".")[0],
                    "index_type": moshaf.segmented_by,
                    "sample_rate": track_info.sample_rate,
                    "duration_minutes": track_info.duration_minutes,
                }
            )

        if metadata:
            metadata = sorted(metadata, key=lambda x: x["sura_or_aya_index"])
            save_jsonl(metadata, dataset_dir / moshaf.path / "metadata.jsonl")


def main(args):
    moshaf_excluded_fields = set(
        [
            "downloaded_sources",
            "recitation_files",
        ]
    )

    reciter_pool = ReciterPool(Path(args.dataset_dir) / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, args.dataset_dir)

    reciter_pool.to_parquet(Path(args.dataset_dir) / "reciter_pool.parquet")
    moshaf_pool.to_parquet(
        Path(args.dataset_dir) / "moshaf_pool.parquet",
        excluded_fields=[
            "downloaded_sources",
            "recitation_files",
        ],
    )
    if not args.not_write_readme_yaml:
        write_redmme(
            args.dataset_dir,
            moshaf_excluded_fields=moshaf_excluded_fields,
            moshaf_pool=moshaf_pool,
            recitation_features=RECITATION_FEATURES,
        )

    write_tracks_metadata(moshaf_pool, args.dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converting The Quran Dataset to huggingface format"
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
        "--not-write-readme-yaml",
        action="store_true",
        help="Not writing the metadata of the dataset insied the yaml section of the README.md",
    )

    args = parser.parse_args()

    main(args)
