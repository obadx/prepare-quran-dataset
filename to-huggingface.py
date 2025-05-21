import argparse
from pathlib import Path
import os

from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool
from prepare_quran_dataset.construct.utils import overwrite_readme_yaml


def write_redmme(dataset_path: str | Path, moshaf_excluded_fields: list[str],):
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
        exclueded_fields=moshaf_excluded_fields)
    reciter_features, _ = Reciter.extract_huggingface_features()
    metadata = {
        # 'dataset_info': {'featrues': features._to_yaml_list()},
        'configs': [
            {
                'config_name': 'recitations_metadata',
                'features': moshaf_features._to_yaml_list(),
                'data_files':
                    [
                        {
                            'split': 'train',
                            'path': str((dataset_path / 'moshaf_pool.parquet').relative_to(dataset_path)),
                        }
                ],

            },
            {
                'config_name': 'reciters_metadata',
                'features': reciter_features._to_yaml_list(),
                'data_files':
                    [
                        {
                            'split': 'train',
                            'path': str((dataset_path / 'reciter_pool.parquet').relative_to(dataset_path)),
                        }
                ],


            },

        ]
    }
    overwrite_readme_yaml(Path(dataset_path) / 'README.md', metadata)


def main(args):
    moshaf_excluded_fields = set([
        'downloaded_sources',
        'recitation_files',
    ])

    reciter_pool = ReciterPool(Path(args.dataset_dir) / 'reciter_pool.jsonl')
    moshab_pool = MoshafPool(reciter_pool, args.dataset_dir)

    reciter_pool.to_parquet(Path(args.dataset_dir) / 'reciter_pool.parquet')
    moshab_pool.to_parquet(
        Path(args.dataset_dir) / 'moshaf_pool.parquet',
        excluded_fields=['downloaded_sources', 'recitation_files',]
    )
    write_redmme(args.dataset_dir, moshaf_excluded_fields)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Converting The Quran Dataset to huggingface format')
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help="""The path to the quran dataset that has the following directory structure
        ../quran-dataset/
        ├── dataset
        │   └── 0.1
        ├── moshaf_pool.jsonl
        └── reciter_pool.jsonl
        """
    )

    args = parser.parse_args()

    main(args)
