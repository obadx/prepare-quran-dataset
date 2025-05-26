from pathlib import Path

from prepare_quran_dataset.hf_dataset_config import HFDatasetConfig, HFDatasetBuilder, HFDatasetSplit
from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter

if __name__ == '__main__':
    moshaf_excluded_fields = set([
        'downloaded_sources',
        'recitation_files',
    ])

    features, metadata = Moshaf.extract_huggingface_features(
        exclueded_fields=moshaf_excluded_fields)

    builder = HFDatasetBuilder(
        configs=[
            HFDatasetConfig(
                config_name='moshaf_metadata',
                features=features,
                data_files=[
                    HFDatasetSplit(
                        split='train',
                        path=Path('./hamo/nono'),

                    )
                ],

            ),
            HFDatasetConfig(
                config_name='moshaf_metadata',
                features=features,
                data_files=[
                    HFDatasetSplit(
                        split='train',
                        path=Path('./hamo/nono'),

                    )
                ],

            ),

        ],
    )

    print(builder.model_dump(exclude_none=True))
