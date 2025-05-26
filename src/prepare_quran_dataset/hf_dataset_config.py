from pathlib import Path

from pydantic import BaseModel, field_serializer, ConfigDict
from datasets import Features, DatasetInfo, DatasetBuilder

from .construct.utils import overwrite_readme_yaml


class HFDatasetSplit(BaseModel):
    split: str
    path: str | Path

    # allow not serialized fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Redact password when using model.model_dump()

    @field_serializer('path')
    def to_str(self, p: str | Path) -> str:
        return str(p)


class HFDatasetConfig(BaseModel):
    config_name: str = 'default'
    features: Features
    data_files: list[HFDatasetSplit]

    # allow not serialized fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Redact password when using model.model_dump()
    @field_serializer('features')
    def to_yml_str(self, feats: Features) -> list:
        return feats._to_yaml_list()


class HFDatasetSplitorConfigInfo(BaseModel):
    name: str
    num_examples: int = None


class HFDatasetInfo(BaseModel):
    splits: list[HFDatasetSplitorConfigInfo] = None
    configs: list[HFDatasetSplitorConfigInfo] = None
    features: Features = None

    # allow not serialized fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Redact password when using model.model_dump()
    @field_serializer('features')
    def to_yml_str(self, feats: Features) -> list:
        return feats._to_yaml_list()


class HFDatasetBuilder(BaseModel):
    """Huggingface Dataset Builder for custom dataset builds

    Instead of manully write the metatdata dictionary to the yaml section of
    the README.md file -> write a rigroues difinition for the builder class

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

    """
    configs: list[HFDatasetConfig]
    # TODO: needs refactory and integeration with DatasetInfo
    dataset_info: HFDatasetInfo = None

    def to_readme_yaml(self, path: str | Path):
        """Overites the yaml section of the huggifnace dataset README.md
        """
        overwrite_readme_yaml(path, self.model_dump(exclude_none=True))
