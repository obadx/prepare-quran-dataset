"""refernce notebook: https://colab.research.google.com/drive/1_jsPUMe4odJiRpuyeE9kjnNu--Lg-MGB?usp=sharing"""

from pathlib import Path
import json
import os
import yaml
import random
from typing import List, Union, Dict
from dataclasses import dataclass
import string


from quran_transcript import quran_phonetizer, MoshafAttributes
import Levenshtein
from dotenv import load_dotenv
import wandb
from transformers import (
    TrainingArguments,
    Trainer,
    AutoFeatureExtractor,
    AutoConfig,
    AutoModel,
    AutoModelForCTC,
    Wav2Vec2BertProcessor,
)
from huggingface_hub import login as hf_login
from pydantic import BaseModel, validator


from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer
from prepare_quran_dataset.modeling.vocab import PAD_TOKEN_IDX

from multi_level_ctc_model.configuration_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTCConfig,
)
from multi_level_ctc_model.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC


class TrainConfig(BaseModel):
    train_moshaf_ids: list[str]
    test_moshaf_ids: list[str] | None = None
    augment_prob: float = 0.4
    loss_weights: dict[str, float] = {"phonemes": 0.4}
    max_audio_seconds: float = 35.0
    num_epochs: int = 1
    devset_ratio: float = 0.1
    save_every: float = 0.2
    learning_rate: float = 5e-5
    seed: int = 42
    per_device_train_batch_size: int = 128
    pre_device_eval_batch_size: int = 128
    num_workers: int | None = None
    weight_decay: float = 0.01
    metric_for_bet_model: str = "f1"
    greater_is_better: bool = True
    hub_model_id: str = "obadx/Muaalem-model-dev"
    wandb_project_name: str = "Muaalem-model-dev"
    warmup_ratio: float = 0.2
    gradient_checkpoiniting: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            TrainConfig instance with loaded configuration

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If there's an error parsing the YAML file
            ValidationError: If the configuration doesn't match the model
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path, "r", encoding="utf-8") as file:
                config_dict = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")

        # Handle case where yaml.safe_load returns None for empty files
        if config_dict is None:
            config_dict = {}

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            yaml_path: Path where the YAML file should be saved
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert model to dict and save as YAML
        config_dict = self.model_dump()

        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(config_dict, file, default_flow_style=False, allow_unicode=True)

    @validator("num_workers", pre=True, always=True)
    def set_num_workers(cls, v):
        if v is None:
            return os.cpu_count() or 1
        return v


def load_secrets():
    # Load environment variables from .env
    load_dotenv()

    # Retrieve tokens
    wandb_token = os.getenv("WANDB_API_KEY")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Initialize WandB (automatic if env var is set)
    if wandb_token:
        wandb.login(key=wandb_token)
    else:
        print("WandB token not found!")

    # Log into HuggingFace Hub
    if hf_token:
        hf_login(token=hf_token)
    else:
        print("HuggingFace token not found!")


def register_model():
    AutoConfig.register("multi_level_ctc", Wav2Vec2BertForMultilevelCTCConfig)
    AutoModel.register(Wav2Vec2BertForMultilevelCTCConfig, Wav2Vec2BertForMultilevelCTC)
    AutoModelForCTC.register(
        Wav2Vec2BertForMultilevelCTCConfig, Wav2Vec2BertForMultilevelCTC
    )


if __name__ == "__main__":
    ckb_path = "./results/"
    repo_id = "obadx/muaalem-model-v2_0"
    # repo_id = "obadx/muaalem-model-v2_1"
    # loading wandb tokens ans HF login
    load_secrets()
    register_model()
    train_config = TrainConfig.from_yaml("./train_config.yml")
    print(train_config)
    processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    multi_level_tokenizer = MultiLevelTokenizer("./")

    with open("./vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}
    for level in train_config.loss_weights:
        if level not in level_to_vocab_size:
            raise ValueError(
                f"The level `{level}` does not exist availabel are: `{list(level_to_vocab_size.values())}`"
            )

    processor.push_to_hub(train_config.hub_model_id)
    multi_level_tokenizer.get_tokenizer().push_to_hub(train_config.hub_model_id)

    config = Wav2Vec2BertForMultilevelCTCConfig(
        level_to_vocab_size=level_to_vocab_size,
        pad_token_id=PAD_TOKEN_IDX,
        level_to_loss_weight=train_config.loss_weights,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
    )
    # model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
    #     "facebook/w2v-bert-2.0", config=config
    # )
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(ckb_path)
    model.push_to_hub(repo_id)
