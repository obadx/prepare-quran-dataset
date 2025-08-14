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
    processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    multi_level_tokenizer = MultiLevelTokenizer("./")

    processor.push_to_hub(repo_id)
    multi_level_tokenizer.get_tokenizer().push_to_hub(repo_id)

    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(ckb_path)
    model.push_to_hub(repo_id)
