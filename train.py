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
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import default_collate
import torch
from torch.nn import CrossEntropyLoss
from pydantic import BaseModel, validator


from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer
from prepare_quran_dataset.modeling.vocab import PAD_TOKEN_IDX

from multi_level_ctc_model.configuration_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTCConfig,
)
from multi_level_ctc_model.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC


# TODO:
# * Hyberparamets
# * Prepare Data
# * Add augmentation
# * Evaluation metrics
# * Push best models every best to the hub
# * disable gradient checkpinting (NO) ????
# * push to hup [DONE]
# * start/resume [DONE]
# * add seed [DONE]
# * see logging (tensor board) [DONE]


class TrainConfig(BaseModel):
    train_moshaf_ids: list[str]
    test_moshaf_ids: list[str] | None = None
    augment_prob: float = 0.4
    phonemes_loss_weight: float = 0.4
    shidda_loss_weight: float | None = None
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


def ctc_decode(batch_arr, blank_id=0, collapse_consecutive=True) -> list[NDArray]:
    decoded_list = []
    for seq in batch_arr:
        if collapse_consecutive:
            tokens = []
            prev = blank_id
            for current in seq:
                if current == blank_id:
                    prev = blank_id
                    continue
                if current == prev:
                    continue
                tokens.append(current)
                prev = current
            decoded_list.append(np.array(tokens, dtype=seq.dtype))
        else:
            tokens = seq[seq != blank_id]
            decoded_list.append(tokens)
    return decoded_list


def sequence_to_chars(labels) -> str:
    t = ""
    for label in labels:
        if label > len(string.ascii_letters):
            raise ValueError(
                f"We only support labels up to : `{len(string.ascii_letters)}` got {label}"
            )
        t += string.ascii_letters[label]
    return t


def compute_per_level(predictions: list[str], references: list[str], pad_token_idx=0):
    """
    Compute Phoneme Error Rate using Levenshtein distance.
    """
    total_distance = 0
    total_length = 0

    pred_ids_list = ctc_decode(
        predictions, collapse_consecutive=True, blank_id=pad_token_idx
    )
    ref_ids_list = ctc_decode(
        references, collapse_consecutive=False, blank_id=pad_token_idx
    )

    for pred, ref in zip(pred_ids_list, ref_ids_list):
        pred_str = sequence_to_chars(pred)
        ref_str = sequence_to_chars(ref)
        # Compute Levenshtein distance
        distance = min(Levenshtein.distance(pred_str, ref_str), len(ref_str))
        total_distance += distance
        total_length += len(ref_str)

    return total_distance / total_length if total_length > 0 else 0.0


def compute_metrics(eval_pred, pad_token_idx=0):
    """
    Compute PER metrics for multi-level predictions.

    Args:
        eval_pred: Tuple of (predictions, labels) where both are dictionaries
        multi_level_tokenizer: MultiLevelTokenizer instance for decoding

    Returns:
        Dictionary with PER metrics for each level and average
    """
    predictions_dict, labels_dict = eval_pred
    metrics = {}

    # remove -100 id
    for level in labels_dict:
        mask = labels_dict[level] < 0
        labels_dict[level][mask] = pad_token_idx

    pred_labels = {
        level: np.argmax(p, axis=-1) for level, p in predictions_dict.items()
    }

    for level in labels_dict:
        metrics[f"per_{level}"] = compute_per_level(
            pred_labels[level],
            labels_dict[level],
            pad_token_idx=pad_token_idx,
        )

    # computing average per
    total_per = 0.0
    N = 0
    for key in metrics:
        total_per += metrics[key]
        N += 1

    metrics["average_per"] = total_per / N

    # Compute PER for each leve
    return metrics


# class CustomTrainer(Trainer):
#     def compute_loss(
#         self, model, inputs, return_outputs=False, num_items_in_batch=None
#     ):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.
#
#         Subclass and override for custom behavior.
#         """
#         outputs = model(
#             input_features=inputs["input_features"],
#             attention_mask=inputs["attention_mask"],
#         )
#         logits = outputs[0]
#         batch_size, seq_len, num_labels = logits.shape
#         loss_fact = CrossEntropyLoss()
#         loss = loss_fact(logits.view(-1, num_labels), inputs["labels"].view(-1))
#
#         return (loss, outputs) if return_outputs else loss


def build_audiomentations_augs(p=0.4, seed=42, all=False):
    """taken form: https://github.com/snakers4/silero-vad/blob/master/tuning/utils.py#L37"""
    # audiomentations usesd python random for its calculations
    random.seed(seed)
    np.random.seed(seed)

    from audiomentations import (
        SomeOf,
        AirAbsorption,
        BandPassFilter,
        BandStopFilter,
        ClippingDistortion,
        HighPassFilter,
        HighShelfFilter,
        LowPassFilter,
        LowShelfFilter,
        Mp3Compression,
        PeakingFilter,
        PitchShift,
        RoomSimulator,
        SevenBandParametricEQ,
        Aliasing,
        AddGaussianNoise,
        GainTransition,
        Compose,
        TimeStretch,
    )

    transforms = [
        Aliasing(p=1),
        AddGaussianNoise(p=1),
        AirAbsorption(p=1),
        BandPassFilter(p=1),
        BandStopFilter(p=1),
        ClippingDistortion(p=1),
        HighPassFilter(p=1),
        HighShelfFilter(p=1),
        LowPassFilter(p=1),
        LowShelfFilter(p=1),
        Mp3Compression(p=1),
        PeakingFilter(p=1),
        PitchShift(p=1),
        RoomSimulator(p=1, leave_length_unchanged=True),
        SevenBandParametricEQ(p=1),
        GainTransition(p=1, min_gain_db=-17),
        TimeStretch(p=1, leave_length_unchanged=False),
    ]
    if all:
        return Compose(transforms, p=p)
    return SomeOf((1, 3), transforms=transforms, p=p)


class Augment(object):
    def __init__(
        self,
        seed=77,
        augment_prob=0.4,
    ):
        self.seed = seed
        self.augment_prob = augment_prob
        self.augment = build_audiomentations_augs(1, seed=seed)

    def apply(
        self,
        wav: NDArray[np.float32],
        sampling_rate=16000,
    ) -> NDArray[np.float32]:
        """
        Returns:
            (new_wav, is_augmented)
        """

        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)

        new_wav = self.augment(wav, sampling_rate)
        return new_wav

    def __call__(self, item):
        item["audio"]["array"] = self.apply(item["audio"]["array"])
        return item


def prepare_dataset(
    train_config: TrainConfig,
    processor,
    multi_level_tokenizer: MultiLevelTokenizer,
    sample_rate=16000,
    is_testset=False,
):
    if is_testset:
        moshaf_ids = train_config.test_moshaf_ids
    else:
        moshaf_ids = train_config.train_moshaf_ids
    # concatenate datasets
    ds = concatenate_datasets(
        [
            load_dataset(
                "obadx/mualem-recitations-annotated",
                name=f"moshaf_{m_id}",
                split="train",
                num_proc=train_config.num_workers,
            )
            for m_id in moshaf_ids
        ]
    )

    # removihg long samples
    max_samples = int(train_config.max_audio_seconds * 16000)
    ds = ds.filter(
        lambda ex: len(ex["audio"]["array"]) <= max_samples,
        num_proc=train_config.num_workers,
    )

    # # Add augmentations
    # if not is_testset:
    #     augment_func = Augment(
    #         augment_prob=train_config.augment_prob, seed=train_config.seed
    #     )
    #     ds = ds.map(augment_func, num_proc=train_config.num_workers)
    #
    # # Add input features
    # ds = ds.map(
    #     lambda ex: {
    #         "inputs": processor(
    #             ex["audio"]["array"],
    #             sampling_rate=sample_rate,
    #             return_tensors="np",
    #         ).input_features[0],
    #     },
    #     num_proc=train_config.num_workers,
    # )

    if is_testset:
        return DatasetDict({"test": ds})

    else:
        # split train / validation
        ds = ds.train_test_split(
            test_size=train_config.devset_ratio,
            generator=np.random.default_rng(train_config.seed),
        )
        return DatasetDict({"train": ds["train"], "validation": ds["test"]})


def register_model():
    AutoConfig.register("multi_level_ctc", Wav2Vec2BertForMultilevelCTCConfig)
    AutoModel.register(Wav2Vec2BertForMultilevelCTCConfig, Wav2Vec2BertForMultilevelCTC)
    AutoModelForCTC.register(
        Wav2Vec2BertForMultilevelCTCConfig, Wav2Vec2BertForMultilevelCTC
    )


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    multi_level_tokenizer: MultiLevelTokenizer
    moshaf_id_to_moshaf_attr: dict[str, MoshafAttributes]
    augment: Augment
    special_moshaf_id_to_seg_to_moshaf_attr: dict[str, dict[str, MoshafAttributes]]

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        waves = [f["audio"]["array"] for f in features]
        for idx in range(len(waves)):
            waves[idx] = self.augment.apply(waves[idx])

        batch = self.processor(
            waves,
            sampling_rate=16000,
            padding="longest",
            return_tensors="pt",
        )

        # Preparing Moshaf Attributes
        moshaf_attrs = []
        for idx in range(len(features)):
            m_id = features[idx]["moshaf_id"]
            if m_id in self.special_moshaf_id_to_seg_to_moshaf_attr:
                seg_idx = features[idx]["segment_index"]
                if seg_idx in self.special_moshaf_id_to_seg_to_moshaf_attr[m_id]:
                    moshaf_attrs.append(
                        self.special_moshaf_id_to_seg_to_moshaf_attr[m_id][seg_idx]
                    )
                else:
                    moshaf_attrs.append(self.moshaf_id_to_moshaf_attr[m_id])
            else:
                moshaf_attrs.append(self.moshaf_id_to_moshaf_attr[m_id])

        photenized_outs = [
            quran_phonetizer(
                features[idx]["uthmani"],
                moshaf_attrs[idx],
                remove_spaces=True,
            )
            for idx in range(len(features))
        ]

        labels = self.multi_level_tokenizer.tokenize(
            [p.phonemes for p in photenized_outs],
            [p.sifat for p in photenized_outs],
            to_dict=True,
            return_tensors="pt",
            padding="longest",
        )

        # replace padding with -100 to ignore loss correctly
        for level in labels["input_ids"]:
            mask = labels["attention_mask"][level] == 0
            labels["input_ids"][level][mask] = -100

        batch["labels"] = labels["input_ids"]

        return batch


def prepare_special_moshaf_ways(
    moshaf_id_to_moshaf_attr: dict[str, dict],
) -> dict[str, dict[str, MoshafAttributes]]:
    moshaf_id_to_seg_moshaf_attr = {
        "6.0": {
            "010.0158": MoshafAttributes(
                **(moshaf_id_to_moshaf_dict["6.0"] | {"tasheel_or_madd": "tasheel"})
            ),
            "027.0107": MoshafAttributes(
                **(moshaf_id_to_moshaf_dict["6.0"] | {"tasheel_or_madd": "tasheel"})
            ),
            "010.0140": MoshafAttributes(
                **(moshaf_id_to_moshaf_dict["6.0"] | {"tasheel_or_madd": "tasheel"})
            ),
            "010.0233": MoshafAttributes(
                **(moshaf_id_to_moshaf_dict["6.0"] | {"tasheel_or_madd": "tasheel"})
            ),
            "012.0023": MoshafAttributes(
                **(moshaf_id_to_moshaf_dict["6.0"] | {"noon_tamnna": "rawm"})
            ),
            "026.0076": MoshafAttributes(
                **(moshaf_id_to_moshaf_dict["6.0"] | {"raa_firq": "tarqeeq"})
            ),
        }
    }

    return moshaf_id_to_seg_moshaf_attr


if __name__ == "__main__":
    # loading wandb tokens ans HF login
    load_secrets()
    register_model()
    train_config = TrainConfig.from_yaml("./train_config.yml")
    print(train_config)
    processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    multi_level_tokenizer = MultiLevelTokenizer("./")

    # Loading moshaf data
    moshaf_dataeet = load_dataset(
        "obadx/mualem-recitations-annotated",
        name="moshaf_metadata",
        split="train",
    )
    moshaf_id_to_moshaf_dict = {ex["id"]: ex for ex in moshaf_dataeet}
    special_moshaf_id_to_seg_to_moshaf_attr = prepare_special_moshaf_ways(
        moshaf_id_to_moshaf_dict
    )
    moshaf_id_to_moshaf_attr = {
        ex["id"]: MoshafAttributes(**ex) for ex in moshaf_dataeet
    }

    processor.push_to_hub(train_config.hub_model_id)
    multi_level_tokenizer.get_tokenizer().push_to_hub(train_config.hub_model_id)

    # Initializaze wanddb
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = train_config.wandb_project_name

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "false"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # Load dataset
    # Update with your dataset path
    dataset = prepare_dataset(train_config, processor, multi_level_tokenizer)

    # Load pre-trained model
    with open("./vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}
    if train_config.shidda_loss_weight is not None:
        loss_weight = {
            "phonemes": train_config.phonemes_loss_weight,
            "shidda_or_rakhawa": train_config.shidda_loss_weight,
        }
    else:
        loss_weight = {"phonemes": train_config.phonemes_loss_weight}

    config = Wav2Vec2BertForMultilevelCTCConfig(
        level_to_vocab_size=level_to_vocab_size,
        pad_token_id=PAD_TOKEN_IDX,
        level_to_loss_weight=loss_weight,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
    )
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        "facebook/w2v-bert-2.0", config=config
    )

    # Configure training arguments
    training_args = TrainingArguments(
        seed=train_config.seed,
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=train_config.save_every,
        save_strategy="steps",
        save_steps=train_config.save_every,
        logging_strategy="steps",
        logging_steps=train_config.save_every,
        learning_rate=train_config.learning_rate,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.pre_device_eval_batch_size,
        num_train_epochs=train_config.num_epochs,
        dataloader_num_workers=train_config.num_workers,
        weight_decay=train_config.weight_decay,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model=train_config.metric_for_bet_model,
        greater_is_better=train_config.greater_is_better,
        # push_to_hub=True,  # this pushed every checkpoint to the hup we want to push the best model only
        hub_model_id=train_config.hub_model_id,  # Update with your model name
        bf16=True,
        warmup_ratio=train_config.warmup_ratio,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        report_to=["tensorboard", "wandb"],
        gradient_checkpointing=train_config.gradient_checkpoiniting,  # Optional for memory savings
        save_total_limit=3,
        hub_strategy="all_checkpoints",  # pushes all checkpoints to the Hub with one checkpoint per subfolder in your model repository
        remove_unused_columns=False,
    )
    print(training_args)

    # Initialize label processor
    data_collector = DataCollatorCTCWithPadding(
        processor=processor,
        multi_level_tokenizer=multi_level_tokenizer,
        moshaf_id_to_moshaf_attr=moshaf_id_to_moshaf_attr,
        augment=Augment(augment_prob=train_config.augment_prob, seed=train_config.seed),
        special_moshaf_id_to_seg_to_moshaf_attr=special_moshaf_id_to_seg_to_moshaf_attr,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collector,
    )

    # Start training
    if list(Path("./results").glob("checkpoint-*")):
        print("Resuming !")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Final evaluation on test set
    if train_config.test_moshaf_ids is not None:
        testset = prepare_dataset(
            train_config, processor, multi_level_tokenizer, is_testset=True
        )
        test_results = trainer.evaluate(testset["test"], metric_key_prefix="test_")
        with open("./results/test_results.json", "w") as f:
            json.dump(test_results, f, indent=4)
        print("Test Results:", test_results)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

    # Push model and tokenizer to Hub
    trainer.push_to_hub()
