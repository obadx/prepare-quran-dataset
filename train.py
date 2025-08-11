"""refernce notebook: https://colab.research.google.com/drive/1_jsPUMe4odJiRpuyeE9kjnNu--Lg-MGB?usp=sharing"""

from pathlib import Path
import json
import os
import yaml
import random
from typing import List, Union, Dict
from dataclasses import dataclass


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


def compute_phoneme_error_rate_sequences_levenshtein(predictions, labels):
    """
    Compute phoneme error rate considering sequences properly using Levenshtein distance.

    Args:
        predictions: 2D array of shape (batch_size, sequence_length)
        labels: 2D array of shape (batch_size, sequence_length)

    Returns:
        Phoneme error rate as float
    """
    total_levenshtein_distance = 0
    total_phonemes = 0

    # Loop over each sequence in the batch
    for pred_seq, label_seq in zip(predictions, labels):
        # Remove padding (-100 values)
        valid_mask = label_seq != -100
        pred_seq_valid = pred_seq[valid_mask]
        label_seq_valid = label_seq[valid_mask]

        # Convert to strings for Levenshtein distance calculation
        pred_str = "".join(map(str, pred_seq_valid))
        label_str = "".join(map(str, label_seq_valid))

        # Calculate Levenshtein distance
        levenshtein_dist = Levenshtein.distance(pred_str, label_str)
        total_levenshtein_distance += levenshtein_dist
        total_phonemes += len(label_str)

    if total_phonemes > 0:
        return total_levenshtein_distance / total_phonemes
    else:
        return 0.0


def compute_metrics(eval_pred):
    """
    Compute metrics for multi-level predictions.

    Args:
        eval_pred: Tuple of (predictions, labels) where both are dictionaries
                  with level names as keys and tensors as values

    Returns:
        Dictionary with computed metrics for each level and averages
    """
    predictions_dict, labels_dict = eval_pred

    # Convert logits to predictions for each level
    preds_dict = {}
    for level_name, logits in predictions_dict.items():
        preds_dict[level_name] = np.argmax(logits, axis=-1)

    metrics = {}

    # Metrics to compute for each level
    metric_functions = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1": lambda y_true, y_pred: f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

    # Compute metrics for each level
    for level_name in preds_dict.keys():
        # Get the raw arrays (batch_size, sequence_length)
        predictions_batch = preds_dict[level_name]  # Shape: (batch_size, seq_len)
        labels_batch = labels_dict[level_name]  # Shape: (batch_size, seq_len)

        # Flatten and filter out ignored indices (-100) for element-wise metrics
        predictions_flat = predictions_batch.flatten()
        labels_flat = labels_batch.flatten()
        mask = labels_flat != -100
        preds_flat = predictions_flat[mask]
        lbs_flat = labels_flat[mask]

        # Compute standard metrics on flattened data
        for metric_name, metric_func in metric_functions.items():
            try:
                metrics[f"{level_name}_{metric_name}"] = metric_func(
                    lbs_flat, preds_flat
                )
            except Exception:
                metrics[f"{level_name}_{metric_name}"] = 0.0

        # Compute phoneme error rate at sequence level using Levenshtein distance
        try:
            per = compute_phoneme_error_rate_sequences_levenshtein(
                predictions_batch, labels_batch
            )
            metrics[f"{level_name}_per"] = per
        except Exception:
            metrics[f"{level_name}_per"] = 0.0

    # Calculate averages across all levels
    metric_types = ["accuracy", "precision", "recall", "f1", "per"]
    for metric_type in metric_types:
        level_values = [
            metrics[f"{level_name}_{metric_type}"] for level_name in preds_dict.keys()
        ]
        if level_values:
            metrics[f"average_{metric_type}"] = np.mean(level_values)
        else:
            metrics[f"average_{metric_type}"] = 0.0

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
        return ds

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

        moshaf = self.moshaf_id_to_moshaf_attr[features[0]["moshaf_id"]]
        # geting ids of the labels
        photenized_outs = [
            quran_phonetizer(
                features[idx]["uthmani"],
                moshaf,
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


if __name__ == "__main__":
    # loading wandb tokens ans HF login
    load_secrets()
    register_model()
    train_config = TrainConfig.from_yaml("./train_config.yml")
    print(train_config)
    processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    multi_level_tokenizer = MultiLevelTokenizer("./")

    # loading moshaf data
    moshaf_dataeet = load_dataset(
        "obadx/mualem-recitations-annotated",
        name="moshaf_metadata",
        split="train",
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

    # For testing only
    # dataset['train'] = dataset['train'].take(400)
    # dataset['validation'] = dataset['validation'].take(100)
    # dataset['test'] = dataset['test'].take(100)
    #
    # # TODO: for testing only
    # new_ds = {'train': [], 'validation': [], 'test': []}
    # for split in dataset:
    #     for item in dataset[split]:
    #         new_ds[split].append(item)
    #     new_ds[split] = Dataset.from_list(new_ds[split])
    # dataset = DatasetDict(new_ds)

    # Load pre-trained model
    with open("./vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}
    config = Wav2Vec2BertForMultilevelCTCConfig(
        level_to_vocab_size=level_to_vocab_size,
        pad_token_id=PAD_TOKEN_IDX,
        level_to_loss_weight={"phonemes": train_config.phonemes_loss_weight},
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
        test_results = trainer.evaluate(dataset["test"], metric_key_prefix="test_")
        with open("./results/test_results.json", "w") as f:
            json.dump(test_results, f, indent=4)
        print("Test Results:", test_results)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

    # Push model and tokenizer to Hub
    trainer.push_to_hub()
