"""refernce notebook: https://colab.research.google.com/drive/1_jsPUMe4odJiRpuyeE9kjnNu--Lg-MGB?usp=sharing"""

import argparse
import io
import json
import os
import random
import re
import string
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import Levenshtein
import librosa
import numpy as np
import torch
import yaml
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub import login as hf_login
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator
from qdat_bench.audio_utils import decode_audio
from qdat_bench.eval_results import align_preditc_ds, eval_qdat_bench
from quran_transcript import MoshafAttributes, quran_phonetizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertProcessor,
)

import wandb
from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTC,
    FastConformerCacheAwareMultilevelCTCConfig,
    FastConformerMelProcessor,
)
from prepare_quran_dataset.modeling_streaming_rnn.modeling_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTC,
    Wav2Vec2BertForRNNStreamingMultilevelCTCConfig,
)
from prepare_quran_dataset.modeling_streaming_rnn.multi_level_tokenizer import (
    MultiLevelTokenizer,
)
from prepare_quran_dataset.modeling_streaming_rnn.vocab import PAD_TOKEN_IDX

ArchitectureName = Literal["w2v2bert-streaming-rnn", "fastconformer-cache-aware"]


def load_vocab(
    vocab_path: str = "./vocab_streaming/vocab.json",
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Load ``vocab_streaming/vocab.json`` and derive per-level vocab sizes."""
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {level: len(tokens) for level, tokens in vocab.items()}
    return vocab, level_to_vocab_size


class TrainConfig(BaseModel):
    train_moshaf_ids: list[str]
    test_moshaf_ids: list[str] | None = None
    augment_prob: float = 0.4
    loss_weights: dict[str, float] = {"phonemes": 0.4}
    max_audio_seconds: float | None = 35.0
    dropout: float = 0.1
    num_epochs: int = 1
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    output_hidden_size: int | None = None
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    output_dir: str = "./results"
    devset_ratio: float = 0.1
    save_every: int = 1774
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
    warmup_steps: int = 2000
    lr_scheduler_type: str = "constant"
    gradient_checkpoiniting: bool = False
    architecture: ArchitectureName = "w2v2bert-streaming-rnn"
    base_model_name_or_path: str = "facebook/w2v-bert-2.0"
    processor_name_or_path: str | None = "facebook/w2v-bert-2.0"
    ignore_mismatched_sizes: bool = True

    # FastConformer cache-aware specific fields
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    from_nemo_model_name_or_path: str | None = None

    # Streaming-specific fields
    chunk_frames: int = 25
    lookback_frames: int = 5
    lookahead_frames: int = 5
    rnn_hidden_size: int = 256
    rnn_dropout: float = 0.1
    max_chunk_batch: int = 1
    conv_depthwise_kernel_size: int = 31
    max_noise_input_seconds: float = 40.0
    include_noise_ds_in_train: bool = True
    include_noise_ds_in_augment: bool = True

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

    @model_validator(mode="after")
    def set_num_workers(self):
        if self.num_workers is None:
            self.num_workers = min(
                os.cpu_count() or 1, self.per_device_train_batch_size
            )
        return self


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


def compute_per_level_stats(predictions, references, pad_token_idx=0):
    """
    Compute raw Levenshtein distance and total length for a batch.

    Returns (total_distance, total_length) for incremental accumulation.
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
        distance = min(Levenshtein.distance(pred_str, ref_str), len(ref_str))
        total_distance += distance
        total_length += len(ref_str)

    return total_distance, total_length


class IncrementalMetrics:
    """
    Callable compute_metrics for batch_eval_metrics=True mode.

    Accumulates PER stats per batch and only computes final metrics
    when compute_result=True (last eval batch), avoiding ever storing
    the full eval set predictions in memory.
    """

    def __init__(self, pad_token_idx=0):
        self.pad_token_idx = pad_token_idx
        self._running = {}
        self._silence_running = {"errors": 0, "total": 0}

    def __call__(self, eval_pred, compute_result=False):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]
        if isinstance(labels, (tuple, list)):
            labels = labels[0]

        predictions_dict = {
            level: p.detach().cpu().numpy() for level, p in predictions.items()
        }
        labels_dict = {level: l.detach().cpu().numpy() for level, l in labels.items()}

        for level in labels_dict:
            labels_dict[level][labels_dict[level] < 0] = self.pad_token_idx

        pred_labels = {
            level: np.argmax(p, axis=-1) for level, p in predictions_dict.items()
        }

        # Separate speech vs silence by checking label content:
        # Silence samples have all-pad labels (1 non-pad token max from EOS),
        # speech samples have real phoneme tokens. This avoids relying on a
        # shared-state buffer that breaks with multiprocessing DataLoaders.
        batch_size = len(next(iter(labels_dict.values())))
        speech_indices = []
        silence_indices = []
        for i in range(batch_size):
            non_pad = (labels_dict["phonemes"][i] != self.pad_token_idx).sum()
            if non_pad <= 1:
                silence_indices.append(i)
            else:
                speech_indices.append(i)

        # Speech: normal PER computation per level
        if speech_indices:
            for level in labels_dict:
                if level not in self._running:
                    self._running[level] = {"distance": 0, "length": 0}
                d, l = compute_per_level_stats(
                    pred_labels[level][speech_indices],
                    labels_dict[level][speech_indices],
                    pad_token_idx=self.pad_token_idx,
                )
                self._running[level]["distance"] += d
                self._running[level]["length"] += l

        # Silence: frame-level error rate for phonemes only
        if silence_indices:
            sil_pred = pred_labels["phonemes"][silence_indices]
            sil_errors = (sil_pred != self.pad_token_idx).sum()
            sil_total = sil_pred.size
            self._silence_running["errors"] += int(sil_errors)
            self._silence_running["total"] += sil_total

        if compute_result:
            metrics = {}
            n = 0
            total_per = 0.0
            for level, stats in self._running.items():
                per = (
                    stats["distance"] / stats["length"] if stats["length"] > 0 else 0.0
                )
                metrics[f"per_{level}"] = per
                total_per += per
                n += 1
            metrics["average_per"] = total_per / n if n > 0 else 0.0

            # Silence PER (not included in average_per)
            if self._silence_running["total"] > 0:
                metrics["silence_per"] = (
                    self._silence_running["errors"] / self._silence_running["total"]
                )

            self._running = {}
            self._silence_running = {"errors": 0, "total": 0}
            return metrics

        return {}


def build_audiomentations_augs(
    p=0.4, seed=42, all=False, noise_dir_path: str | None = None
):
    """taken form: https://github.com/snakers4/silero-vad/blob/master/tuning/utils.py#L37"""
    # audiomentations usesd python random for its calculations
    random.seed(seed)
    np.random.seed(seed)

    from audiomentations import (
        AddBackgroundNoise,
        AddGaussianNoise,
        AirAbsorption,
        Aliasing,
        BandPassFilter,
        BandStopFilter,
        ClippingDistortion,
        Compose,
        GainTransition,
        HighPassFilter,
        HighShelfFilter,
        LowPassFilter,
        LowShelfFilter,
        Mp3Compression,
        PeakingFilter,
        PitchShift,
        RoomSimulator,
        SevenBandParametricEQ,
        SomeOf,
        TimeStretch,
    )

    transforms = (
        [
            AddBackgroundNoise(sounds_path=noise_dir_path, p=1.0),
        ]
        if noise_dir_path is not None
        else []
    )

    transforms += [
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
        noise_dir_path: str | None = None,
    ):
        self.seed = seed
        self.augment_prob = augment_prob
        self.augment = build_audiomentations_augs(
            1, seed=seed, noise_dir_path=noise_dir_path
        )

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
        audio_dict = item["audio"]
        src = audio_dict["path"] or io.BytesIO(audio_dict["bytes"])
        wav, _ = librosa.load(src, sr=16000, mono=True)
        item["audio"] = self.apply(wav)
        return item


def fix_dataset_len(ds: Dataset, batch_size: int) -> Dataset:
    """Fix dataset len to be multiple of batch_size as the model
    config asserts that config.max_chunk_batch be mutliple of batch_size"""
    ds_len = (len(ds) // batch_size) * batch_size
    return ds.select(range(ds_len))


def prepare_noise_dataset(
    train_config: TrainConfig,
    augmentation_ratio: float = 0.3,
    extend_seconds: float = 1.0,
    extended_seconds_threshold: float = 10.0,
    all_noise_as_input: bool = False,
) -> dict:
    """Split noise dataset into augmentation + input parts.

    Returns:
        {"augmentation": str(path to saved noise WAVs), "input": Dataset}
    """
    noise_ds = load_dataset(
        "obadx/freesound-commercial-50k-noise-only",
        split="train",
    )
    noise_ds = noise_ds.cast_column("audio", Audio(decode=False))

    split_ds = noise_ds.train_test_split(
        test_size=augmentation_ratio,
        generator=np.random.default_rng(train_config.seed),
    )
    aug_ds = split_ds["test"]
    if all_noise_as_input:
        input_ds = noise_ds
    else:
        input_ds = split_ds["train"]

    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    noise_dir = Path(hf_cache) / "noise-dir"

    if not noise_dir.exists():
        noise_dir.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        for i, example in enumerate(tqdm(aug_ds, desc="Saving noise samples")):
            audio_dict = example["audio"]
            src = io.BytesIO(audio_dict["bytes"])
            wav, sr = librosa.load(src, sr=16000, mono=True)
            if len(wav) / sr < extended_seconds_threshold:
                extend_len = int(extend_seconds * sr)
                wav = np.pad(wav, (0, extend_len))
            sf.write(str(noise_dir / f"noise_{i:06d}.wav"), wav, sr)

    input_ds = input_ds.add_column("input_type", ["silence"] * input_ds.num_rows)

    return {"augmentation": str(noise_dir), "input": input_ds}


def prepare_dataset(
    train_config: TrainConfig,
    processor,
    multi_level_tokenizer: MultiLevelTokenizer,
    sample_rate=16000,
    is_testset=False,
    input_noise_ds: Dataset | None = None,
):
    if is_testset:
        moshaf_ids = train_config.test_moshaf_ids
    else:
        moshaf_ids = train_config.train_moshaf_ids
    # concatenate datasets
    ds = concatenate_datasets(
        [
            load_dataset(
                "obadx/muaalem-annotated-v3",
                name=f"moshaf_{m_id}",
                split="train",
                num_proc=train_config.num_workers,
            )
            for m_id in moshaf_ids
        ]
    )

    # disable torchcodec decoding, use liborsa instead
    ds = ds.cast_column("audio", Audio(decode=False))

    def _audio_len(audio_dict):
        src = audio_dict["path"] or io.BytesIO(audio_dict["bytes"])
        wav, _ = librosa.load(src, sr=16000, mono=True)
        return len(wav)

    if train_config.max_audio_seconds is not None:
        # removihg long samples
        max_samples = int(train_config.max_audio_seconds * sample_rate)

        ds = ds.filter(
            lambda ex: _audio_len(ex["audio"]) <= max_samples,
            num_proc=train_config.num_workers,
        )

    # Add input_type column to speech samples
    ds = ds.add_column("input_type", ["speech"] * ds.num_rows)

    if input_noise_ds is not None:
        input_noise_ds = input_noise_ds.cast_column("audio", Audio(decode=False))
        input_noise_ds = input_noise_ds.add_column(
            "uthmani", [""] * input_noise_ds.num_rows
        )
        input_noise_ds = input_noise_ds.add_column(
            "moshaf_id", [""] * input_noise_ds.num_rows
        )
        input_noise_ds = input_noise_ds.add_column(
            "segment_index", ["-1"] * input_noise_ds.num_rows
        )
        input_noise_ds = input_noise_ds.remove_columns(
            [
                "title",
                "description",
                "tags",
                "username",
                "freesound_id",
                "license",
                "attribution_required",
                "commercial_use",
            ]
        )
        ds = concatenate_datasets([ds, input_noise_ds])

    if is_testset:
        return DatasetDict(
            {"test": fix_dataset_len(ds, train_config.pre_device_eval_batch_size)}
        )

    else:
        # split train / validation
        ds = ds.train_test_split(
            test_size=train_config.devset_ratio,
            generator=np.random.default_rng(train_config.seed),
        )

        return DatasetDict(
            {
                "train": fix_dataset_len(
                    ds["train"], train_config.per_device_train_batch_size
                ),
                "validation": fix_dataset_len(
                    ds["test"], train_config.pre_device_eval_batch_size
                ),
            }
        )


def build_model_components(
    train_config: TrainConfig,
    level_to_vocab_size: dict[str, int],
    pad_token_id: int,
    ignore_mismatched_sizes: bool = True,
):
    if train_config.architecture == "fastconformer-cache-aware":
        config = FastConformerCacheAwareMultilevelCTCConfig(
            level_to_vocab_size=level_to_vocab_size,
            pad_token_id=pad_token_id,
            level_to_loss_weight=train_config.loss_weights,
            **train_config.model_kwargs,
        )

        if train_config.from_nemo_model_name_or_path is not None:
            model = FastConformerCacheAwareMultilevelCTC.from_nemo(
                train_config.from_nemo_model_name_or_path,
                config,
                map_location="cpu",
            )
        else:
            model = FastConformerCacheAwareMultilevelCTC(config)

        # Mel-spectrogram extraction happens in the collator via a standalone
        # FastConformerMelProcessor (same config as the model's internal one),
        # mirroring how other architectures preprocess audio outside the model.
        processor = FastConformerMelProcessor(**config.processor_kwargs)
        return processor, config, model

    if train_config.processor_name_or_path is None:
        raise ValueError(
            "`processor_name_or_path` is required for the "
            f"`{train_config.architecture}` architecture.  Got None."
        )

    processor = AutoFeatureExtractor.from_pretrained(
        train_config.processor_name_or_path,
    )

    config = Wav2Vec2BertForRNNStreamingMultilevelCTCConfig(
        level_to_vocab_size=level_to_vocab_size,
        pad_token_id=pad_token_id,
        level_to_loss_weight=train_config.loss_weights,
        attention_dropout=train_config.dropout,
        hidden_dropout=train_config.dropout,
        feat_proj_dropout=train_config.dropout,
        num_hidden_layers=train_config.num_hidden_layers,
        hidden_size=train_config.hidden_size,
        output_hidden_size=train_config.output_hidden_size,
        intermediate_size=train_config.intermediate_size,
        num_attention_heads=train_config.num_attention_heads,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=False,
        adapter_stride=1,
        chunk_frames=train_config.chunk_frames,
        lookback_frames=train_config.lookback_frames,
        lookahead_frames=train_config.lookahead_frames,
        rnn_hidden_size=train_config.rnn_hidden_size,
        rnn_dropout=train_config.rnn_dropout,
        max_chunk_batch=train_config.max_chunk_batch,
        conv_depthwise_kernel_size=train_config.conv_depthwise_kernel_size,
    )

    model = Wav2Vec2BertForRNNStreamingMultilevelCTC.from_pretrained(
        train_config.base_model_name_or_path,
        config=config,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )

    return processor, config, model


@dataclass
class NemoFastConformerBatch:
    processed_signal: torch.Tensor
    processed_length: torch.Tensor


def nemo_fasterconformer_preprocesor(
    waves: list[NDArray[np.float32]],
    processor: FastConformerMelProcessor,
) -> NemoFastConformerBatch:
    """Zero-pad raw waveforms and extract mel features with the NeMo processor."""
    max_len = max(len(w) for w in waves)
    raw_audio = torch.zeros((len(waves), max_len), dtype=torch.float32)
    for i, w in enumerate(waves):
        raw_audio[i, : len(w)] = torch.from_numpy(np.asarray(w))
    processed_signal, processed_length = processor(
        input_signal=raw_audio,
        length=torch.tensor([len(w) for w in waves], dtype=torch.long),
    )
    return NemoFastConformerBatch(
        processed_signal=processed_signal,
        processed_length=processed_length,
    )


@dataclass
class QdatBenchCollator:
    processor: Wav2Vec2BertProcessor | FastConformerMelProcessor | None
    architecture: ArchitectureName = "w2v2bert-streaming-rnn"

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        waves = [decode_audio(f["audio"]).wav for f in features]
        ids = [f["id"] for f in features]
        if self.architecture == "fastconformer-cache-aware":
            batch = asdict(nemo_fasterconformer_preprocesor(waves, self.processor))
        else:
            batch = self.processor(
                waves,
                sampling_rate=16000,
                padding="longest",
                return_tensors="pt",
            )
        batch["id"] = ids
        return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor | FastConformerMelProcessor | None
    multi_level_tokenizer: MultiLevelTokenizer
    moshaf_id_to_moshaf_attr: dict[str, MoshafAttributes]
    augment: Augment
    special_moshaf_id_to_seg_to_moshaf_attr: dict[str, dict[str, MoshafAttributes]]
    max_noise_input_seconds: float = 40.0
    chunk_frames: int = 25
    sample_rate: int = 16000
    architecture: ArchitectureName = "w2v2bert-streaming-rnn"
    apply_augment: bool = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        waves = []
        phonemes_list = []
        sifat_list = []
        features_input_types = []

        min_silence_samples = self.chunk_frames * 20 * self.sample_rate // 1000
        max_silence_samples = int(self.max_noise_input_seconds * self.sample_rate)

        for f in features:
            audio_dict = f["audio"]
            src = io.BytesIO(audio_dict["bytes"])
            wav, _ = librosa.load(src, sr=16000, mono=True)
            input_type = f.get("input_type", "speech")
            features_input_types.append(input_type)

            if input_type == "silence":
                # Enforce min length (chunk_frames * 20ms)
                if len(wav) < min_silence_samples:
                    wav = np.pad(wav, (0, min_silence_samples - len(wav)))
                # Enforce max length
                if len(wav) > max_silence_samples:
                    wav = wav[:max_silence_samples]
            else:
                if self.apply_augment:
                    wav = self.augment.apply(wav)

            waves.append(wav)

        for idx in range(len(features)):
            if features_input_types[idx] == "silence":
                phonemes_list.append("")
                sifat_list.append([])
            else:
                m_id = features[idx]["moshaf_id"]
                if m_id in self.special_moshaf_id_to_seg_to_moshaf_attr:
                    seg_idx = features[idx]["segment_index"]
                    if seg_idx in self.special_moshaf_id_to_seg_to_moshaf_attr[m_id]:
                        moshaf_attr = self.special_moshaf_id_to_seg_to_moshaf_attr[
                            m_id
                        ][seg_idx]
                    else:
                        moshaf_attr = self.moshaf_id_to_moshaf_attr[m_id]
                else:
                    moshaf_attr = self.moshaf_id_to_moshaf_attr[m_id]

                phonized = quran_phonetizer(
                    features[idx]["uthmani"],
                    moshaf_attr,
                    remove_spaces=True,
                )
                phonemes_list.append(phonized.phonemes)
                sifat_list.append(phonized.sifat)

        if self.architecture == "fastconformer-cache-aware":
            batch = asdict(nemo_fasterconformer_preprocesor(waves, self.processor))
        else:
            batch = self.processor(
                waves,
                sampling_rate=16000,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                do_normalize_per_mel_bins=False,  # Very vital bug we have not to normalize mel frequency bins
            )

        labels = self.multi_level_tokenizer.tokenize(
            phonemes_list,
            sifat_list,
            to_dict=True,
            add_eos=True,
            return_tensors="pt",
            padding="longest",
        )

        # Use tokenizer's attention_mask as labels_mask for the model
        batch["labels_mask"] = labels["attention_mask"]

        batch["labels"] = labels["input_ids"]

        return batch


class StreamingTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        apply_augment=True,
        *args,
        **kwargs,
    ):
        """
        Run evaluation, optionally toggling augmentation for the data collator.

        The default collator (``DataCollatorCTCWithPadding``) applies audio
        augmentations to every speech sample, which is desirable for the
        validation/dev set so its metrics reflect the same distribution the
        model is trained on. However, the held-out test set must be scored on
        clean, un-augmented audio to report an honest estimate of generalization
        performance. Since stock ``Trainer.evaluate`` does not forward extra
        kwargs to the collator (it uses ``self.data_collator`` from
        ``get_eval_dataloader``), we swap in a copy of the collator with
        ``apply_augment=False`` for the duration of the evaluation. The swap is
        confined to the branch where the collator setting actually differs, and
        ``try/finally`` guarantees restoration even if evaluation raises, so a
        mid-eval failure can never leave the trainer stuck on the no-augment
        collator. When ``apply_augment`` is not passed (e.g. the in-training
        eval on the dev set), it defaults to ``True`` and nothing is swapped.
        """
        prev = self.data_collator
        if hasattr(prev, "apply_augment") and prev.apply_augment != apply_augment:
            self.data_collator = replace(prev, apply_augment=apply_augment)
            try:
                return super().evaluate(
                    *args,
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                    **kwargs,
                )
            finally:
                self.data_collator = prev
        return super().evaluate(
            *args,
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )


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


def run_qdat_bench_test(
    model: Wav2Vec2BertForRNNStreamingMultilevelCTC
    | FastConformerCacheAwareMultilevelCTC,
    processor: Wav2Vec2BertProcessor | FastConformerMelProcessor | None,
    multi_level_tokenizer: MultiLevelTokenizer,
    output_dir: str | Path,
    vocab: dict[str, dict[str, int]],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
    model_suffix: str = "",
    architecture: ArchitectureName = "w2v2bert-streaming-rnn",
) -> None:
    """
    Run inference + evaluation on the qdat_bench dataset.

    Loads ``obadx/qdat_bench``, runs the model with greedy CTC decoding,
    saves predictions as ``qdat_bench_predictions.jsonl``, then evaluates
    with ``eval_qdat_bench`` and saves metrics as ``qdat_bench_test_results.json``.

    Args:
        model: Trained Wav2Vec2BertForRNNStreamingMultilevelCTC / FastConformer
            cache-aware model (eval mode).
        processor: AutoFeatureExtractor for audio processing (``None`` when
            audio is not preprocessed by a collator processor).
        multi_level_tokenizer: MultiLevelTokenizer instance for eval.
        output_dir: Path to save prediction/results JSON files.
        vocab: Dict mapping level name -> {label: id} (from ``vocab.json``).
        device: torch device (cuda/cpu).
        dtype: torch dtype (bfloat16/float32).
        batch_size: Inference batch size.
        architecture: Model architecture name; controls how audio batches are
            built (HF processor mel features vs raw zero-padded audio).
    """
    ds = load_dataset("obadx/qdat_bench", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    collator = QdatBenchCollator(processor=processor, architecture=architecture)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator,
    )

    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            ids = batch.pop("id")
            if architecture == "fastconformer-cache-aware":
                # processed_signal must stay float32 for the encoder (no bf16)
                batch = {k: v.to(device) for k, v in batch.items()}
            else:
                batch = {k: v.to(device, dtype=dtype) for k, v in batch.items()}

            outputs = model(**batch)
            level_to_logits = outputs.logits

            level_to_labels = {}
            for level in level_to_logits:
                labels = level_to_logits[level].argmax(dim=-1).cpu().numpy()
                level_to_labels[level] = ctc_decode(labels)

            special_ids = set(multi_level_tokenizer.special_tokens)
            for level in level_to_labels:
                level_to_labels[level] = [
                    [t for t in seq if t not in special_ids]
                    for seq in level_to_labels[level]
                ]

            level_to_batch_to_script = {}
            for level in vocab:
                ids_to_ph = {idx: label for label, idx in vocab[level].items()}
                batch_list = []
                for seq in level_to_labels[level]:
                    seq_list = [ids_to_ph[int(label)] for label in seq]
                    batch_list.append(seq_list)
                level_to_batch_to_script[level] = batch_list

            for idx in range(len(ids)):
                results.append(
                    {
                        "id": ids[idx],
                        "levels_labels": {
                            level: [int(x) for x in level_to_labels[level][idx]]
                            for level in level_to_labels
                        },
                        "level_to_scripts": {
                            level: (
                                "".join(level_to_batch_to_script[level][idx])
                                if level == "phonemes"
                                else level_to_batch_to_script[level][idx]
                            )
                            for level in level_to_batch_to_script
                        },
                    }
                )

    pred_path = (
        Path(output_dir) / f"qdat_bench_predictions_{model_suffix}.jsonl"
        if model_suffix
        else Path(output_dir) / "qdat_bench_predictions.jsonl"
    )
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {pred_path}")

    pred_ds = Dataset.from_json(str(pred_path))
    metrics = eval_qdat_bench(
        pred_trans_ds=pred_ds,
        multi_level_tokenizer=multi_level_tokenizer,
        bootstrap=False,
    )

    results_path = (
        Path(output_dir) / f"qdat_bench_test_results_{model_suffix}.json"
        if model_suffix
        else Path(output_dir) / "qdat_bench_test_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"QDAT benchmark results saved to {results_path}")

    print("Speech metrics:", metrics.get("speech_metrics"))
    print("QDAT metrics:", metrics.get("qdat_metrics"))
    print("QDAT avg metrics:", metrics.get("qdat_avg_metrics"))

    return metrics


def compute_qdat_top_phonemes_per(
    pred_trans_ds: Dataset,
    multi_level_tokenizer: MultiLevelTokenizer,
    vocab: dict[str, dict[str, int]],
    qdat_bench_name_or_path: str = "obadx/qdat_bench",
    pad_token_id: int = PAD_TOKEN_IDX,
    eos_token_id: int = 1,
    top_k: int | None = None,
) -> dict:
    """Rank the phonemes that contribute most to the phonemes-level PER.

    Uses ``Levenshtein.editops`` between the CTC-decoded predicted phoneme ids
    and the tokenized reference ids (the exact inputs ``compute_per_level``
    uses), attributing every edit to a single phoneme:

    - substitution (``replace``): the reference phoneme the model confused.
    - insertion (``delete`` in the pred->ref alignment): an extra predicted
      phoneme, credited to the hallucinated predicted phoneme.
    - deletion (``insert`` in the pred->ref alignment): a reference phoneme the
      model missed.

    Args:
        pred_trans_ds: Predictions dataset with ``id`` and
            ``levels_labels[phonemes]`` fields (from the saved
            ``qdat_bench_predictions_{suffix}.jsonl``).
        multi_level_tokenizer: Tokenizer used to build the reference ids.
        vocab: Dict mapping level name -> {label: id} (from ``vocab.json``).
        qdat_bench_name_or_path: The qdat_bench dataset to load.
        pad_token_id: PAD token id to strip from reference ids.
        eos_token_id: EOS token id to strip from reference ids.
        top_k: Limit the returned ``phoneme_ranking`` to the top-k phonemes.

    Returns:
        Dict with overall PER summary (``per_phonemes`` should match
        ``speech_metrics.per_phonemes`` from ``eval_qdat_bench``) and a
        ``phoneme_ranking`` sorted descending by total error contribution.
    """
    qdat_bench_ds = load_dataset(qdat_bench_name_or_path)["train"].cast_column(
        "audio", Audio(decode=False)
    )
    pred_trans_ds = align_preditc_ds(pred_trans_ds, qdat_bench_ds["id"])

    levels_ref = multi_level_tokenizer.tokenize(
        [re.sub(r"\s+", "", ph) for ph in qdat_bench_ds["phonetic_transcript"]],
        qdat_bench_ds["sifat"],
        to_dict=True,
    )
    ref_phonemes_ids = levels_ref["input_ids"]["phonemes"]
    pred_phonemes_ids = [item["phonemes"] for item in pred_trans_ds["levels_labels"]]

    ids_to_ph = {idx: label for label, idx in vocab["phonemes"].items()}
    special_ids = {pad_token_id, eos_token_id}

    tally: dict[str, dict[str, int]] = {}
    ref_counts: dict[str, int] = {}

    total_errors = 0
    total_ref_phonemes = 0

    for pred_ids, ref_ids in zip(pred_phonemes_ids, ref_phonemes_ids):
        pred_ids = [t for t in pred_ids if t not in special_ids]
        ref_ids = [t for t in ref_ids if t not in special_ids]

        for rid in ref_ids:
            label = ids_to_ph[rid]
            ref_counts[label] = ref_counts.get(label, 0) + 1
        total_ref_phonemes += len(ref_ids)

        # Empty predicted phonemes: editops turns every reference phoneme into
        # a deletion (insert op), matching the official PER which charges the
        # full reference length as errors for blank predictions.
        pred_str = sequence_to_chars(pred_ids)
        ref_str = sequence_to_chars(ref_ids)

        sample_errors = 0
        for op, i, j in Levenshtein.editops(pred_str, ref_str):
            if op == "replace":
                label = ids_to_ph[ref_ids[j]]
                err_type = "substitutions"
            elif op == "delete":
                # Extra predicted chars: model insertion.
                label = ids_to_ph[pred_ids[i]]
                err_type = "insertions"
            else:  # insert
                # Missing reference chars: model deletion.
                label = ids_to_ph[ref_ids[j]]
                err_type = "deletions"

            entry = tally.setdefault(
                label, {"substitutions": 0, "insertions": 0, "deletions": 0}
            )
            entry[err_type] += 1
            sample_errors += 1

        # Match the official PER clamp: min(distance, ref_len).
        total_errors += min(sample_errors, len(ref_ids))

    ranking = []
    for label, entry in tally.items():
        total = sum(entry.values())
        ref_count = ref_counts.get(label, 0)
        ranking.append(
            {
                "phoneme": label,
                "total_errors": total,
                "substitutions": entry["substitutions"],
                "insertions": entry["insertions"],
                "deletions": entry["deletions"],
                "ref_count": ref_count,
                "error_rate": total / ref_count if ref_count > 0 else 0.0,
                "per_contribution": (
                    total / total_ref_phonemes if total_ref_phonemes > 0 else 0.0
                ),
            }
        )
    ranking.sort(key=lambda r: (r["total_errors"], r["error_rate"]), reverse=True)

    result = {
        "per_phonemes": total_errors / total_ref_phonemes
        if total_ref_phonemes > 0
        else 0.0,
        "total_errors": total_errors,
        "total_ref_phonemes": total_ref_phonemes,
        "n_samples": len(pred_phonemes_ids),
        "phoneme_ranking": ranking[:top_k] if top_k is not None else ranking,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train/streaming/train_config.yml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub after training",
    )
    parser.add_argument(
        "--rerun-testset",
        action="store_true",
        help="Force re-run test set evaluation even if test_results.json exists",
    )
    parser.add_argument(
        "--rerun-qdat-bench",
        action="store_true",
        help="Force re-run qdat_bench inference + evaluation even if results exist",
    )
    parser.add_argument(
        "--load-last-model",
        action="store_true",
        default=False,
        help="Use the last saved checkpoint instead of the best model when running test set evaluation, qdat_bench, and pushing to the Hub (default: best model)",
    )
    parser.add_argument(
        "--reset-learning-rate",
        action="store_true",
        default=False,
        help="Override the checkpoint's learning rate with the config value on resume (keeps optimizer momentum intact)",
    )
    args = parser.parse_args()
    load_best_model_at_end = not args.load_last_model
    model_suffix = "last" if args.load_last_model else "best"

    # loading wandb tokens and HF login
    load_secrets()
    train_config = TrainConfig.from_yaml(args.config)
    print(train_config)
    multi_level_tokenizer = MultiLevelTokenizer("./vocab_streaming")

    vocab, level_to_vocab_size = load_vocab()
    for level in train_config.loss_weights:
        if level not in level_to_vocab_size:
            raise ValueError(
                f"The level `{level}` does not exist availabel are: `{list(level_to_vocab_size.values())}`"
            )

    # Load pre-trained model components
    processor, config, model = build_model_components(
        train_config,
        level_to_vocab_size,
        PAD_TOKEN_IDX,
        ignore_mismatched_sizes=train_config.ignore_mismatched_sizes,
    )

    # WARN: Note used but for histoy leaving it
    # Freeze all layers except feature_projection to recover from the feature
    # normalization bug. Only the feature projection (layer_norm + projection)
    # will be trained while the pretrained encoder weights stay intact.
    # model.freeze_all_except_feature_projection()

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

    if args.push_to_hub and processor is not None:
        processor.push_to_hub(train_config.hub_model_id)
        multi_level_tokenizer.get_tokenizer().push_to_hub(train_config.hub_model_id)

    # Initializaze wanddb
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = train_config.wandb_project_name

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "false"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # Prepare noise dataset (only the parts enabled by the config flags)
    use_noise = (
        train_config.include_noise_ds_in_train
        or train_config.include_noise_ds_in_augment
    )
    noise_data = prepare_noise_dataset(train_config) if use_noise else None

    input_noise_ds = (
        noise_data["input"]
        if noise_data and train_config.include_noise_ds_in_train
        else None
    )
    noise_dir_path = (
        noise_data["augmentation"]
        if noise_data and train_config.include_noise_ds_in_augment
        else None
    )

    # Load dataset
    # Update with your dataset path
    dataset = prepare_dataset(
        train_config,
        processor,
        multi_level_tokenizer,
        input_noise_ds=input_noise_ds,
    )

    # Configure training arguments
    training_args = TrainingArguments(
        seed=train_config.seed,
        output_dir=train_config.output_dir,
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
        logging_dir=str(Path(train_config.output_dir) / "logs"),
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=train_config.metric_for_bet_model,
        greater_is_better=train_config.greater_is_better,
        # push_to_hub=True,  # this pushed every checkpoint to the hup we want to push the best model only
        hub_model_id=train_config.hub_model_id,  # Update with your model name
        bf16=True,
        warmup_steps=train_config.warmup_steps,
        optim="adamw_torch",
        lr_scheduler_type=train_config.lr_scheduler_type,
        report_to=["tensorboard", "wandb"],
        gradient_checkpointing=train_config.gradient_checkpoiniting,  # Optional for memory savings
        save_total_limit=3,
        hub_strategy="all_checkpoints",  # pushes all checkpoints to the Hub with one checkpoint per subfolder in your model repository
        remove_unused_columns=False,
        batch_eval_metrics=True,
    )
    print(training_args)

    # Initialize label processor
    data_collector = DataCollatorCTCWithPadding(
        processor=processor,
        multi_level_tokenizer=multi_level_tokenizer,
        moshaf_id_to_moshaf_attr=moshaf_id_to_moshaf_attr,
        augment=Augment(
            augment_prob=train_config.augment_prob,
            seed=train_config.seed,
            noise_dir_path=noise_dir_path,
        ),
        special_moshaf_id_to_seg_to_moshaf_attr=special_moshaf_id_to_seg_to_moshaf_attr,
        max_noise_input_seconds=train_config.max_noise_input_seconds,
        chunk_frames=train_config.chunk_frames,
        architecture=train_config.architecture,
    )

    # Initialize Trainer
    trainer = StreamingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=IncrementalMetrics(
            pad_token_idx=PAD_TOKEN_IDX,
        ),
        data_collator=data_collector,
    )

    # Start training
    if list(Path(train_config.output_dir).glob("checkpoint-*")):
        print("Resuming !")
        if args.reset_learning_rate:
            print(
                f"🚨🚨🚨🚨🚨🚨  Resting Learning rate to {train_config.learning_rate}"
            )
            # Monkey-patch _load_optimizer_and_scheduler to override LR after checkpoint restore.
            # This keeps Adam momentum/variance intact — only the LR value changes.
            _original_load_opt = trainer._load_optimizer_and_scheduler

            def _patched_load_opt(checkpoint):
                _original_load_opt(checkpoint)
                new_lr = train_config.learning_rate
                for group in trainer.optimizer.param_groups:
                    group["lr"] = new_lr
                if trainer.lr_scheduler is not None:
                    trainer.lr_scheduler.base_lrs = [new_lr] * len(
                        trainer.optimizer.param_groups
                    )
                    if hasattr(trainer.lr_scheduler, "get_last_lr"):
                        for i in range(len(trainer.lr_scheduler.get_last_lr())):
                            trainer.lr_scheduler.optimizer.param_groups[i]["lr"] = (
                                new_lr
                            )

            trainer._load_optimizer_and_scheduler = _patched_load_opt

        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Final evaluation on test set
    test_results_path = (
        Path(train_config.output_dir) / f"test_results_{model_suffix}.json"
    )
    if train_config.test_moshaf_ids is not None:
        if test_results_path.exists() and not args.rerun_testset:
            print(
                f"Found existing {test_results_path}, skipping test evaluation. Use --rerun-testset to force."
            )
        else:
            testset = prepare_dataset(
                train_config, processor, multi_level_tokenizer, is_testset=True
            )
            test_results = trainer.evaluate(
                testset["test"], metric_key_prefix="test_", apply_augment=False
            )
            with open(test_results_path, "w") as f:
                json.dump(test_results, f, indent=4)
            print("Test Results:", test_results)

    # QDAT benchmark evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    qdat_pred_path = (
        Path(train_config.output_dir) / f"qdat_bench_predictions_{model_suffix}.jsonl"
    )
    qdat_results_path = (
        Path(train_config.output_dir) / f"qdat_bench_test_results_{model_suffix}.json"
    )

    if qdat_results_path.exists() and not args.rerun_qdat_bench:
        print(
            f"Found existing {qdat_results_path}, skipping qdat_bench. Use --rerun-qdat-bench to force."
        )
    elif qdat_pred_path.exists() and not args.rerun_qdat_bench:
        print("Found predictions file, running qdat_bench evaluation only...")
        pred_ds = Dataset.from_json(str(qdat_pred_path))
        metrics = eval_qdat_bench(
            pred_trans_ds=pred_ds,
            multi_level_tokenizer=multi_level_tokenizer,
            bootstrap=False,
        )
        with open(qdat_results_path, "w") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"QDAT benchmark results saved to {qdat_results_path}")
    else:
        print("Running full qdat_bench test...")
        run_qdat_bench_test(
            model=trainer.model,
            processor=processor,
            multi_level_tokenizer=multi_level_tokenizer,
            output_dir=train_config.output_dir,
            vocab=vocab,
            device=device,
            dtype=dtype,
            batch_size=1,
            model_suffix=model_suffix,
            architecture=train_config.architecture,
        )

    # Per-phoneme PER analysis on the phonemes level (cached like the other
    # qdat_bench results, re-computed only when --rerun-qdat-bench is passed).
    qdat_top_phonemes_path = (
        Path(train_config.output_dir) / f"qdat_top_phonemes_per_{model_suffix}.json"
    )
    if qdat_top_phonemes_path.exists() and not args.rerun_qdat_bench:
        print(
            f"Found existing {qdat_top_phonemes_path}, skipping top-phonemes PER analysis. "
            "Use --rerun-qdat-bench to force."
        )
    elif qdat_pred_path.exists():
        print("Running top-phonemes PER analysis...")
        pred_ds = Dataset.from_json(str(qdat_pred_path))
        top_phonemes = compute_qdat_top_phonemes_per(
            pred_trans_ds=pred_ds,
            multi_level_tokenizer=multi_level_tokenizer,
            vocab=vocab,
        )
        with open(qdat_top_phonemes_path, "w") as f:
            json.dump(top_phonemes, f, indent=4, ensure_ascii=False)
        print(f"Top phonemes PER analysis saved to {qdat_top_phonemes_path}")
        print(f"Phonemes PER: {top_phonemes['per_phonemes']:.6f}")
        for rank in top_phonemes["phoneme_ranking"][:10]:
            print(
                f"  {rank['phoneme']}: total={rank['total_errors']} "
                f"(sub={rank['substitutions']} ins={rank['insertions']} "
                f"del={rank['deletions']}) err_rate={rank['error_rate']:.4f}"
            )
    else:
        print(
            "Predictions file missing; re-run with --rerun-qdat-bench to regenerate "
            "and compute top-phonemes PER analysis."
        )

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

    # Push model and tokenizer to Hub
    if args.push_to_hub:
        trainer.push_to_hub()

        api = HfApi()
        for fname in [
            f"test_results_{model_suffix}.json",
            f"qdat_bench_test_results_{model_suffix}.json",
            f"qdat_top_phonemes_per_{model_suffix}.json",
        ]:
            fpath = Path(train_config.output_dir) / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=train_config.hub_model_id,
                    repo_type="model",
                )
                print(f"Uploaded {fname} to {train_config.hub_model_id}")
