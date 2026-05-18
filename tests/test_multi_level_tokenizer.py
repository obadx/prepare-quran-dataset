import json
import string

from quran_transcript import quran_phonetizer, Aya, MoshafAttributes
from time import perf_counter
import numpy as np
from numpy.typing import NDArray
import Levenshtein

from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer


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


def labels_to_logits(label_dict):
    logit_dict = {}
    for key, label_array in label_dict.items():
        # Assuming label_array is 2D with shape (batch_size, 1) or (batch_size,)
        labels = label_array.squeeze()  # Remove singleton dimensions if any

        # Determine number of classes
        num_classes = int(np.max(labels) + 1)  # Assumes classes are 0, 1, 2,...,n-1

        # Convert to one-hot encoding
        one_hot = np.eye(num_classes)[labels]

        # You might want to convert these to logits by applying inverse softmax
        # But typically labels would be converted to one-hot for loss functions
        # If you truly want "logits" from labels, this is unusual, but you could:
        # Add some small noise to avoid infinite values when log is applied
        epsilon = 1e-10
        noisy_one_hot = one_hot * (1 - 2 * epsilon) + epsilon

        # Inverse softmax to get logits (this is unconventional)
        logits = np.log(noisy_one_hot)

        logit_dict[key] = logits

    return logit_dict


if __name__ == "__main__":
    tokenizer = MultiLevelTokenizer("./")

    uth_strings = [Aya(2, i).get().uthmani for i in range(1, 2)]
    moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=4,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=4,
        tasheel_or_madd="madd",
    )

    photenized_outs = [
        quran_phonetizer(
            uth_str,
            moshaf,
            remove_spaces=True,
        )
        for uth_str in uth_strings
    ]

    start = perf_counter()
    token_out = tokenizer.tokenize(
        [p.phonemes for p in photenized_outs],
        [p.sifat for p in photenized_outs],
        to_dict=True,
        padding="longest",
        return_tensors="np",
    )
    for level in token_out["input_ids"]:
        mask = token_out["attention_mask"][level] == 0
        token_out["input_ids"][level][mask] = -100
    print(f"Total tokenization time: {perf_counter() - start} ")

    pred_logits = labels_to_logits(token_out["input_ids"])
    labels = {
        "phonemes": np.array(
            [
                [1, 1, 2, -100, -100],
                [5, 5, 4, -100, -100],
            ]
        )
    }
    pred = {
        "phonemes": np.array(
            [
                [[0, 9, 0], [9, 0, 0], [0, 9, 0], [9, 0, 0]],
                [[0, 9, 0], [9, 0, 0], [0, 9, 0], [9, 0, 0]],
            ]
        )
    }

    metrics = compute_metrics((pred, labels))
    print(metrics)

    # print(token_out["input_ids"]["phonemes"])

    # print(json.dumps(token_out["input_ids"], indent=1))
    #
    # for level in token_out["input_ids"]:
    #     mask = token_out["input_ids"][level] == -100
    #     token_out["input_ids"][level][mask] = 0
    # decoded_outs = tokenizer.decode(
    #     token_out["input_ids"],
    #     place_zeros_in_between=True,
    # )
    # print(json.dumps(decoded_outs, indent=1, ensure_ascii=False))
