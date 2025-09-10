import json
import argparse
from transformers import AutoFeatureExtractor
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from multi_level_ctc_model.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
import os
import json
from tqdm import tqdm


def ctc_decode(batch_arr, blank_id=0, collapse_consecutive=True) -> list:
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
            decoded_list.append(tokens)
        else:
            tokens = seq[seq != blank_id]
            decoded_list.append(tokens)
    return decoded_list


def decoed_phonemes_batch(batch: list[list[int]], vocab: dict) -> list[str]:
    ids_to_ph = list(vocab["phonemes"].keys())
    batch_str = []
    for seq in batch:
        seq_str = ""
        for label in seq:
            seq_str += ids_to_ph[label]
        batch_str.append(seq_str)
    return batch_str


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoFeatureExtractor

    def __call__(self, features):
        waves = [f["audio"]["array"] for f in features]
        ids = [f["id"] for f in features]
        original_ids = [f["original_id"] for f in features]

        batch = self.processor(
            waves,
            sampling_rate=16000,
            padding="longest",
            return_tensors="pt",
        )
        batch["id"] = ids
        batch["original_id"] = original_ids
        return batch


def main(args):
    # Load model and processor
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(args.model_id)
    processor = AutoFeatureExtractor.from_pretrained(args.model_id)
    with open("./vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Load dataset
    test_ds = load_dataset(args.dataset_name, split=args.split)

    # Filter dataset if specific IDs are provided
    if args.ids:
        test_ds = test_ds.filter(lambda x: x["id"] in args.ids)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Create DataLoader
    dataloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    model.to(device, dtype=dtype)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            ids = batch.pop("id")
            original_ids = batch.pop("original_id")
            batch = {k: v.to(device, dtype=dtype) for k, v in batch.items()}

            outputs = model(**batch)
            phonemes_logits = outputs[0]["phonemes"]
            phonemes_labels = phonemes_logits.argmax(dim=-1).cpu().tolist()
            phonemes_labels = ctc_decode(phonemes_labels)
            phonemes_batch = decoed_phonemes_batch(phonemes_labels, vocab=vocab)

            # Convert logits to CPU and process
            for idx in range(len(ids)):
                results.append(
                    {
                        "id": ids[idx],
                        "original_id": original_ids[idx],
                        "phonemes": phonemes_batch[idx],
                    }
                )

    # Create output filename based on model ID
    if args.output_file:
        output_filename = args.output_file
    else:
        # Extract model name for filename
        model_name = (
            args.model_id.split("/")[-1] if "/" in args.model_id else args.model_id
        )
        output_filename = f"{model_name}_predictions.jsonl"

    # Save to JSONL file
    with open(output_filename, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on multiple model versions"
    )

    # Required arguments
    parser.add_argument(
        "--model-id",
        default="obadx/muaalem-model-v3_2",
        help="Model identifier from Hugging Face Hub or local path",
    )

    # Optional arguments
    parser.add_argument(
        "--dataset-name",
        default="obadx/qdat",
        help="Dataset name to load from Hugging Face Hub",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (e.g., train, test, validation)",
    )
    parser.add_argument(
        "--output-file",
        help="Output filename. If not provided, will generate based on model name",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument(
        "--ids", nargs="+", help="Specific IDs to process (space separated)"
    )

    args = parser.parse_args()
    main(args)
