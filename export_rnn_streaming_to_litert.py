#!/usr/bin/env python3
"""
Export RNN Streaming model to LiteRT (.tflite) format in streaming mode.
Produces float32, int8, and int4 quantized versions.

The exported TFLite model accepts rnn_history (h0, c0) as inputs and outputs
updated (h_n, c_n), so the caller manages RNN state externally.

Usage:
    uv run export_rnn_streaming_to_litert.py [--model-id MODEL_ID] [--output-dir OUTPUT_DIR]
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor
import litert_torch
from ai_edge_quantizer import quantizer, recipe


from prepare_quran_dataset.modeling_streaming_rnn.inference import (
    cal_samples_from_frames,
)
from prepare_quran_dataset.modeling_streaming_rnn.modeling_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTC,
)
from prepare_quran_dataset.modeling_streaming_rnn.configuration_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCConfig,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_sample(model, processor):
    config = model.config
    streaming_len = (
        config.chunk_frames + config.lookback_frames + config.lookahead_frames
    )
    input_samples = cal_samples_from_frames(streaming_len)
    dummy_audio = np.zeros(input_samples, dtype=np.float32)
    inputs = processor(
        dummy_audio,
        sampling_rate=16000,
        return_tensors="pt",
        do_normalize_per_mel_bins=False,
    )
    h0 = torch.zeros(1, 1, config.rnn_hidden_size)
    c0 = torch.zeros(1, 1, config.rnn_hidden_size)
    return (
        inputs["input_features"].float(),
        inputs["attention_mask"].float(),
        h0,
        c0,
    )


class MuaalemStreamingWrapper(torch.nn.Module):
    def __init__(self, model, selected_levels: Optional[set[str]] = None):
        super().__init__()
        self.model = model
        self.selected_levels = selected_levels

    def forward(self, input_features, attention_mask, h0, c0):
        out = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            rnn_history=(h0, c0),
            stream_inference=True,
            selected_levels=self.selected_levels,
        )
        return tuple(out.logits[name] for name in out.logits) + (
            out.rnn_history[0],
            out.rnn_history[1],
        )


def main():

    parser = argparse.ArgumentParser(
        description="Export RNN Streaming to LiteRT (streaming mode)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="rnn_streaming",
        help="Output file prefix (default: rnn_streaming)",
    )
    parser.add_argument(
        "--model-id",
        default="./results-streaming-rnn-v3/checkpoint-46124",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR,
        help="Directory to save .tflite files (default: script directory)",
    )
    parser.add_argument(
        "--selected-levels",
        type=str,
        default=None,
        help="Comma-separated list of level names to include. "
        "If omitted, all levels are used. "
        "Example: --selected-levels 'phonemes,tafkheem_or_taqeeq'",
    )
    args = parser.parse_args()
    args.selected_levels = (
        set(args.selected_levels.split(",")) if args.selected_levels else None
    )
    os.makedirs(args.output_dir, exist_ok=True)

    f32_path = os.path.join(args.output_dir, f"{args.prefix}_float32.tflite")
    int8_path = os.path.join(args.output_dir, f"{args.prefix}_int8.tflite")
    int4_path = os.path.join(args.output_dir, f"{args.prefix}_int4.tflite")

    print("Loading model...")
    config = Wav2Vec2BertForRNNStreamingMultilevelCTCConfig.from_pretrained(
        args.model_id, max_chunk_batch=1
    )
    processor = AutoFeatureExtractor.from_pretrained("obadx/muaalem-model-v3_2")
    model = Wav2Vec2BertForRNNStreamingMultilevelCTC.from_pretrained(
        args.model_id, config=config
    )

    wrapped = (
        MuaalemStreamingWrapper(model, selected_levels=args.selected_levels)
        .eval()
        .float()
    )
    sample = _make_sample(model, processor)

    # 1. float32
    print("[1/3] Converting float32...")
    tflite_f32 = litert_torch.convert(wrapped, sample)
    tflite_f32.export(f32_path)
    print(f"Saved {f32_path} ({os.path.getsize(f32_path) / 1e6:.1f} MB)")

    # 2. int8 dynamic
    print("[2/3] Quantizing to int8...")
    qt = quantizer.Quantizer(f32_path)
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    qt.quantize().export_model(int8_path)
    print(f"Saved {int8_path} ({os.path.getsize(int8_path) / 1e6:.1f} MB)")

    # 3. int4 dynamic
    print("[3/3] Quantizing to int4...")
    qt = quantizer.Quantizer(f32_path)
    qt.load_quantization_recipe(recipe.dynamic_wi4_afp32())
    qt.quantize().export_model(int4_path)
    print(f"Saved {int4_path} ({os.path.getsize(int4_path) / 1e6:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
