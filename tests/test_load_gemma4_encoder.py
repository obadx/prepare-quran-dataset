import torch
import numpy as np
from transformers import Gemma4AudioModel, Gemma4AudioFeatureExtractor, AutoConfig


config = AutoConfig.from_pretrained(
    "rnagabh/gemma4-audio-encoder",
)
# Load audio encoder directly from this repo
audio_tower = Gemma4AudioModel.from_pretrained(
    "rnagabh/gemma4-audio-encoder",
    config=config,
    torch_dtype=torch.bfloat16,
)
print(audio_tower)
# audio_tower.to("cuda")
audio_tower.eval()

# Load feature extractor (saved in this repo)
feature_extractor = Gemma4AudioFeatureExtractor.from_pretrained(
    "rnagabh/gemma4-audio-encoder"
)

# Extract features from audio
waveform = np.random.randn(16000).astype(np.float32)  # 4s @ 16kHz
inputs = feature_extractor([waveform], sampling_rate=16000, return_tensors="pt")
print(inputs.keys())
print(f" Input Feature Shape: {inputs['input_features'].shape}")

with torch.no_grad():
    mel = inputs["input_features"].to(dtype=torch.bfloat16)

    # === Option 1: Text-projected embeddings (1536-dim) ===
    # Use this if feeding into an LLM or need the full model output.
    output = audio_tower(mel)
    text_projected = output.last_hidden_state  # (1, 100, 1536)
    print(f"Text projected output shape: {text_projected.shape}")

# === Option 2: Pure audio embeddings (1024-dim) ===
# Captures the conformer output BEFORE the text projection layer.
# Recommended for downstream audio tasks (classification, verification, etc.)
# Note: this registers a hook and runs a separate forward pass.
pre_proj_features = {}


def hook_fn(module, input, output):
    pre_proj_features["hidden"] = input[0]


handle = audio_tower.output_proj.register_forward_hook(hook_fn)
with torch.no_grad():
    _ = audio_tower(mel)
handle.remove()
audio_embeddings = pre_proj_features["hidden"]  # (1, 100, 1024)
print(f"Ouput only shape: {audio_embeddings.shape}")
