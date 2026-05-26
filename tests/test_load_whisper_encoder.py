import torch
from transformers import AutoFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.models.whisper.configuration_whisper import WhisperConfig
from prepare_quran_dataset.modeling_whisper.modeling_multi_level_ctc_whisper_encoder import (
    WhisperEncoderForMultilevelCTC,
)

# Load encoder and feature extractor
model_id = "openai/whisper-small"
config = WhisperConfig.from_pretrained(
    model_id,
    # max_source_positions=35,
)
encoder = WhisperEncoderForMultilevelCTC.from_pretrained(model_id, config=config)
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")

# Create 0.7 seconds of silence at 16 kHz
audio = torch.zeros(1, int(16000 * 0.7))  # shape: (1, 11200)

# Convert to log‑mel spectrogram (the encoder's expected input)
inputs = feature_extractor(
    audio.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=False
)
print("Inut Features")
print(inputs["input_features"].shape)
input_features = inputs.input_features  # (1, 80, 70)

# Forward through encoder
with torch.no_grad():
    outputs = encoder(input_features)

# Expected output: (batch, seq_len, d_model) -> (1, 18, 768)
print("Outputs")
print(outputs[0].shape)
