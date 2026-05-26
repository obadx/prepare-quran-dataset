import torch
from transformers import WhisperModel, AutoFeatureExtractor
from transformers.models.whisper.configuration_whisper import WhisperConfig
from prepare_quran_dataset.modeling_whisper.modeling_multi_level_ctc_whisper_encoder import (
    WhisperEncoderVariableLength,
)

# 1. Load the full whisper-small and extract only the encoder state dict
full_model = WhisperModel.from_pretrained("openai/whisper-small")
encoder_state_dict = {
    k.removeprefix("encoder."): v
    for k, v in full_model.state_dict().items()
    if k.startswith("encoder.")
}
del full_model  # free memory

# 2. Create your custom encoder with the original config
config = WhisperConfig.from_pretrained("openai/whisper-small")
encoder = WhisperEncoderVariableLength(config)

# 3. Load the pretrained weights – strict loading will succeed now
encoder.load_state_dict(encoder_state_dict)

# 4. Use feature extractor as before
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
audio = torch.zeros(1, int(16000 * 0.7))
inputs = feature_extractor(
    audio.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=False
)
mel = inputs.input_features

# 5. Forward – now uses actual pretrained features
with torch.no_grad():
    outputs = encoder(mel)
print(outputs[0].shape)  # (1, 35, 768) – but this time with real weights

# encoder.save_pretrained("./whisper-small-encoder-only")

