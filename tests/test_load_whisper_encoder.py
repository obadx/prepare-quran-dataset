import torch
from transformers import WhisperModel, AutoFeatureExtractor
from transformers.models.whisper.configuration_whisper import WhisperConfig
from prepare_quran_dataset.modeling_whisper.modeling_multi_level_ctc_whisper_encoder import (
    WhisperEncoderVariableLength,
    WhisperEncoderForMultilevelCTC,
)
from prepare_quran_dataset.modeling_whisper.configuration_for_multi_level_ctc import (
    WhisperEncoderForMultilevelCTCConfig,
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

# 4. Wrap in the outer model and save a proper checkpoint
outer_config = WhisperEncoderForMultilevelCTCConfig(
    level_to_vocab_size={"phonemes": 43},
)
outer_model = WhisperEncoderForMultilevelCTC(outer_config)
outer_model.encoder = encoder  # substitute with pretrained encoder

# Saving the model
# outer_model.save_pretrained("./whisper-small-encoder-only")

# 5. Verify the saved checkpoint loads correctly
print("Verifying saved checkpoint...")
loaded = WhisperEncoderForMultilevelCTC.from_pretrained(
    "./whisper-small-encoder-only",
    config=outer_config,
)
print("Checkpoint verified successfully!")

# 6. Use feature extractor as before
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "openai/whisper-small",
)
audio1 = [0] * int(0.7 * 16000)
audio2 = [0] * int(0.75 * 16000)
inputs = feature_extractor(
    [audio1, audio2],
    sampling_rate=16000,
    return_tensors="pt",
    padding="longest",
    return_attention_mask=True,
)
print(inputs.keys())
print(f"Inputs Shape: {inputs['input_features'].shape}")
print(f"Masks Shape: {inputs['attention_mask'].shape}")
mel = inputs.input_features

# 7. Forward through the loaded model
with torch.no_grad():
    outputs = loaded.encoder(mel)
print(outputs[0].shape)  # (1, 35, 768) – with real weights
