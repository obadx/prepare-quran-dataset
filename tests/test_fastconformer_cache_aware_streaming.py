from math import floor

import torch
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from omegaconf import OmegaConf

vocab = [f"c{i:02d}" for i in range(50)]

cfg = OmegaConf.create(
    {
        "name": "FastConformer-CTC-BPE-Streaming",
        "model": {
            "sample_rate": 16000,
            "log_prediction": True,
            "ctc_reduction": "mean_batch",
            "skip_nan_grad": False,
            "train_ds": None,
            "validation_ds": None,
            "test_ds": None,
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "sample_rate": 16000,
                "normalize": "NA",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 80,
                "n_fft": 512,
                "frame_splicing": 1,
                "dither": 0.00001,
                "pad_to": 0,
            },
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 80,
                "feat_out": -1,
                "n_layers": 17,
                "d_model": 512,
                "use_bias": True,
                "subsampling": "dw_striding",
                "subsampling_factor": 4,
                "subsampling_conv_channels": 256,
                "causal_downsampling": True,
                "ff_expansion_factor": 4,
                "self_attention_model": "rel_pos",
                "n_heads": 8,
                "att_context_size": [78, 12],
                "att_context_style": "chunked_limited",
                "att_context_probs": None,
                "xscaling": True,
                "pos_emb_max_len": 5000,
                "conv_kernel_size": 9,
                "conv_norm_type": "layer_norm",
                "conv_context_size": "causal",
                "dropout": 0.1,
                "dropout_pre_encoder": 0.1,
                "dropout_emb": 0.0,
                "dropout_att": 0.1,
                "stochastic_depth_drop_prob": 0.0,
                "stochastic_depth_mode": "linear",
                "stochastic_depth_start_layer": 1,
            },
            "decoder": {
                "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
                "feat_in": None,
                "num_classes": 50,
                "vocabulary": vocab,
            },
        },
    }
)

print("Initializing Cache-Aware Streaming FastConformer...")
model = EncDecCTCModel(cfg=cfg.model)
model.eval()

chunk_len_s = 0.52
lookahead_s = 0.48
sample_rate = 16000
B = 1

chunk_samples = int(chunk_len_s * sample_rate)
lookahead_samples = int(lookahead_s * sample_rate)
total_samples = chunk_samples + lookahead_samples
audio_input = torch.randn(B, total_samples) * 0.1
audio_length = torch.full((B,), total_samples, dtype=torch.long)

print(f"Chunk shape: {chunk_samples} ({int(chunk_len_s * 1000)}ms)")
print(f"Lookahead shape: {lookahead_samples} ({int(lookahead_s * 1000)}ms)")
print(f"Total audio input shape: {audio_input.shape}")

# Preprocess audio to mel spectrogram
print("\nPreprocessing audio...")
with torch.no_grad():
    processed_signal, processed_length = model.preprocessor(
        input_signal=audio_input, length=audio_length
    )
print(f"Processed signal shape: {processed_signal.shape}")
print(f"Processed length: {processed_length}")

# Setup streaming params
print("\nSetting up streaming params...")
model.encoder.setup_streaming_params()

# Get initial cache
print("Getting initial cache state...")
cache_last_channel, cache_last_time, cache_last_channel_len = (
    model.encoder.get_initial_cache_state(batch_size=B)
)
print(f"cache_last_channel shape: {cache_last_channel.shape}")
print(f"cache_last_time shape: {cache_last_time.shape}")
print(f"cache_last_channel_len: {cache_last_channel_len}")

# Fill cache with random values to simulate previous context
cache_last_channel = torch.randn_like(cache_last_channel) * 0.01
cache_last_time = torch.randn_like(cache_last_time) * 0.01

# Run encoder with cache
print("\nRunning encoder with cache-aware streaming...")
with torch.no_grad():
    encoder_out, out_lens, new_cache_ch, new_cache_t, new_cache_len = model.encoder(
        audio_signal=processed_signal,
        length=processed_length,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_last_channel_len=cache_last_channel_len,
    )
    print("Encoder Ouput with without truncating lookahead")
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Output lengths: {out_lens}")
    print()

    rets = model.encoder.forward_internal(
        audio_signal=processed_signal,
        length=processed_length,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_last_channel_len=cache_last_channel_len,
    )
    encoder_out, out_lens, new_cache_ch, new_cache_t, new_cache_len = (
        model.encoder.streaming_post_process(rets, keep_all_outputs=False)
    )


print(f"Encoder output shape: {encoder_out.shape}")
print(f"Output lengths: {out_lens}")

logits = model.decoder(encoder_output=encoder_out)
print(f"Logits shape: {logits.shape}")

blank_id = 0
argmax_ids = logits.argmax(dim=-1)[0].tolist()

decoded_chars = []
prev_id = None
for token_id in argmax_ids:
    if token_id != prev_id and token_id != blank_id:
        decoded_chars.append(vocab[token_id - 1])
    prev_id = token_id

decoded_text = "".join(decoded_chars)

print("\n--- Inference Results ---")
print(f"Encoder output shape : {tuple(encoder_out.shape)}")
print(f"Logits shape         : {tuple(logits.shape)}")
print(f"Argmax frame IDs     : {argmax_ids}")
print(f"Decoded output       : '{decoded_text}'")
print(f"New cache_ch shape   : {new_cache_ch.shape}")
print(f"New cache_t shape    : {new_cache_t.shape}")
print(f"New cache len        : {new_cache_len}")
print("-------------------------\n")
