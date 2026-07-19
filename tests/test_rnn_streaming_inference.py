import numpy as np
import torch
from librosa import load

from prepare_quran_dataset.modeling_streaming_rnn.inference import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCInference,
)
from prepare_quran_dataset.modeling_streaming_rnn.modeling_rnn_streaming_multi_level_ctc import (
    convert_input_to_chunked_for_offline,
)
from prepare_quran_dataset.modeling_streaming_rnn.multi_level_tokenizer import (
    MultiLevelTokenizer,
)


def ctc_decode(batch_arr, blank_id=0, collapse_consecutive=True) -> list[np.ndarray]:
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


def run_original_streaming(rnn_inf, audio, ph_ids_to_str):
    out_ids = []
    n = rnn_inf.get_chunk_samples()
    rnn_inf.reset()
    consumed = 0
    while consumed < len(audio):
        wav = audio[consumed : consumed + n]
        if len(wav) < n:
            wav = np.pad(wav, (0, n - len(wav)))
        for output in rnn_inf(wav):
            out_ids += output.phonemes_ids
        consumed += n
    decoded = ctc_decode([np.array(out_ids)])[0].tolist()
    text = "".join(ph_ids_to_str[idx] for idx in decoded)
    return decoded, text


def run_chunk_aligned_streaming(rnn_inf, audio, ph_ids_to_str, device):
    inputs = rnn_inf.processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device, dtype=rnn_inf.dtype)
    attention_mask = inputs["attention_mask"].to(device, dtype=rnn_inf.dtype)

    cfg = rnn_inf.config
    chunk_frames = cfg.chunk_frames
    lookback_frames = cfg.lookback_frames
    lookahead_frames = cfg.lookahead_frames

    feat_chunked = convert_input_to_chunked_for_offline(
        input_features,
        lookback=lookback_frames,
        chunk=chunk_frames,
        lookahead=lookahead_frames,
        max_chunk_batch_size=1,
    )
    mask_chunked = convert_input_to_chunked_for_offline(
        attention_mask,
        lookback=lookback_frames,
        chunk=chunk_frames,
        lookahead=lookahead_frames,
        max_chunk_batch_size=1,
    )

    out_ids = []
    rnn_history = None
    num_chunks = feat_chunked.shape[0]
    for i in range(num_chunks):
        chunk_feat = feat_chunked[i : i + 1]
        chunk_mask = mask_chunked[i : i + 1]
        outputs = rnn_inf.model(
            input_features=chunk_feat,
            attention_mask=chunk_mask,
            stream_inference=True,
            rnn_history=rnn_history,
            return_dict=True,
        )
        phoneme_ids = outputs.logits["phonemes"].argmax(dim=-1)[0].tolist()
        out_ids += phoneme_ids
        rnn_history = outputs.rnn_history

    decoded = ctc_decode([np.array(out_ids)])[0].tolist()
    text = "".join(ph_ids_to_str[idx] for idx in decoded)
    return decoded, text


def run_offline(rnn_inf, audio, ph_ids_to_str, device):
    rnn_inf.reset()
    inputs = rnn_inf.processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device, dtype=rnn_inf.dtype) for k, v in inputs.items()}
    outputs = rnn_inf.model(**inputs, stream_inference=False, return_dict=True)
    phoneme_logits = outputs.logits["phonemes"]
    phoneme_ids = phoneme_logits.argmax(dim=-1)[0].tolist()
    decoded = ctc_decode([np.array(phoneme_ids)])[0].tolist()
    text = "".join(ph_ids_to_str[idx] for idx in decoded)
    return decoded, text


def run_inference_comparison(
    audio_path: str,
    model_path: str = "./results-streaming-rnn-v3/checkpoint-46124",
    device: str = "cpu",
):
    rnn_inf = Wav2Vec2BertForRNNStreamingMultilevelCTCInference(
        model_path, device=device
    )
    tokenizer = MultiLevelTokenizer("./vocab_streaming")
    ph_ids_to_str = tokenizer.get_level_to_id_to_voab()["phonemes"]

    audio, sr = load(audio_path, sr=16000)

    # ── 1. Original streaming ────────────────────────────────────────────
    print("=" * 60)
    print("1) ORIGINAL STREAMING (inference.py __call__)")
    print("=" * 60)
    dec_stream, text_stream = run_original_streaming(rnn_inf, audio, ph_ids_to_str)
    print(f"tokens: {len(dec_stream)}")
    print(f"text: {text_stream}")

    # ── 2. Chunk-aligned streaming ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(
        "2) CHUNK-ALIGNED STREAMING (convert_input_to_chunked_for_offline + stream_inference)"
    )
    print("=" * 60)
    rnn_inf.reset()
    dec_chunk, text_chunk = run_chunk_aligned_streaming(
        rnn_inf, audio, ph_ids_to_str, device
    )
    print(f"tokens: {len(dec_chunk)}")
    print(f"text: {text_chunk}")

    # ── 3. Offline ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3) OFFLINE (stream_inference=False)")
    print("=" * 60)
    rnn_inf.reset()
    dec_offline, text_offline = run_offline(rnn_inf, audio, ph_ids_to_str, device)
    print(f"tokens: {len(dec_offline)}")
    print(f"text: {text_offline}")

    # ── Three-way comparison ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("THREE-WAY COMPARISON")
    print("=" * 60)
    print(f"{'method':<30} {'tokens':>8} {'text preview':<30}")
    print("-" * 68)
    print(f"{'original streaming':<30} {len(dec_stream):>8} {text_stream[:28]:<30}")
    print(f"{'chunk-aligned streaming':<30} {len(dec_chunk):>8} {text_chunk[:28]:<30}")
    print(f"{'offline':<30} {len(dec_offline):>8} {text_offline[:28]:<30}")
    print()
    print(f"original vs chunk-aligned tokens: {len(dec_stream)} vs {len(dec_chunk)}")
    print(f"chunk-aligned vs offline tokens:  {len(dec_chunk)} vs {len(dec_offline)}")

    return text_stream, text_chunk, text_offline


if __name__ == "__main__":
    run_inference_comparison(
        # audio_path="../quran-muaalem/assets/fatiha_long_track.wav",
        audio_path="/home/abdullah/Downloads/alif-madd.ogg",
    )
