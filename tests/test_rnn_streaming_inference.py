import torch
from librosa import load
import numpy as np

from prepare_quran_dataset.modeling_streaming_rnn.inference import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCInference,
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


def run_inference_comparison(
    audio_path: str,
    model_path: str = "./results-streaming-rnn-v2/checkpoint-35480",
    device: str = "cpu",
):
    rnn_inf = Wav2Vec2BertForRNNStreamingMultilevelCTCInference(
        model_path, device=device
    )
    tokenizer = MultiLevelTokenizer("./vocab_streaming")
    ph_ids_to_str = tokenizer.get_level_to_id_to_voab()["phonemes"]

    audio, sr = load(audio_path, sr=16000)

    # ── Streaming Inference ──────────────────────────────────────────────
    print("=" * 60)
    print("STREAMING INFERENCE")
    print("=" * 60)

    chunk_idx = 0
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
            print(f"chunk {chunk_idx:02d} phonemes_ids: {output.phonemes_ids}")
        consumed += n
        chunk_idx += 1

    decoded_streaming = ctc_decode([np.array(out_ids)])[0].tolist()
    text_streaming = "".join(ph_ids_to_str[idx] for idx in decoded_streaming)
    print(f"\nstreaming decoded IDs: {decoded_streaming}")
    print(f"streaming text: {text_streaming}")

    # ── Offline Inference ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OFFLINE INFERENCE")
    print("=" * 60)

    rnn_inf.reset()
    inputs = rnn_inf.processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device, dtype=rnn_inf.dtype) for k, v in inputs.items()}

    outputs = rnn_inf.model(**inputs, stream_inference=False, return_dict=True)
    phoneme_logits = outputs.logits["phonemes"]  # (1, T, vocab_size)
    phoneme_ids_offline = phoneme_logits.argmax(dim=-1)[0].tolist()

    decoded_offline = ctc_decode([np.array(phoneme_ids_offline)])[0].tolist()
    text_offline = "".join(ph_ids_to_str[idx] for idx in decoded_offline)

    print(f"offline raw IDs (first 200): {phoneme_ids_offline[:200]}")
    print(f"offline decoded IDs: {decoded_offline}")
    print(f"offline text: {text_offline}")

    # ── Comparison ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"streaming: {text_streaming}")
    print(f"offline:   {text_offline}")
    print(f"streaming tokens: {len(decoded_streaming)}")
    print(f"offline tokens:   {len(decoded_offline)}")

    return text_streaming, text_offline


if __name__ == "__main__":
    run_inference_comparison(
        # audio_path="../quran-muaalem/assets/fatiha_long_track.wav",
        audio_path="/home/abdullah/Downloads/alif-madd.ogg",
    )
