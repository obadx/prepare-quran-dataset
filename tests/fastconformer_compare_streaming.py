"""Compare offline vs cache-aware streaming inference on a single audio file.

Runs the FastConformer multi-level CTC model twice on the same test file:

1.  **Offline** — the full utterance in one forward pass (``cache=None``).
2.  **Streaming** — chunk-by-chunk cache-aware inference emulating real-time
    audio via ``infer_fastconformer_streaming``.

For every level the predicted label sequences are aligned with Levenshtein
distance and a matching ratio is computed *with respect to the offline
output* (same convention as PER in ``train_streaming.py``, inverted):
``matching_ratio = 1 - min(distance, len(offline)) / len(offline)``.
The decoded phonemes of both paths are printed side by side.
"""

import json

import Levenshtein
import librosa
import torch

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTC,
    FastConformerMelProcessor,
    infer_fastconformer_streaming,
)
from prepare_quran_dataset.modeling_fastconformer_cache_aware.multi_level_tokenizer import (
    MultiLevelTokenizer,
)

# ckpt_dir = "./results-fastconformer-v6/checkpoint-113560"
# ckpt_dir = "./results-fastconformer-c5-v4/checkpoint-113560"
ckpt_dir = "./results-fastconformer-base-v1/checkpoint-2180265/"
# audio_path = "./assets/audio-sampels/test_sample_3.mp3"
audio_path = "./assets/audio-sampels/test_sample_4.ogg"
vocab_dir = "./vocab_streaming"


def ids_to_chars(ids: list[int]) -> str:
    """Map label ids to distinct ASCII chars so Levenshtein can align them."""
    return "".join(chr(ord("a") + t) for t in ids)


def matching_ratio(stream_ids: list[int], offline_ids: list[int]) -> float:
    """Matching ratio of the streaming labels w.r.t. the offline labels."""
    if not offline_ids:
        return 1.0 if not stream_ids else 0.0
    distance = min(
        Levenshtein.distance(ids_to_chars(stream_ids), ids_to_chars(offline_ids)),
        len(offline_ids),
    )
    return 1.0 - distance / len(offline_ids)


if __name__ == "__main__":
    print(ckpt_dir)
    # ---- 1. Setup ----
    tokenizer = MultiLevelTokenizer(vocab_dir)
    device = "cpu"
    dtype = torch.float32

    model = FastConformerCacheAwareMultilevelCTC.from_pretrained(ckpt_dir)
    model.to(device=device, dtype=dtype)
    model.eval()
    # The mel processor is not an nn.Module, so model.to() skips it.
    model.processor.to(device)
    processor = FastConformerMelProcessor(**model.config.processor_kwargs)

    wav, _ = librosa.load(audio_path, sr=16000, mono=True)

    # ---- 2. Offline inference (reference) ----
    wav_tensor = torch.tensor([wav], dtype=torch.float32, device=device)
    wav_lengths = torch.tensor([len(wav)], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(raw_audio=wav_tensor, audio_length=wav_lengths)
    offline_labels: dict[str, list[int]] = {}
    for level, logits in out.logits.items():
        n_frames = int(out.encoder_lengths[0])
        ids = logits[0, :n_frames].argmax(dim=-1).tolist()
        offline_labels[level] = ids

    # ---- 3. Streaming inference ----
    streaming_logits = infer_fastconformer_streaming(
        [audio_path],
        device,
        dtype,
        model,
        processor,
    )
    streaming_labels: dict[str, list[int]] = {}
    for level, logits in streaming_logits.items():
        n_frames = int(logits.shape[1])
        ids = logits[0, :n_frames].argmax(dim=-1).tolist()
        streaming_labels[level] = ids

    # ---- 4. Alignment + matching ratio per level ----
    print("=" * 70)
    ratios = []
    for level in offline_labels:
        off = offline_labels[level]
        strm = streaming_labels.get(level, [])
        ratio = matching_ratio(strm, off)
        distance = len(off) - round(ratio * len(off))
        ratios.append(ratio)
        print(
            f"[{level}] offline={len(off):4d} tokens | "
            f"streaming={len(strm):4d} tokens | edits={distance:3d} | "
            f"matching_ratio={ratio:.4f}"
        )
    print(f"average matching_ratio: {sum(ratios) / len(ratios):.4f}")

    # ---- 5. Decoded phonemes for both paths ----
    decoded_offline = tokenizer.decode({"phonemes": [offline_labels["phonemes"]]})
    decoded_streaming = tokenizer.decode({"phonemes": [streaming_labels["phonemes"]]})
    print("=" * 70)
    print(
        "Offline phonemes:  ",
        json.dumps(decoded_offline["phonemes"][0], ensure_ascii=False),
    )
    print(
        "Streaming phonemes:",
        json.dumps(decoded_streaming["phonemes"][0], ensure_ascii=False),
    )
