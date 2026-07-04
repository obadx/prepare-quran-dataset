"""Count total hours in obadx/freesound-commercial-50k-noise-only"""

from datasets import load_dataset, Audio
import librosa
import io
from tqdm import tqdm

if __name__ == "__main__":
    ds = load_dataset(
        "obadx/freesound-commercial-50k-noise-only",
        split="train",
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    total_seconds = 0.0
    fil_seconds = 0.0
    for i, example in enumerate(tqdm(ds)):
        wav, _ = librosa.load(
            io.BytesIO(example["audio"]["bytes"]), sr=16000, mono=True
        )
        wav_len = len(wav) / 16000
        if wav_len < 10:
            fil_seconds += wav_len + 1
        elif wav_len > 80:
            fil_seconds += 80
        total_seconds += wav_len

    print(f"Total Filtered hours: {fil_seconds / 3600:.2f}")
    print(f"Total hours: {total_seconds / 3600:.2f}")
