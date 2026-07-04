"""Mix first muaalem sample with a noise sample using AddBackgroundNoise"""
from datasets import load_dataset, Audio
from audiomentations import AddBackgroundNoise
import librosa
import soundfile as sf
import io
import tempfile

if __name__ == "__main__":
    # 1. Load first muaalem sample
    ds = load_dataset(
        "obadx/muaalem-annotated-v3",
        name="moshaf_6.0",
        split="train",
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    audio_muaalem = ds[0]["audio"]
    wav, sr = librosa.load(io.BytesIO(audio_muaalem["bytes"]), sr=16000, mono=True)

    # 2. Load noise sample at index 1
    noise_ds = load_dataset("obadx/freesound-commercial-50k-noise-only", split="train")
    noise_ds = noise_ds.cast_column("audio", Audio(decode=False))
    audio_noise = noise_ds[1]["audio"]
    noise_wav, _ = librosa.load(io.BytesIO(audio_noise["bytes"]), sr=16000, mono=True)

    # 3. Save noise to temp file (AddBackgroundNoise needs file paths)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        sf.write(tmp_path, noise_wav, 16000)

    # 4. Mix with AddBackgroundNoise
    augment = AddBackgroundNoise(sounds_path=[tmp_path], p=1.0)
    mixed = augment(wav, sample_rate=16000)

    # 5. Save mixed sample
    sf.write("mixed_sample.wav", mixed, 16000)
    print("Saved mixed_sample.wav")
