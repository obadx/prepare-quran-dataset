import asyncio

from librosa.core import load

from prepare_quran_dataset.annotate.tarteel import tarteel_transcript, merge_transcripts


if __name__ == "__main__":
    # Transcribe audio
    file_path = "../002007.mp3"
    wave, sr = load(file_path, sr=16000)

    trans = asyncio.run(
        tarteel_transcript(wave, chunck_overlap_sec=10, timeout_sec=100)
    )
    print(trans)

    merged_trans = merge_transcripts(trans)
    print(merged_trans)
