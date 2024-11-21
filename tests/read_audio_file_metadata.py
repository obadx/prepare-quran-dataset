from pathlib import Path
import librosa
import magic

from prepare_quran_dataset.construct.utils import get_audiofile_info


def get_audio_metadata_librsoa(filepath):
    # sr=None to preserve original sample rate
    y, sr = librosa.core.load(filepath, sr=None)
    duration = librosa.core.get_duration(y=y, sr=sr)
    return duration, sr


if __name__ == '__main__':
    # mp3_pathes = Path('/home/abdullah/Downloads').glob('*.mp3')
    mp3_pathes = [
        Path('/home/abdullah/Downloads/128.adts'),
        # Path('/home/abdullah/Downloads/001.adts'),
        # Path('/home/abdullah/Downloads/001_To_002_25.adts'),
    ]
    for p in sorted(mp3_pathes):
        info = get_audio_metadata_librsoa(p)
        print(get_audiofile_info(p))
        print(p.name)
        print(info)
        print(magic.from_file(p, mime=True))
        print('-' * 30)
