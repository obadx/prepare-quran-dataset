from pathlib import Path
from prepare_quran_dataset.construct.utils import get_audiofile_info


if __name__ == '__main__':
    filepath = Path('data/001.mp3')
    print(get_audiofile_info(filepath))
