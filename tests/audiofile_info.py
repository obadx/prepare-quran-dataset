from pathlib import Path
from prepare_quran_dataset.construct.utils import get_audiofile_info


if __name__ == '__main__':
    filepath = Path('/home/abdullah/Downloads/128.mp3')
    print(get_audiofile_info(filepath))
