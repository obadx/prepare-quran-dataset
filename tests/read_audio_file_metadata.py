from pathlib import Path
import traceback
import json

from prepare_quran_dataset.construct.utils import get_audiofile_info


if __name__ == '__main__':
    mp3_pathes = Path(
        '/home/abdullah/Documents/courses/quran-dataset/Downloads/0.0').glob('*.mp3')
    path_to_error = {}
    for p in mp3_pathes:
        try:
            info = get_audiofile_info(p)
        except Exception as e:
            path_to_error[str(p)] = traceback.format_exc()

    print('Errors: ')
    print(json.dumps(path_to_error))
