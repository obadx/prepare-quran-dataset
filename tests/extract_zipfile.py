import shutil
from pathlib import Path
import os

from prepare_quran_dataset.construct.utils import extract_zipfile

if __name__ == '__main__':
    # big zipfile
    home = Path(os.getenv('HOME'))

    extract_path = home / 'Downloads'
    extract_zipfile(zipfile_path=home / 'Downloads/prepare-quran-dataset-main.zip',
                    extract_path=extract_path, num_workers=12)

    # # empy zipfil2
    # extract_path = 'data/out_test'
    # extract_zipfile(zipfile_path='data/test.zip',
    #                 extract_path=extract_path, num_workers=12)

    # shutil.rmtree(extract_path)
