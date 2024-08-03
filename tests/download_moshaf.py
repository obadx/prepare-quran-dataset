from pathlib import Path

from prepare_quran_dataset.construct.database import (
    get_files,
    download_moshaf_from_urls
)


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # Test get_files
    # ----------------------------------------------------------------------
    # pathes = [Path('data/'), Path('src')]
    # files = get_files(pathes)
    # for file in files:
    #     print(file)
    # print('Len of files', len(files))

    # ----------------------------------------------------------------------
    # Test download_moshaf
    # ----------------------------------------------------------------------
    urls = [
        'https://storage.googleapis.com/drive-bulk-export-anonymous/20240803T123812.801Z/4133399871716478688/647c412c-3306-49d5-bc8a-b8ab504391d6/1/975a2587-26d0-4c35-87c7-450e6b001b14?authuser',

        # 'https://download.quran.islamway.net/quran3/696/001.mp3',
        'https://download.quran.islamway.net/quran3/696/110.mp3',
        'https://download.quran.islamway.net/quran3/696/111.mp3',
        'https://download.quran.islamway.net/quran3/696/112.mp3',
    ]
    download_moshaf_from_urls(
        urls=urls,
        moshaf_path='data/mohaf_123',
        moshaf_name='moshaf_123',
        download_path='data/downloads',
        remove_after_download=True,
    )
