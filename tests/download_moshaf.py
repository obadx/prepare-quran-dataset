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
        'https://storage.googleapis.com/drive-bulk-export-anonymous/20240802T211752.944Z/4133399871716478688/24f18390-7ef1-41b7-8020-16deecd29b4a/1/835a22ad-83c9-49b4-ace0-bd186fe72a99?authuser',

        'https://download.quran.islamway.net/quran3/696/001.mp3',
        'https://download.quran.islamway.net/quran3/696/110.mp3',
        'https://download.quran.islamway.net/quran3/696/111.mp3',
        'https://download.quran.islamway.net/quran3/696/112.mp3',
    ]
    download_moshaf_from_urls(
        urls=urls,
        moshaf_path='data/mohaf_123',
        download_path='data/downloads')
