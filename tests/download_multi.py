from prepare_quran_dataset.construct.utils import download_multi_file_fast
from pypdl import Pypdl
from pathlib import Path


if __name__ == '__main__':
    # url = 'https://download.quran.islamway.net/quran3/965/211/192/001.mp3'
    # url = 'https://download.quran.islamway.net/quran3/965/211/192/002.mp3'
    # url = 'https://download.quran.islamway.net/quran3/965/211/192/068.mp3'
    # out_path = download_file(
    #     url=url,
    #     out_path='./tests'
    # )
    # print(out_path)

    # url = 'https://download.quran.islamway.net/archives/212-hi.zip'
    # url = 'https://download.quran.islamway.net/quran3/965/211/192/002.mp3'
    # dl = Pypdl()
    # out = dl.start(url, file_path=Path('./data/hamo'), segments=25)
    # print(out.path)
    # url = 'https://download.quran.islamway.net/archives/212-hi.zip'
    # url = 'https://storage.googleapis.com/drive-bulk-export-anonymous/20240729T201343.771Z/4133399871716478688/993638ee-5f39-4b28-86b7-1585fcbdc45c/1/8b2db98d-1d1a-4c48-ace6-1567e6784960?authuser'
    urls = [
        'https://download.quran.islamway.net/quran3/965/211/192/001.mp3',
        'https://download.quran.islamway.net/quran3/965/211/192/110.mp3',
        'https://download.quran.islamway.net/quran3/965/211/192/111.mp3',
        'https://download.quran.islamway.net/quran3/965/211/192/112.mp3',
        # 'https://download.quran.islamway.net/archives/212.zip',
    ]
    out_path = 'tests/Downloads'
    url_to_path = download_multi_file_fast(
        urls=urls,
        out_pathes=[out_path] * len(urls),
        max_dl_workers=10,
        hash_download=False,
        num_download_segments=20,
        redownload=True,
    )
    print(url_to_path)
