from prepare_quran_dataset.construct.utils import download_file_slow
from pypdl import Pypdl
from pathlib import Path


if __name__ == '__main__':
    url = 'https://download.quran.islamway.net/quran3/965/211/192/001.mp3'
    # url = 'https://download.quran.islamway.net/quran3/965/211/192/002.mp3'
    # url = 'https://download.quran.islamway.net/quran3/965/211/192/068.mp3'
    # out_path = download_file(
    #     url=url,
    #     out_path='./tests'
    # )
    # print(out_path)

    dl = Pypdl()
    out = dl.start(url, file_path=Path('./tests/'))
    print(out.path)
