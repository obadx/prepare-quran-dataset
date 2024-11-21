import requests
import magic
from pathlib import Path
import json
import mimetypes
import filetype

from prepare_quran_dataset.construct.utils import deduce_filename

if __name__ == '__main__':
    urls = [
        'https://cdns1.zekr.online/quran/5210/111/64.mp3',
        'https://cdns1.zekr.online/quran/10341/1/128.mp3',
        'https://zekr.online/ar/single/Download/Sura/263038?title=%D8%A7%D9%84%D9%86%D8%A7%D8%B3%20-%20%D8%A3%D8%AD%D9%85%D8%AF%20%D9%85%D8%AD%D9%85%D8%AF%20%D8%B9%D8%A7%D9%85%D8%B1',
        'https://codeload.github.com/Abdullahaml1/prepare-quran-dataset/zip/refs/heads/main',
        'https://download.quran.islamway.net/archives/212-hi.zip',
        'https://archive.org/download/Musshaf-Mujawwed-Kamel-Yousef-Al-Buthimi-High-Quality/002_105-123.ogg',
        'https://archive.org/download/Musshaf-Mujawwed-Kamel-Yousef-Al-Buthimi-High-Quality/002_189-208.mp3',
        'https://download.quran.islamway.net/quran3/316/011.mp3',
        'https://dynamic-media-cdn.tripadvisor.com/media/photo-o/0e/72/44/3b/caption.jpg?w=1200&h=1200&s=1',
        'https://www.pexels.com/download/video/3309011/',
        'https://videos.pexels.com/video-files/3309011/3309011-hd_1920_1080_30fps.mp4',
        'https://www.pexels.com/download/video/33239809011/',

    ]

    for url in urls:
        print('url', url)
        print(deduce_filename(url))

        print('\n' * 3)

    # print('\n\n\n\n')
    # p = Path('/home/abdullah/Downloads/128.aac')
    # print(p.name)
    # print(filetype.guess_extension(p))
    # print(filetype.guess_mime(p))
