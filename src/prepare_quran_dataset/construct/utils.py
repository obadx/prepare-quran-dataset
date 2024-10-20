import requests
from pathlib import Path
import os
import functools
import time
from zipfile import ZipFile, is_zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse
import json
import re
from typing import Any
import urllib
from hashlib import sha256


from tqdm import tqdm
from pypdl import Pypdl
from mutagen import File
from bs4 import BeautifulSoup
from quran_transcript.utils import normalize_aya

DATA_PATH = Path(__file__).parent.parent / 'data'


def extract_suar_from_mp3quran(url):
    """Extract sepecific Moshaf from https://mp3quran.net/ as:
    {
        "sura_id": "sura_link",
    }
    Example:
    {
        "108": "https://server10.mp3quran.net/download/Aamer/108.mp3",
        "109": "https://server10.mp3quran.net/download/Aamer/109.mp3",
        "110": "https://server10.mp3quran.net/download/Aamer/110.mp3",
        "111": "https://server10.mp3quran.net/download/Aamer/111.mp3",
        "112": "https://server10.mp3quran.net/download/Aamer/112.mp3",
        "113": "https://server10.mp3quran.net/download/Aamer/113.mp3",
        "114": "https://server10.mp3quran.net/download/Aamer/114.mp3"
    }

        Args:
            url (str): the url of the moshaf example:
                https://mp3quran.net/ar/Aamer
    """
    # Send a GET request to the provided URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to load page: {
                        response.status_code}, url={url}")

    soup = BeautifulSoup(response.content, 'html.parser')

    sura_links = {}

    # Find all the list items that contain both data-url and an image tag
    sura_items = soup.find_all(
        'div', class_='sora-item')

    for item in sura_items:
        # find_all instead of find_next because find_all searches within
        # the same tag unlike find_next is searches for items on the whole soup
        sura_num_item = item.find_all('div', class_='sora-num')
        if sura_num_item:
            assert len(sura_num_item) == 1, (
                f'Found more than sura number for the item:\n{item}'
                f'\nResults: {sura_num_item}\nURL: {url}'
            )
            sura_index = sura_num_item[0].string

            sura_url = item.find_all(
                'a', class_=["sora-bt", "download-btn"], attrs={'href': True})
            if sura_url:
                assert len(sura_url) == 1, (
                    f'Found more than sura url for the item:\n{item}'
                    f'\nResults: {sura_url}\nURL: {url}'
                )
                sura_links[sura_index] = sura_url[0]['href']

    return sura_links


def extract_sura_from_zekr(url):
    """Extract sepecific Moshaf from https://zekr.online/ as:
    {
        "sura_id": "sura_link",
    }
    Example:
        {
            "109": "https://cdns1.zekr.online/quran/5403/109/32.mp3",
            "110": "https://cdns1.zekr.online/quran/5403/110/32.mp3",
            "111": "https://cdns1.zekr.online/quran/5403/111/32.mp3",
            "112": "https://cdns1.zekr.online/quran/5403/112/32.mp3",
            "113": "https://cdns1.zekr.online/quran/5403/113/32.mp3",
            "114": "https://cdns1.zekr.online/quran/5403/114/32.mp3"
        }

        Args:
            url (str): the url of the moshaf example:
                https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403
    """
    # Send a GET request to the provided URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to load page: {
                        response.status_code}, url={url}")

    soup = BeautifulSoup(response.content, 'html.parser')

    sura_links = {}

    # Find all the list items that contain both data-url and an image tag
    sura_items = soup.find_all(
        'li', class_='sura-item', attrs={'data-url': True, 'data-image': True})

    for item in sura_items:
        # Extract the data-url attribute for the audio link
        audio_link = item['data-url']

        # Extract the sura index from the image URL
        img_src = item['data-image']
        sura_index = img_src.split('/')[-1].replace('.svg', '')

        sura_links[f'{int(sura_index):0{3}}'] = audio_link

    return sura_links


class DownloadError(Exception):
    ...


def load_jsonl(filepath: str | Path) -> list[Any]:
    """Loads a Json file data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: list[Any], filepath: str | Path) -> None:
    """Saves list[Any] data into a `filepath` ins JSON line format"""
    data_str = ""
    for item in data:
        data_str += json.dumps(item, ensure_ascii=False) + '\n'
    if data_str:
        data_str = data_str[:-1]  # removes '\n' from last line

    with open(filepath, 'w+', encoding='utf-8') as f:
        f.write(data_str)


def timer(func):
    """Decorator Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__}() in {run_time:.4f} secs")

        return value
    return wrapper_timer


def normalize_text(text: str) -> str:
    out_text = re.sub(r'-|_', '', text)
    out_text = re.sub("أ|إ|آ", "ا", out_text)
    return normalize_aya(
        out_text,
        remove_spaces=True,
        remove_tashkeel=True,
        remove_small_alef=True)


def get_suar_list(suar_path=DATA_PATH / 'suar_list.json') -> list[str]:
    """Return the suar names of the Holy Quran in an ordered list
    """
    with open(suar_path, 'r', encoding='utf8') as f:
        suar_list = json.load(f)
    return suar_list


@dataclass
class AudioFileInfo:
    sample_rate: int
    duration_seconds: float


def get_audiofile_info(audiofile_path: str | Path) -> AudioFileInfo:
    """Reads the file metadata and return its information
    Returns:
        (AudioFileInfo): if the audiofile is not valid return None
    """
    audio = File(audiofile_path)
    if audio is None:
        return None
    return AudioFileInfo(
        sample_rate=audio.info.sample_rate,
        duration_seconds=audio.info.length)


def download_file_fast(
    url: str,
    out_path: str | Path,
    extract_zip=True,
    hash_download=False,
    num_download_segments=20,
    num_unzip_workers=12,
    remove_zipfile=True,
    redownload=False,
) -> Path:
    """Downloads a file and extract if if it is zipfile
    Args:
        out_path (str | Path): the path to the Download (Directory)
        extract_zip (bool): if true extract a zip file to `out_path`
        remove_zipfile (bool): remove zipfile after downloading it
        redownload (bool): redownload the file if it exists
        hash_download (bool): if True will the file name will be the hash(url).
            if False it will deduce the file name from the url like "001.mp3"
    """
    out_path = Path(out_path)
    assert not out_path.is_file(), (
        'Download Path `out_path` has to be a directory not a file')
    os.makedirs(out_path, exist_ok=True)

    filename = deduce_filename(url)
    if hash_download:
        splits = filename.split('.')
        if len(splits) == 1:
            raise ValueError(f'The file has to extention for url: {url}')
        ext = splits[-1]
        filename = sha256(url.encode()).hexdigest() + f'.{ext}'

    out_path /= filename
    if out_path.exists() and not redownload:
        return out_path

    dl = Pypdl()
    out = dl.start(url, file_path=out_path, segments=num_download_segments)
    if out is None:
        raise DownloadError(f'Error while downloading or url: {url}')
    out_path = Path(out.path)

    if is_zipfile(out_path):
        zipfile_path = out_path.rename(
            out_path.parent / f'{out_path.name}.download')
        extract_zipfile(zipfile_path=zipfile_path,
                        extract_path=out_path, num_workers=num_unzip_workers)

        # remove zipfile
        if remove_zipfile:
            zipfile_path.unlink()

    return out_path


def deduce_filename(url) -> str:
    """extracts file name from url"""
    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = parsed_url.path.split('/')[-1]

    # paterns to get the filename from http header
    # The priority is for the Arabic name i.e paatterns[0]
    # then ofr the English name i.e: patterns[1]
    patterns = [r'filename\*=utf-8\'\'(.*)$', r'filename="?([^$]+)"?$']

    # Make a HEAD request to get headers
    try:
        response = requests.head(url, allow_redirects=True)

        # Check for Content-Disposition header
        if 'content-disposition' in response.headers:
            content_disposition = response.headers['content-disposition']

            parts = content_disposition.split(';')
            for pattern in patterns:
                for part in parts:
                    match = re.search(pattern, part)
                    if match:
                        filename = match.group(1)
                        if filename.endswith('"'):
                            filename = filename[:-1]
                        return urllib.parse.unquote(filename)

    except Exception as e:
        print(f'Error connecting to the url: {e}')

    return urllib.parse.unquote(filename)


def download_file_slow(
    url: str,
    out_path: str | Path,
    filename: str = None,
    extrct_zip=True,
    block_size=8192,
) -> Path:
    """Download a file with and the ability to resume downloading
    Args:
        url (str): the link to the file
        out_path (str | Path): the direcotry to store and extract the file
        filename (str): the file name. If None we will extract the name from the url
        extract_zip (bool): extract zip file

    Sources:
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    https://realpython.com/python-download-file-from-url/
    """

    # process file name
    out_path = Path(out_path)
    if filename is None:
        filename = url.split('/')[-1]
    filepath = out_path / filename

    # Check if file already exists
    resume_header = {}
    initial_size = 0
    if filepath.is_file():
        # Get the size of the existing file
        initial_size = os.path.getsize(filepath)
        resume_header = {'Range': f'bytes={initial_size}-'}

    # Make a request with the Range header
    # Streaming, so we can iterate over the response.
    response = requests.get(url, headers=resume_header, stream=True)

    # Streaming, so we can iterate over the response.
    # response = requests.get(url, stream=True)

    if response.status_code == 416:
        print("Download already complete.")
        return filepath

    elif response.status_code in (200, 206):
        # Sizes in bytes.
        total_size = int(response.headers.get(
            "content-length", 0)) + initial_size

        with tqdm(
                total=total_size,
                initial=initial_size,
                unit="iB",
                unit_scale=True) as progress_bar:

            mode = 'ab' if 'Range' in resume_header else 'wb'
            with open(filepath, mode) as file:
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    if chunk:
                        file.write(chunk)

        if total_size != 0 and os.path.getsize(filepath) != total_size:
            print(os.path.getsize(filepath))
            print(total_size)
            print(initial_size)
            raise RuntimeError("Could not download file")
        return filepath
    else:
        raise RuntimeError(f"Failed to download file: {response.status_code}")


# @exception_catcher
def unzip_files(
    zipfile_path: str | Path,
    files: list,
    extract_path: Path
):
    """unzip part of files (files) in extract_path
    """
    # open the zip file
    with ZipFile(zipfile_path, 'r') as handle:
        # unzip multiple files
        for file in files:
            # unzip the file
            handle.extract(file, extract_path)
            # report progress
            # print(f'.unzipped {file}')


def extract_zipfile(
    zipfile_path: str | Path,
    extract_path: str | Path,
    num_workers: int = 8,
):
    # WARN: we ingore empty files in zip file
    """Extract zipfiles using multiple processes eachone is working on group of files
    source: https://superfastpython.com/multithreaded-unzip-files/
    Args:
        zipfile_path (str | Path): path to zipfile
        extract_path (str | Path): path to extract zipfile
        num_worker (int): number of worker to process zipfile in parallel

    """
    extract_path = Path(extract_path)
    # open the zip file
    files = []
    with ZipFile(zipfile_path, 'r') as handle:
        # list of all files to unzip
        # files = handle.namelist()
        for zip_info in handle.infolist():
            if not zip_info.is_dir():
                files.append(zip_info.filename)

    # creating directory structure of the unziped file
    dirs_to_make = set()
    for file in files:
        dirs_to_make.add(extract_path / Path(file).parent)
    for dir in dirs_to_make:
        os.makedirs(dir, exist_ok=True)

    # determine chunksize
    chunksize = max(len(files) // num_workers, 1)
    # start the thread pool
    with ProcessPoolExecutor(num_workers) as exe:
        # split the copy operations into chunks
        for i in range(0, len(files), chunksize):
            # select a chunk of filenames
            selected_files = files[i:(i + chunksize)]
            # submit the batch copy task
            feature = exe.submit(unzip_files,
                                 zipfile_path, selected_files, extract_path)

    # executing this to account for errors
    feature.result()
