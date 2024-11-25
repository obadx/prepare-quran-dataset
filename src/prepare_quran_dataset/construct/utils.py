import requests
from pathlib import Path
import os
import functools
import time
from zipfile import ZipFile, is_zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import urlparse
import json
import re
from typing import Any
import urllib
from hashlib import sha256
import signal


from tqdm import tqdm
from pypdl import Pypdl
from mutagen import File
from bs4 import BeautifulSoup
import filetype
from quran_transcript.utils import normalize_aya

DATA_PATH = Path(__file__).parent.parent / 'data'


def extract_suar_from_archive(
        url: str, prefared_types: list[str] = ['mp3', 'oog']
) -> list[str]:
    """Extract sepecific Moshaf from https://mp3quran.net/ as list of Audio links:

        Args:
            url (str): the url of the moshaf example:
                https://archive.org/
            prefared_list: (list[str]): list of prefresh audio file types.
                default= ['mp3', 'ogg'] if the media type is not in the prefared list
                it will be returned

        Returns:
            list[str]: list of media links. Example:
            [
            ]
    """
    def get_media_type(url: str) -> str:
        return url.split('/')[-1].split('.')[-1]

    # Send a GET request to the provided URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(
            f"Failed to load page: {response.status_code}, url={url}")

    soup = BeautifulSoup(response.content, 'html.parser')

    suar_links: list[str] = []

    # Find all the list items that contain both data-url and an image tag
    sura_items = soup.find_all(
        'div', attrs={'itemprop': 'hasPart', 'itemscope': True, 'itemtype': 'http://schema.org/AudioObject'})

    for item in sura_items:
        # find_all instead of find_next because find_all searches within
        # the same tag unlike find_next is searches for items on the whole soup
        track_url = None
        media_items = item.find_all(
            'link', attrs={'itemprop': 'associatedMedia', 'href': True})
        type_to_url = {get_media_type(m['href']): m['href']
                       for m in media_items}
        assert type_to_url != {}, 'There has to be media types'

        for prefred_t in prefared_types:
            if prefred_t in type_to_url:
                track_url = type_to_url[prefred_t]
                break

        # if the track_type is not in the prefred_type
        # then get the first media type
        if track_url is None:
            track_url = list(type_to_url.values())[0]

        suar_links.append(track_url)

    return suar_links


def extract_suar_from_mp3quran(url) -> dict[str, str]:
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
        raise Exception(
            f"Failed to load page: {response.status_code}, url={url}")

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


def extract_sura_from_zekr(url) -> dict[str, str]:
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
        raise Exception(
            f"Failed to load page: {response.status_code}, url={url}")

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


def kill_process(pid):
    os.kill(pid, signal.SIGTERM)  # or signal.SIGKILL for force killing
    print(f"Process {pid} has been terminated.")


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


def download_multi_file_fast(
    urls: list[str],
    out_pathes: list[str | Path],
    max_dl_workers=10,
    extract_zip=True,
    hash_download=False,
    num_download_segments=20,
    num_unzip_workers=12,
    remove_zipfile=True,
    redownload=False,
    show_progress=True,
    max_retries=0,
) -> dict[str, Path]:
    """ same as `download_file_fast` but with multithreading
    Args:
        max_dl_workers (int): the max number of threads to run downloads

    Return:
        dict[str, Path]: {url: the output download path}
    """
    if len(urls) == 0:
        return []
    max_workers = min(len(urls), max_dl_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_to_url = {}
        for url, out_path in zip(urls, out_pathes):
            future = exe.submit(
                download_file_fast,
                url,
                out_path,
                extract_zip=extract_zip,
                hash_download=hash_download,
                num_download_segments=num_download_segments,
                num_unzip_workers=num_unzip_workers,
                remove_zipfile=remove_zipfile,
                redownload=redownload,
                show_progress=show_progress,
                max_retries=max_retries,
            )
            future_to_url[future] = url

        # Collect results as they complete
        url_to_path = {}
        for future in as_completed(future_to_url.keys()):
            url = future_to_url[future]
            url_to_path[url] = future.result()
        return url_to_path


def download_file_fast(
    url: str,
    out_path: str | Path,
    extract_zip=True,
    hash_download=False,
    num_download_segments=20,
    num_unzip_workers=12,
    remove_zipfile=True,
    redownload=False,
    show_progress=True,
    max_retries: int = 0,
) -> Path:
    """Downloads a file and extract if if it is zipfile
    Args:
        out_path (str | Path): the path to the Download (Directory)
        extract_zip (bool): if true extract a zip file to `out_path`
        remove_zipfile (bool): remove zipfile after downloading it
        redownload (bool): redownload the file if it exists
        hash_download (bool): if True the file name will be the hash(url).
            if False it will deduce the file name from the url like "001.mp3"
        max_retries (int): the number of times to try download if an error occured
            default is 0
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
    out = dl.start(
        url,
        file_path=out_path,
        segments=num_download_segments,
        display=show_progress,
        retries=max_retries,
    )
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


def deduce_extention_from_url(url: str) -> str | None:
    """Tries to guess the extention of a media file from its url
    by downloading the first 2 bytes

    Returns:
        str if we successfully extracts extention else `None`
    Raises:
        TypeError: if obj is not a supported type.
    """
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    # Read the first 2 KB of the file
    first_bytes = response.raw.read(2048)
    return filetype.guess_extension(first_bytes)


def deduce_filename(url, verbose=False) -> str:
    """extracts filename from url
    * if the url in reachable:
        1. take the last redirected url
        2. extract file name from the last url
        3. extract extention using `filetype` package
    * if there is not internet or the url in not reachable:
        - it returns the filename the url
    """

    try:
        filename = None
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        # Read the first 2 KB of the file
        first_bytes = response.raw.read(2048)

        # Check for Content-Disposition header
        if 'content-disposition' in response.headers:
            filename = get_filename_from_header(
                response.headers['content-disposition'])
        if filename is None:
            # get filename from the redirected url
            filename = get_filename_from_url(response.url)
            if verbose:
                print(f'Faild to read header {response.url}, {filename}')
        else:
            if verbose:
                print(
                    f'Success in reading Header, header filename: {filename}')

        try:
            # trying to guess extention form first_bytes
            segs = filename.split('.')
            # old_ext = segs[-1]
            name = ''.join(segs[:-1]) if len(segs) > 1 else filename
            ext = filetype.guess_extension(first_bytes)
            if ext:
                if verbose:
                    print('Success in reading mime type')
                return name + '.' + ext
            else:
                return filename
        except Exception as e:
            if verbose:
                print(f'Error {e} while extracting file exctention for {url}')
            return filename

    except Exception as e:
        print(f'Error {e} connecting to {url}')
        return get_filename_from_url(url)


def get_filename_from_header(content_disposition) -> str | None:
    """ Gets the filename from content-disposition in GET request header
    Args:
        contnet_disposition: The respose.headers['content-disposition']

    Return:
        str | None: the filename in the header if found else `None`
    """
    # paterns to get the filename from http header
    # The priority is for the Arabic name i.e paatterns[0]
    # then ofr the English name i.e: patterns[1]
    patterns = [r'filename\*=utf-8\'\'(.*)$', r'filename="?([^$]+)"?$']
    parts = content_disposition.split(';')
    for pattern in patterns:
        for part in parts:
            match = re.search(pattern, part)
            if match:
                filename = match.group(1)
                if filename.endswith('"'):
                    filename = filename[:-1]
                return urllib.parse.unquote(filename)
    return None


def get_filename_from_url(url_link) -> str:
    """Extract filename from URL"""
    segments = urlparse(url_link).path.split('/')
    for name in segments[::-1]:
        if name:
            return urllib.parse.unquote(name)


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
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        # split the copy operations into chunks
        for i in range(0, len(files), chunksize):
            # select a chunk of filenames
            selected_files = files[i:(i + chunksize)]
            # submit the batch copy task
            feature = exe.submit(unzip_files,
                                 zipfile_path, selected_files, extract_path)

    # executing this to account for errors
    feature.result()
