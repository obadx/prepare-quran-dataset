import requests
from pathlib import Path
import os
import functools
import time
from zipfile import ZipFile, is_zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse


from tqdm import tqdm
from pypdl import Pypdl
from mutagen import File


class DownloadError(Exception):
    ...


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
    num_download_segments=20,
    num_unzip_workers=12,
    remove_zipfile=True,
) -> Path:
    """Downloads a file and extract if if it is zipfile
    Args:
        out_path (str | Path): the path to the Download (Directory)
        extract_zip (bool): if true extract a zip file to `out_path`
        remove_zipfile (bool): remove zipfile after downloading it
    """
    out_path = Path(out_path)
    assert not out_path.is_file(), (
        'Download Path `out_path` has to be a directory not a file')
    os.makedirs(out_path, exist_ok=True)
    filename = deduce_filename(url)
    out_path /= filename
    if out_path.exists():
        return out_path

    dl = Pypdl()
    out = dl.start(url, file_path=out_path, segments=num_download_segments)
    if out is None:
        raise DownloadError
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

    # Make a HEAD request to get headers
    try:
        response = requests.head(url, allow_redirects=True)

        # Check for Content-Disposition header
        if 'content-disposition' in response.headers:
            content_disposition = response.headers['content-disposition']
            filename = content_disposition.split('filename=')[1].strip('"')
    except Exception as e:
        print(f'Error connecting to the url: {e}')

    return filename


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
