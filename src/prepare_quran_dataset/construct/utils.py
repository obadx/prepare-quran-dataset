from tqdm import tqdm
import requests
from pathlib import Path
import os
import functools
import time
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def exception_catcher(func):
    """Decorator Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_catcher(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e
    return wrapper_catcher


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
