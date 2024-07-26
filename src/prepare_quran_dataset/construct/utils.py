from tqdm import tqdm
import requests
from pathlib import Path
import os


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
