import re
from typing import Any
from pathlib import Path
import shutil
import os

from prepare_quran_dataset.construct.base import Pool
from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf
from prepare_quran_dataset.construct.utils import download_file_fast, get_audiofile_info


class ReciterPool(Pool):
    """Pool or Reciter see Pool to understand structure"""

    def __init__(self, path='data/reciter_pool.jsonl'):
        super().__init__(
            path=path,
            item_type=Reciter,
            id_column='id',
        )

    def get_hash(self, item: dict[str, Any] | Reciter) -> str:
        if isinstance(item, Reciter):
            item = item.model_dump()
        clean_text = re.sub(r'\s+', '', item['arabic_name'])
        return f'{clean_text}_{item["country_code"]}'

    def generate_id(self, new_item: Reciter) -> int:
        return self.__len__()


class MoshafPool(Pool):
    """Pool or Moshaf see Pool to understand structure"""

    def __init__(
        self,
        reciter_pool: ReciterPool,
        dataset_path='data/',
        metadata_path='data/moshaf_pool.jsonl'
    ):
        super().__init__(
            path=metadata_path,
            item_type=Moshaf,
            id_column='id',
        )
        self._reciter_pool = reciter_pool
        self.dataset_path = Path(dataset_path)

    def get_hash(self, item: dict[str, Any] | Moshaf) -> str:
        """We will use url and reciter's ID as a unique Identifier"""
        if isinstance(item, dict):
            item = Moshaf(**item)
        urls_text = ''
        for url in item.urls:
            urls_text += url + '_'
        return f'{item.reciter_id}_{urls_text}'

    def generate_id(self, new_item: Moshaf) -> int:
        """The id is "{reciter_id}.{moshaf_id}" example (0.1)
        Reciter ID = 0
        Moshaf number 1 of Reciter(0)
        """
        return len(self._reciter_pool[new_item.reciter_id].moshaf_ids)

    def process_new_item_before_insert(self, new_item: Moshaf):
        new_item = new_item.copy(deep=True)


def download_media_and_fill_metadata(item: Moshaf, database_path: Path | str) -> Moshaf:
    """

    Downlaoad audio files from url, gather them in a single directory, and fill metadata of the moshaf

    Returns:
        (Moshaf): the moshaf filled with metadata
    """
    item = item.copy(deep=True)
    database_path = Path(database_path)
    moshaf_path = database_path / item.id
    os.makedirs(moshaf_path, exist_ok=False)
    download_moshaf_from_urls(urls=item.sources, moshaf_path=moshaf_path)

    # Fill Moshaf's Metadata


def download_moshaf_from_urls(
        urls: list[str], moshaf_path: str | Path, download_path: str | Path):
    """download and gather sources so that the output file structure is:
    mohaf_path:
        media_file1
        media_file2
        media_file3
        ...
    """
    download_path = Path(download_path)
    moshaf_path = Path(moshaf_path)
    downloaded_pathes: list[Path] = []
    for idx, url in enumerate(urls):
        url_path = download_file_fast(
            url=url,
            out_path=download_path,
            extract_zip=True,
            remove_zipfile=True)

        downloaded_pathes.append(url_path)

    # get files pathes out of cascadded directories
    files_pathes = get_files(downloaded_pathes)
    file_names = [p.name for p in files_pathes]
    assert len(set(file_names)) == len(file_names), (
        f'There are duplicate files in downloading. Download_path: {download_path.absolute()}')

    # TODO: check files are:
    # *. media
    # *. check for name resolution 001, 002, ...

    # move these files into moshaf_path
    for filepath in files_pathes:
        filepath.rename(moshaf_path / filepath.name)

    # remove downloaded directories
    for path in downloaded_pathes:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            raise RuntimeError('File has to be moved to moshaf directory')


def get_files(pathes: list[Path]) -> list[Path]:
    """retrive files from tree structerd files

    Example:
    .
    ├── 001.mp3
    ├── alhossary
    ├── hafs
    │   └── quran-recitations
    │       ├── 001.mp3
    │       ├── 008 - الانقال.mp3
    │       ├── 018.mp3
    │       ├── 053_مشاري.mp3
    │       ├── 054_الحصري.mp3
    │       ├── 054_الرزيقي.mp3
    │       ├── 054_محمود_عبدالحكم.mp3
    │       ├── 114.mp3
    │       └── elhossary_fast
    │           ├── 001 - الفاتحة.mp3
    │           ├── 008 - الانقال.mp3
    │           ├── 035 - فاطر.mp3
    │           ├── 048 - الفتح.mp3
    │           ├── 053 - النجم.mp3
    │           ├── 054 - القمر.mp3
    │           ├── 055 - الرحمن.mp3
    │           ├── 062 - الجمعة.mp3
    │           ├── 078 - النبأ.mp3
    │           ├── 093 - الضحى.mp3
    │           └── 094 - الشرح.mp3
    ├── reciters.jsonl
    └── test.zip

    So the outfiles:
        data/hafs/quran-recitations/054_الرزيقي.mp3
        data/hafs/quran-recitations/008 - الانقال.mp3
        data/hafs/quran-recitations/018.mp3
        data/hafs/quran-recitations/001.mp3
        data/hafs/quran-recitations/054_الحصري.mp3
        data/hafs/quran-recitations/elhossary_fast/093 - الضحى.mp3
        data/hafs/quran-recitations/elhossary_fast/055 - الرحمن.mp3
        data/hafs/quran-recitations/elhossary_fast/094 - الشرح.mp3
        data/hafs/quran-recitations/elhossary_fast/008 - الانقال.mp3
        data/hafs/quran-recitations/elhossary_fast/054 - القمر.mp3
        data/hafs/quran-recitations/elhossary_fast/062 - الجمعة.mp3
        data/hafs/quran-recitations/elhossary_fast/048 - الفتح.mp3
        data/hafs/quran-recitations/elhossary_fast/053 - النجم.mp3
        data/hafs/quran-recitations/elhossary_fast/035 - فاطر.mp3
        data/hafs/quran-recitations/elhossary_fast/078 - النبأ.mp3
        data/hafs/quran-recitations/elhossary_fast/001 - الفاتحة.mp3
        data/hafs/quran-recitations/053_مشاري.mp3
        data/hafs/quran-recitations/114.mp3
        data/hafs/quran-recitations/054_محمود_عبدالحكم.mp3
        data/001.mp3
        data/test.zip
        data/alhossary
        data/reciters.jsonl
    """

    def recursive_search(path: Path, files_list: list[Path]):
        if path.is_file():
            files_list.append(path)
            return

        for curr_path in path.iterdir():
            if curr_path.is_file():
                files_list.append(curr_path)
            elif curr_path.is_dir():
                recursive_search(curr_path, files_list)

    files_list = []
    for path in pathes:
        recursive_search(path, files_list)

    return files_list
