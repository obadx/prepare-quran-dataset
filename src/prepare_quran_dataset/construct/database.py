import re
from typing import Any
from pathlib import Path
import shutil
import os
from collections import defaultdict
import urllib.parse


from prepare_quran_dataset.construct.base import Pool
from prepare_quran_dataset.construct.data_classes import (
    Reciter,
    Moshaf,
    AudioFile
)
from prepare_quran_dataset.construct.utils import (
    download_file_fast,
    get_audiofile_info,
    get_suar_list,
    normalize_text
)


class ReciterPool(Pool):
    """Pool or Reciter see Pool to understand structure"""

    def __init__(self, path='data/reciter_pool.jsonl'):
        super().__init__(
            path=path,
            item_type=Reciter,
            id_column='id',
        )

        # the id of the last element
        self._last_idx = -1
        if self.dataset_dict:
            self._last_idx = sorted(self.dataset_dict.keys())[-1]

    def get_hash(self, item: dict[str, Any] | Reciter) -> str:
        if isinstance(item, Reciter):
            item = item.model_dump()
        clean_text = re.sub(r'\s+', '', item['arabic_name'])
        return f'{clean_text}_{item["country_code"]}'

    def generate_id(self, new_item: Reciter) -> int:
        self._last_idx += 1
        return self._last_idx

    def after_delete(self, reciter: Reciter) -> None:
        """update self._last_idx"""
        if self._last_idx == reciter.id:
            self._last_idx -= 1


class MoshafPool(Pool):
    """Pool or Moshaf see Pool to understand structure"""

    def __init__(
        self,
        reciter_pool: ReciterPool,
        dataset_path='data/',
        metadata_path='data/moshaf_pool.jsonl',
        download_path='data/Downloads',
    ):
        super().__init__(
            path=metadata_path,
            item_type=Moshaf,
            id_column='id',
        )
        self._reciter_pool = reciter_pool
        self.dataset_path = Path(dataset_path)
        self.download_path = Path(download_path)

    def get_hash(self, item: dict[str, Any] | Moshaf) -> str:
        """We will use url and reciter's ID as a unique Identifier"""
        if isinstance(item, dict):
            item = Moshaf(**item)
        urls_text = '_'.join(item.sources)
        return f'{item.reciter_id}_{urls_text}'

    def generate_id(self, new_moshaf: Moshaf) -> int:
        """The id is "{reciter_id}.{moshaf_id}" example (0.1)
        Reciter ID = 0
        Moshaf number 1 of Reciter(0)
        """
        reciter: Reciter = self._reciter_pool[new_moshaf.reciter_id]

        moshaf_list: list[int] = []
        for str_idx in reciter.moshaf_set_ids:
            moshaf_list.append(int(str_idx.split('.')[1]))
        moshaf_id = 0
        if moshaf_list:
            moshaf_id = sorted(moshaf_list)[-1] + 1

        return f'{new_moshaf.reciter_id}.{moshaf_id}'

    def process_new_item_before_insert(self, new_item: Moshaf) -> Moshaf:
        return self.update_reciter_metadata_in_moshaf(new_item)

    def after_insert(self, new_moshaf: Moshaf) -> None:
        self._add_moshaf_to_reciter(new_moshaf)

    def update(self, new_moshaf: Moshaf) -> None:
        moshaf_id = new_moshaf.id
        if self.__getitem__(moshaf_id).reciter_id != new_moshaf.reciter_id:
            super().update(new_moshaf, generate_new_id=True)
        else:
            super().update(new_moshaf, generate_new_id=False)

    def process_new_item_before_update(self, new_item: Moshaf) -> Moshaf:
        return self.update_reciter_metadata_in_moshaf(new_item)

    def update_reciter_metadata_in_moshaf(self, new_item: Moshaf) -> Moshaf:
        new_item = new_item.model_copy(deep=True)
        reciter = self._reciter_pool[new_item.reciter_id]
        new_item.reciter_arabic_name = reciter.arabic_name
        new_item.reciter_english_name = reciter.english_name
        new_item.model_validate(new_item)
        return new_item

    # removing the moshaf form reciter's moshaf_set_ids
    def after_update(self, old_moshaf: Moshaf, new_moshaf: Moshaf) -> None:
        reciter = self._reciter_pool[old_moshaf.reciter_id]
        reciter.moshaf_set_ids.discard(old_moshaf.id)
        self._reciter_pool.update(reciter)

        self._add_moshaf_to_reciter(new_moshaf)

    def after_delete(self, deleted_moshaf: Moshaf) -> None:
        """delete the moshaf id from the reciter pool"""
        reciter: Reciter = self._reciter_pool[deleted_moshaf.reciter_id]
        reciter.moshaf_set_ids.discard(deleted_moshaf.id)
        self._reciter_pool.update(reciter)

        # saving pools
        self._reciter_pool.save()
        self.save()

        # delete moshsf_item media files
        if Path(deleted_moshaf.path).is_dir():
            shutil.rmtree(deleted_moshaf.path)

    def _add_moshaf_to_reciter(
        self,
        new_moshaf: Moshaf,
        save_reciter_pool=False,
    ) -> None:
        """Adding a moshaf to a reciter"""
        reciter = self._reciter_pool[new_moshaf.reciter_id]
        reciter.moshaf_set_ids.add(new_moshaf.id)
        self._reciter_pool.update(reciter)
        if save_reciter_pool:
            self._reciter_pool.save()

    def download_moshaf(self, id: str, redownload=False, save_on_disk=True):
        """Download a moshaf and add it to the reciter"""
        moshaf = self.__getitem__(id)
        moshaf = download_media_and_fill_metadata(
            moshaf,
            database_path=self.dataset_path,
            download_path=self.download_path,
            redownload=redownload,
            is_sura_parted=moshaf.is_sura_parted,
        )

        # update the moshaf in the pool
        self.update(moshaf)

        # update the moshaf for the reciter pool
        reciter = self._reciter_pool[moshaf.reciter_id]
        reciter.moshaf_set_ids.add(moshaf.id)
        self._reciter_pool.update(reciter)

        # saving on disk
        if save_on_disk:
            self.save()
            self._reciter_pool.save()

    def download_all_moshaf(self, redownload=False, save_on_disk=True):
        """Download all moshaf and updae MohafPool and ReciterPool"""
        for moshaf in self:
            self.download_moshaf(
                moshaf.id, redownload=redownload, save_on_disk=save_on_disk)


def download_media_and_fill_metadata(
    item: Moshaf,
    database_path: Path | str,
    download_path: Path | str,
    redownload=False,
    is_sura_parted=True,
) -> Moshaf:
    """
    Downlaoad audio files from url, gather them in a single directory
    and fill metadata of the moshaf

    Args:
        is_sura_parted (bool): if every recitation file is a sperate
            sura or not

    Returns:
        (Moshaf): the moshaf filled with metadata
    """
    item = item.model_copy(deep=True)

    # if the moshaf is already downloaded do not download it unless `redownload`
    if item.is_downloaded and not redownload:
        print(f'Mohaf({item.id} Downloaded and processed Existing ........')
        return item

    # Downloadint the moshaf and processing metadata
    database_path = Path(database_path)
    moshaf_path = database_path / item.id
    download_path = Path(download_path)
    os.makedirs(moshaf_path, exist_ok=True)
    download_moshaf_from_urls(
        urls=item.sources,
        moshaf_name=item.id,
        moshaf_path=moshaf_path,
        download_path=download_path,
        is_sura_parted=is_sura_parted,
    )

    # Fill Moshaf's Metadata
    total_duration_minutes = 0.0
    total_size_megabytes = 0.0
    for filepath in moshaf_path.iterdir():
        audio_file_info = get_audiofile_info(filepath)
        if audio_file_info:
            audio_file = AudioFile(
                name=filepath.name,
                path=str(filepath.absolute()),
                sample_rate=audio_file_info.sample_rate,
                duration_minutes=audio_file_info.duration_seconds / 60.0)
            item.recitation_files.append(audio_file)
            total_duration_minutes += audio_file_info.duration_seconds / 60.0
            total_size_megabytes += filepath.stat().st_size / (1024.0 * 1024.0)

    item.path = str(moshaf_path.absolute())
    item.num_recitations = len(item.recitation_files)
    item.total_duraion_minutes = total_duration_minutes
    item.is_complete = len(item.recitation_files) == 114
    item.total_size_mb = total_size_megabytes
    item.is_downloaded = True

    return item


def download_moshaf_from_urls(
    urls: list[str],
    moshaf_path: str | Path,
    moshaf_name: str,
    download_path: str | Path,
    remove_after_download=False,
    is_sura_parted=True,
):
    """Download the moshaf media files and store it in `moshaf_path`

    Download the media from urls either (zip file, or media file) in
    `download_path/moshaf_name` and then extract all media files in
    `moshaf_path`
    mohaf_path:
        media_file1
        media_file2
        media_file3
        ...

    Args:
        urls list[str]: list of urls either zip or single medila files
        moshaf_path (str | Path): path to storm moshaf media files
        moshaf_name (str): the moshaf name
        download_path (str): Base Path to download files
        is_sura_parted (bool): if every recitation file is a sperate sura or not
    """
    def get_file_name(name: str) -> str:

        # converting unicode url name to Actual unicode
        name = urllib.parse.unquote(name)
        if is_sura_parted:
            return get_sura_standard_name(name)
        return name

    download_path = Path(download_path) / moshaf_name
    moshaf_path = Path(moshaf_path)
    downloaded_pathes: list[Path] = []
    for idx, url in enumerate(urls):
        url_path = download_file_fast(
            url=url,
            out_path=download_path,
            extract_zip=True,
            remove_zipfile=True)

        downloaded_pathes.append(url_path)

    # Check for duplicate files in downloaded media
    files_pathes = get_files(downloaded_pathes)
    file_names = [p.name for p in files_pathes]
    files_count = defaultdict(lambda: 0)
    for file in file_names:
        files_count[get_file_name(file)] += 1
    duplicate_files = [file for file,
                       count in files_count.items() if count > 1]
    assert duplicate_files == [], (
        f'Duplicate Files. These files are duplicated {duplicate_files}. Download_path: {download_path.absolute()}')

    # copy downloaded media into moshaf path
    for filepath in files_pathes:
        shutil.copy(
            filepath, moshaf_path / get_file_name(filepath.name))

    if remove_after_download:
        shutil.rmtree(download_path)


def get_sura_standard_name(filename: str) -> str:
    """Returns the standard name of the sura represnted by the Sura's Index i.e("001")
    Args:
        name (str): the name of the sura represnted by:
        * the sura Arabic name i.e("البقرة")
        * or by the sura's index i.e ("002") or ("2")
        * or by bothe i.e("البقرة 2")

    Returns:
        (str): the sura's index as a standard name i.e("002")
    """
    splits = filename.split('.')
    assert len(splits) == 2, (
        f'The filename ({filename}) does not has an extention ex:(.mp3) or have more than one dot (.)')
    name = splits[0]
    extention = splits[1]
    sura_idx = None

    # search first for numbers "002", or "2"
    re_result = re.search(r'\d+', name)
    if re_result:
        sura_idx = int(re_result.group())

    # searching for the Arabic name of the sura
    else:
        name_normalized = normalize_text(name)
        suar_list = get_suar_list()
        for idx, sura_name in enumerate(suar_list):
            sura_name_normalized = normalize_text(sura_name)
            if re.search(sura_name_normalized, name_normalized):
                sura_idx = idx + 1

    assert sura_idx is not None, (
        f'Sura name is not handeled in this case. name={name}')
    return f'{sura_idx:0{3}}.{extention}'


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
