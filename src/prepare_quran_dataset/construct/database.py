import re
from typing import Any
from pathlib import Path

from prepare_quran_dataset.construct.base import Pool
from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf


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
