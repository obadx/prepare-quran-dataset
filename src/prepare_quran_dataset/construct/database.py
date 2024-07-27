import re
from typing import Any

from prepare_quran_dataset.construct.base import Pool
from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf


class ReciterPool(Pool):
    """Pool or Reciter see Pool to understand structure"""

    def __init__(self, path='data/reciters.jsonl'):
        super().__init__(
            path=path,
            item_type=Reciter,
            id_column='id',
        )

    def get_hash(self, item: dict[str, Any] | Reciter) -> str:
        if isinstance(item, Reciter):
            item = dict(item)
        clean_text = re.sub(r'\s+', '', item['arabic_name'])
        return f'{clean_text}_{item["country_code"]}'

    def generate_id(self, new_item: Reciter) -> int:
        return self.__len__()
