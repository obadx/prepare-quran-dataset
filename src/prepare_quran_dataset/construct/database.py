import re
from typing import Any

from prepare_quran_dataset.construct.base import Pool
from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf


class ItemExistsError(Exception):
    pass


class ReciterPool(Pool):
    """Pool or Reciter see Pool to understand structure"""

    def __init__(self, path='data/reciters.jsonl'):
        super().__init__(
            path=path,
            item_type=Reciter,
            id_column='id',
        )

    def reset(self):
        # items hash to validate Unique ID generation
        self._items_hash = {}
        for id, item in self.dataset_dict.items():
            self._items_hash[id] = self._get_hash(item)

    def _get_hash(self, item: dict[str, Any] | Reciter) -> str:
        if isinstance(item, Reciter):
            item = dict(item)
        clean_text = re.sub(r'\s+', '', item['arabic_name'])
        return f'{clean_text}_{item["country_code"]}'

    def generate_id(self, new_item: Reciter) -> tuple[int, str]:
        new_item_hash = self._get_hash(new_item)
        if new_item_hash in self._items_hash.values():
            raise ItemExistsError('The item already exists')

        return self.__len__(), new_item_hash

    def insert(self, new_item: Reciter):
        new_id, new_item_hash = self.generate_id(new_item)
        self.dataset_dict[new_id] = dict(new_item)
        self.dataset_dict[new_id][self.id_column] = new_id
        self._items_hash[new_id] = new_item_hash

    def after_update(self, new_item: Reciter):
        new_item_hash = self._get_hash(new_item)
        if self._items_hash[new_item.id] == new_item_hash:
            return

        for id, hash in self._items_hash.items():
            if id != new_item.id:
                if hash == new_item_hash:
                    raise ItemExistsError(
                        f'Your updating a reciter of with another reciters that exists in our database.\n Your New Reciter: {new_item}.\nThe exist Reciter: {self.__getitem__(id)}')
        self._items_hash[new_item.id] = new_item_hash
