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
        self._items_hash = set()
        for item in self.dataset_dict.values():
            self._items_hash.add(self._get_hash(item))
        assert len(self._items_hash) == len(self.dataset_dict), (
            'Duplicate Items inside the Reciter Database')

    def _get_hash(self, item: dict[str, Any] | Reciter) -> str:
        if isinstance(item, Reciter):
            item = dict(item)
        clean_text = re.sub(r'\s+', '', item['arabic_name'])
        return f'{clean_text}_{item["country_code"]}'

    def generate_id(self, new_item: Reciter) -> tuple[int, str]:
        new_item_hash = self._get_hash(new_item)
        if new_item_hash in self._items_hash:  # O(1) using hashing
            raise ItemExistsError('The item already exists')

        return self.__len__(), new_item_hash

    def insert(self, new_item: Reciter):
        new_id, new_item_hash = self.generate_id(new_item)
        self.dataset_dict[new_id] = dict(new_item)
        self.dataset_dict[new_id][self.id_column] = new_id
        self._items_hash.add(new_item_hash)  # O(1)

    def before_update(self, new_item: Reciter):
        old_item = self.__getitem__(new_item.id)
        new_item_hash = self._get_hash(new_item)
        old_item_hash = self._get_hash(old_item)
        if old_item_hash == new_item_hash:
            return

        self._items_hash.discard(old_item_hash)  # O(1)
        # the new item share save iformatin for items exists in the pool
        if new_item_hash in self._items_hash:  # O(1)
            raise ItemExistsError(
                f'Your updating a reciter of with another reciters that exists in our database.\n Your New Reciter: {new_item}.\nThe  Reciter you are updating: {old_item}')
        # Every Thing is working -> remove old hash & insert new hash
        self._items_hash.add(new_item_hash)  # O(1)
