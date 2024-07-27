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

    # def reset(self):
    #     # items hash to validate Unique ID generation
    #     self._items_hash = set()
    #     for item in self.dataset_dict.values():
    #         self._items_hash.add(self._get_hash(item))
    #     assert len(self._items_hash) == len(self.dataset_dict), (
    #         'Duplicate Items inside the Reciter Database')

    def get_hash(self, item: dict[str, Any] | Reciter) -> str:
        if isinstance(item, Reciter):
            item = dict(item)
        clean_text = re.sub(r'\s+', '', item['arabic_name'])
        return f'{clean_text}_{item["country_code"]}'

    def generate_id(self, new_item: Reciter) -> int:
        return self.__len__()

    # def insert(self, new_item: Reciter):
    #     new_item_hash = self.get_hash(new_item)
    #
    #     # check if the item aleady exists
    #     if new_item_hash in self.items_hash:  # O(1) using hashing
    #         raise ItemExistsError('The item already exists')
    #
    #     new_id = self.generate_id(new_item)
    #     self.dataset_dict[new_id] = dict(new_item)
    #
    #     self.dataset_dict[new_id][self.id_column] = new_id
    #
    #     self.items_hash.add(new_item_hash)  # O(1)

    # def validate_before_update(self, new_item: BaseModel):
    #     old_item = self.__getitem__(new_item.id)
    #     new_item_hash = self.get_hash(new_item)
    #     old_item_hash = self.get_hash(old_item)
    #     if old_item_hash == new_item_hash:
    #         return
    #
    #     # removing old_item's hash
    #     self.items_hash.discard(old_item_hash)  # O(1)
    #
    #     # the new item share save iformatin for items exists in the pool
    #     if new_item_hash in self.items_hash:  # O(1)
    #         raise ItemExistsError(
    #             f'Your updating a reciter of with another reciters that exists in our database.\n Your New Reciter: {new_item}.\nThe  Reciter you are updating: {old_item}')
    #
    #     # Every Thing is working ->insert new hash
    #     self.items_hash.add(new_item_hash)  # O(1)
