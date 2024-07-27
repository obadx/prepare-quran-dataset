from abc import ABC, abstractmethod
from pathlib import Path
from datasets import load_dataset, Dataset
from pydantic import BaseModel
from typing import Any, Iterable


class ItemExistsInPoolError(Exception):
    pass


class Pool(ABC):
    """Abstract class of pool of pydantic Models

    Attributes:
        dataset_dict (dict[Any, Any]): dict[id_item: item] for easy indexing
        items_hash (set[str]): a set of hash (unique id) for all items in the dataset
        id_column (str): the column that contains the id
        path (Path): the path to the ".jsonl" (json line) file
        item_type : the item class i.e (item type in the pool)
    """

    def __init__(self, path: str | Path, item_type, id_column='id'):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f'File "{self.path.absolute()}" not found')

        self.id_column = id_column
        self.item_type = item_type
        self.dataset_dict = {}
        self.items_hash = set()
        self.refresh()

    def refresh(self):
        """Reloading the dataset from storage
        """
        dataset = []
        with open(self.path, 'r') as f:
            file_content = f.read()
        if file_content:
            dataset = load_dataset(
                'json', data_files=self.path.absolute().__str__())['train']

        for item in dataset:
            self.dataset_dict[item[self.id_column]] = item

        # items hash to validate Unique ID generation
        for item in self.dataset_dict.values():
            self.items_hash.add(self.get_hash(item))
        assert len(self.items_hash) == len(self.dataset_dict), (
            'Duplicate Items inside the Reciter Database')

        self.reset()

    def reset(self):
        """[Optional]: You can override this method

        This method is called after refresh
        """
        ...

    @abstractmethod
    def get_hash(self, items: dict[str, Any] | BaseModel) -> str:
        ...

    def insert(self, new_item: BaseModel):
        new_item = self.process_new_item_before_insert(new_item)
        new_item_hash = self.get_hash(new_item)

        # check if the item aleady exists
        if new_item_hash in self.items_hash:  # O(1) using hashing
            raise ItemExistsInPoolError('The item already exists')

        new_id = self.generate_id(new_item)
        self.dataset_dict[new_id] = dict(new_item)
        self.dataset_dict[new_id][self.id_column] = new_id
        self.items_hash.add(new_item_hash)  # O(1)

    def process_new_item_before_insert(self, new_item: BaseModel) -> BaseModel:
        """[Optional]: You can override this method"""
        return new_item

    @abstractmethod
    def generate_id(self, item: BaseModel | dict) -> Any:
        """Returns: the item's new ID"""
        ...

    def get_huggingface_dataset(self) -> Dataset:
        if len(self.dataset_dict) == 0:
            return Dataset.from_list([])
        return Dataset.from_list(list(self.dataset_dict.values()))

    def update(self, new_item: BaseModel):
        """Updates and item in the dataset
        """
        new_item = self.process_new_item_before_update(new_item)
        new_item_hash = self.validate_before_update(new_item)

        id = dict(new_item)[self.id_column]
        if id in self.dataset_dict:
            self.dataset_dict[id] = dict(new_item)
        else:
            raise KeyError(f'{id} is not found in the recitaion pool')

        # Every Thing is working ->insert new hash
        self.items_hash.add(new_item_hash)  # O(1)

    def validate_before_update(self, new_item: BaseModel) -> str:
        """Validate the new item
        Returns:
            (str): The new_item's hash
        """
        old_item = self.__getitem__(new_item.id)
        new_item_hash = self.get_hash(new_item)
        old_item_hash = self.get_hash(old_item)
        if old_item_hash == new_item_hash:
            return

        # removing old_item's hash
        self.items_hash.discard(old_item_hash)  # O(1)

        # the new item share save iformatin for items exists in the pool
        if new_item_hash in self.items_hash:  # O(1)
            raise ItemExistsInPoolError(
                f'Your updating a reciter of with another reciters that exists in our database.\n Your New Reciter: {new_item}.\nThe  Reciter you are updating: {old_item}')

        return new_item_hash

    def process_new_item_before_update(self, new_item: BaseModel) -> BaseModel:
        """[Optional]: You can override this method"""
        return new_item

    def __getitem__(self, id: Any):
        try:
            return self.item_type(**self.dataset_dict[id])
        except KeyError:
            raise KeyError(
                f'The item with ID={id} does not exists in the database')

    def __len__(self):
        return len(self.dataset_dict)

    def save(self):
        self.get_huggingface_dataset().to_json(self.path, force_ascii=False)

    def __str__(self):
        return self.get_huggingface_dataset().__str__()

    def __iter__(self) -> Iterable[BaseModel]:
        for item in self.dataset_dict.values():
            yield self.item_type(**item)
