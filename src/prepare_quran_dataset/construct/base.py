from abc import ABC, abstractmethod
from pathlib import Path
from datasets import load_dataset, Dataset
from pydantic import BaseModel
from typing import Any, Iterable


class Pool(ABC):
    """Abstract class of pool of pydantic Models

    Attributes:
        dataset_dict (dict[Any, Any]): dict[id_item: item] for easy indexing
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

        self.reset()

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def insert(self, new_item: BaseModel):
        ...

    @abstractmethod
    def generate_id(self, item: BaseModel | dict):
        ...

    def get_huggingface_dataset(self) -> Dataset:
        if len(self.dataset_dict) == 0:
            return Dataset.from_list([])
        return Dataset.from_list(list(self.dataset_dict.values()))

    def update(self, new_item: BaseModel):
        """Updates and item in the dataset
        """
        id = dict(new_item)[self.id_column]
        if id in self.dataset_dict:
            self.dataset_dict[id] = dict(new_item)
        else:
            raise KeyError(f'{id} is not found in the recitaion pool')
        self.after_update(new_item)

    def after_update(self, new_item: BaseModel):
        ...

    def __getitem__(self, id: Any):
        return self.item_type(**self.dataset_dict[id])

    def __len__(self):
        return len(self.dataset_dict)

    def save(self):
        self.get_huggingface_dataset().to_json(self.path, force_ascii=False)

    def __str__(self):
        return self.get_huggingface_dataset().__str__()

    def __iter__(self) -> Iterable[BaseModel]:
        for item in self.dataset_dict.values():
            yield self.item_type(**item)
