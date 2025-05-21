from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Literal,
    Any,
    get_args,
    get_origin,
    Iterable,
    Optional,
)

from datasets import load_dataset, Dataset, Features, ClassLabel, Value, Sequence
from pydantic import BaseModel
from pydantic.fields import FieldInfo, PydanticUndefined

from .utils import load_jsonl, save_jsonl
from .docs_utils import get_arabic_attributes, get_arabic_name


class ItemExistsInPoolError(Exception):
    pass


def get_field_name(field_name: str, field_info: FieldInfo) -> str:
    """Recturn the Arabic name of the field if applicable else the field_name
    """
    label = field_name
    arabic_name = get_arabic_name(field_info)
    if arabic_name:
        label = f"{arabic_name} ({field_name})"

    return label


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
            # dataset = load_dataset(
            #     'json', data_files=self.path.absolute().__str__(),
            #     encoding='utf8',
            # )['train']
            dataset = load_jsonl(self.path)

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
    def get_hash(self, item: dict[str, Any] | BaseModel) -> str:
        ...

    def insert(self, new_item: BaseModel):
        new_item = self.process_new_item_before_insert(new_item)
        new_item_hash = self.get_hash(new_item)

        # check if the item aleady exists
        if new_item_hash in self.items_hash:  # O(1) using hashing
            raise ItemExistsInPoolError('The item already exists')

        new_id = self.generate_id(new_item)
        self.dataset_dict[new_id] = new_item.model_dump()
        self.dataset_dict[new_id][self.id_column] = new_id
        self.items_hash.add(new_item_hash)  # O(1)

        self.after_insert(self.__getitem__(new_id))

    def process_new_item_before_insert(self, new_item: BaseModel) -> BaseModel:
        """[Optional]: You can override this method"""
        return new_item

    def after_insert(self, new_item: BaseModel) -> None:
        """[Optional]: You can override this method"""
        ...

    def delete(self, id: Any) -> BaseModel:
        """Delete an item from the pool"""
        item = self.__getitem__(id)
        self.before_delete(item)

        item_hash = self.get_hash(item)
        self.items_hash.remove(item_hash)
        del self.dataset_dict[id]

        self.after_delete(item)

        return item

    def before_delete(self, deleted_item: BaseModel) -> None:
        """[Optional]: You can override this method"""
        ...

    def after_delete(self, deleted_item: BaseModel) -> None:
        """[Optional]: You can override this method"""
        ...

    @abstractmethod
    def generate_id(self, item: BaseModel | dict) -> Any:
        """Returns: the item's new ID"""
        ...

    def get_huggingface_dataset(self) -> Dataset:
        if len(self.dataset_dict) == 0:
            return Dataset.from_list([])
        return Dataset.from_list(list(self.dataset_dict.values()))

    def update(self, new_item: BaseModel, generate_new_id=False):
        """Updates and item in the dataset
        """
        old_item = self.__getitem__(new_item.id).model_copy(deep=True)
        assert old_item.id == new_item.id, (
            "The user has chaned the items'id which is forbidden")

        new_item = self.process_new_item_before_update(new_item)
        new_item_hash = self.validate_before_update(new_item)

        id = old_item.id
        if generate_new_id:
            del self.dataset_dict[id]
            id = self.generate_id(new_item)
        self.dataset_dict[id] = new_item.model_dump()

        # Every Thing is working ->insert new hash
        self.items_hash.add(new_item_hash)  # O(1)

        self.after_update(old_item, new_item)

    def after_update(self, old_item: BaseModel, new_item: BaseModel) -> None:
        """[Optional]: You can override this method"""
        ...

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

    def __str__(self):
        return self.get_huggingface_dataset().__str__()

    def __iter__(self) -> Iterable[BaseModel]:
        for item in self.dataset_dict.values():
            yield self.item_type(**item)

    def to_jsonl(self) -> str:
        """Converts the model to JSON Line string

        Returns:
            (str) the JSON Line represntation of the pool
        """
        text = ""
        for item in self.__iter__():
            text += item.model_dump_json() + '\n'
        if text:
            text = text[:-1]  # removes '\n' from last line
        return text

    def to_parquet(self, path: str | Path, excluded_fields: Optional[list[str]] = None) -> str:
        """Saves the dataset as parquet file
        """

        if excluded_fields is None:
            excluded_fields = set()

        # Finding dict[int, str] to cast it to dict[str, str]
        to_cast_keys = set()
        base_item = self.__getitem__(list(self.dataset_dict.keys())[0])
        item_type = type(base_item)

        items = []
        for item in self.__iter__():
            item_dict = item.model_dump(exclude=set(excluded_fields))
            # TODO: Hard coded
            if 'specific_sources' in item_dict:
                item_dict['specific_sources'] = [
                    {'sura_or_aya_index': str(k),
                     'url': v}
                    for k, v in item_dict['specific_sources'].items()]
            items.append(item_dict)

        features, metadata = item_type.extract_huggingface_features(
            excluded_fields)
        ds = Dataset.from_list(
            items,
            features=features,
        )
        ds.to_parquet(path)

    def save(self):
        json_line_pool = self.to_jsonl()
        with open(self.path, 'w+', encoding='utf-8') as f:
            f.write(json_line_pool)


class BaseDatasetModel(BaseModel):
    @classmethod
    def extract_huggingface_features(
        cls,
        exclueded_fields: Optional[list[str]] = None,
    ) -> tuple[Features, dict]:
        """Extracts Hugginface Dataset Features from BaseModel

        Args:
            required_fields (list[str]): the requried files to be inclued in the HF Dataset Features
        """
        if exclueded_fields is None:
            exclueded_fields = set()
        required_fields = set(cls.model_fields.keys()) - set(exclueded_fields)

        features = Features()  # a dict object
        metadata = {}
        for fieldname in required_fields:
            fieldinfo: FieldInfo = cls.model_fields[fieldname]

            metadata[fieldname] = {
                'arabic_name': get_arabic_name(fieldinfo),
                'arabic_attributes': get_arabic_attributes(fieldinfo),
                'description': fieldinfo.description,
            }

            # the args of a Literal typs > 0 EX: Literal[3, 4]
            dtype = fieldinfo.annotation
            if get_origin(fieldinfo.annotation) is Literal:
                choices = list(get_args(fieldinfo.annotation))
                dtype = type(choices[0])

            if dtype in [str, Optional[str]]:
                features[fieldname] = Value(dtype='string')

            elif dtype in [int, Optional[int]]:
                features[fieldname] = Value(dtype='int32')

            elif dtype in [float, Optional[float]]:
                features[fieldname] = Value(dtype='float32')

            elif dtype in [bool, Optional[bool]]:
                features[fieldname] = Value(dtype='bool')

            elif dtype in [
                list[str], Optional[list[str]], list, Optional[list],
                set[str], Optional[set[str]], set, Optional[set]
            ]:
                features[fieldname] = Sequence(feature=Value(dtype='string'))

            # TODO: Hard coded
            elif fieldname == 'specific_sources':
                features[fieldname] = [{
                    "sura_or_aya_index": Value(dtype="string"),
                    "url": Value(dtype="string"),
                }]

            else:
                raise ValueError(
                    f'Type: `{fieldinfo.annotation}` is not supported')

        return features, metadata
