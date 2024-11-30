from dataclasses import dataclass
from typing import Any, get_args, get_origin, Literal

from pydantic.fields import FieldInfo, PydanticUndefined


@dataclass
class MoshafFieldDocs:
    english_name: str
    arabic_name: str
    english2arabic_map: dict[Any, Any]
    more_info: str = ''


def get_moshaf_field_docs(fieldname: str, fieldinfo: FieldInfo) -> MoshafFieldDocs:
    """Retuns the Moshaf Field docs as a `MoshafFieldDocs`

    If the attribute is not Quran specifc will return None

    Returns:
        MoshafFieldDocs
    """
    if get_origin(fieldinfo.annotation) != Literal:
        return None
    docs = fieldinfo.description
    if docs == PydanticUndefined or not docs:
        return None

    arabic_name = get_arabic_name(fieldinfo)
    # Filterout None Quranic Attributes
    if arabic_name is None:
        return None

    english2arabic_map = get_arabic_attributes(fieldinfo)
    if english2arabic_map is None:
        choices = list(get_args(fieldinfo.annotation))
        english2arabic_map = {c: c for c in choices}

    return MoshafFieldDocs(
        arabic_name=arabic_name,
        english_name=fieldname,
        english2arabic_map=english2arabic_map,
        more_info=docs,
    )


def get_arabic_attributes(field_info: FieldInfo) -> dict[str, str] | None:
    """get the Arabic attributes maping from English for `Literal` type fields

    Returns:
        * the Arabic Attributes as {"English vlaue": "Arabie Value"}
        * None: if there is no Arabic Name
    """
    if field_info.json_schema_extra:
        if 'field_arabic_attrs_map' in field_info.json_schema_extra:
            return field_info.json_schema_extra['field_arabic_attrs_map']
    return None


def get_arabic_name(field_info: FieldInfo) -> str | None:
    """get the Arabic name out of the field description

    Retusns:
        * the Arabic Name, rest of the docs
        * None: if there is no Arabic Name

    """
    if field_info.json_schema_extra:
        if 'field_arabic_name' in field_info.json_schema_extra:
            return field_info.json_schema_extra['field_arabic_name']

    return None
