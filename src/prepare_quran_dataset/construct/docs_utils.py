import re
from dataclasses import dataclass
from typing import Any, get_args, get_origin, Literal
import json

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

    arabic_out = get_arabic_name(docs)
    # Filterout None Quranic Attributes
    if not arabic_out:
        return None
    arabic_name, docs = arabic_out

    attr_out = get_arabic_attributes(docs)
    english2arabic_map = {}
    if attr_out:
        english2arabic_map, docs = attr_out
    else:
        choices = list(get_args(fieldinfo.annotation))
        english2arabic_map = {c: c for c in choices}

    return MoshafFieldDocs(
        arabic_name=arabic_name,
        english_name=fieldname,
        english2arabic_map=english2arabic_map,
        more_info=docs,
    )


def get_arabic_attributes(docs: str) -> dict[str, str] | None:
    """get the Arabic attributes for `Literal` type fields

    Returns:
        * tuple[dict[str, str], str]:
            (the Arabic Attributes as {"English vlaue": "Arabie Value"}, rest of the docs)
        * None: if there is no Arabic Name
    """
    if docs:
        match = re.search(
            r'ArabicAttr\((.*?)\)',
            docs, re.DOTALL)
        if match:
            return json.loads(match.group(1)), docs[: match.start()] + docs[match.end():]
    return None


def get_arabic_name(docs: str) -> tuple[str, str]:
    """get the Arabic name out of the field description

    Retusns:
        * tuple[str, str]: (the Arabic Name, rest of the docs)
        * None: if there is no Arabic Name

    """
    if docs:
        match = re.search(
            r'ArabicName\((.*)\)',
            docs, re.UNICODE)
        if match:
            return match.group(1), docs[: match.start()] + docs[match.end():]
    return None
