import json
from pathlib import Path
import re

from quran_transcript.utils import normalize_aya

DATA_PATH = Path(__file__).parent.parent / 'data'


def normalize_text(text: str) -> str:
    out_text = re.sub(r'-|_', '', text)
    out_text = re.sub("أ|إ|آ", "ا", out_text)
    return normalize_aya(
        out_text,
        remove_spaces=True,
        remove_tashkeel=True,
        remove_small_alef=True)


def get_suar_list(suar_path=DATA_PATH / 'suar_list.json') -> list[str]:
    """Return the suar names of the Holy Quran in an ordered list
    """
    with open(suar_path, 'r', encoding='utf8') as f:
        suar_list = json.load(f)
    return suar_list


def get_sura_to_aya_count(path=DATA_PATH / 'sura_to_aya_count.json') -> dict[int, int]:
    """Loads sura to aya count
    """
    with open(path, 'r') as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}


SUAR_LIST = get_suar_list()
SURA_TO_AYA_COUNT = get_sura_to_aya_count()
