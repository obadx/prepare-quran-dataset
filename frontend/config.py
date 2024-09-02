from pathlib import Path
import re
import json


def text_to_list(text: str, line_determiner: re.Pattern = re.compile(r'^http')):
    """Creates a text area with each line is a sperate item identified by `line_determiner`

    Args:
        line_determiner (str): the re pattern that each line should be with EX: re.compile(r'^http')
    """
    text_list: list[str] = text.split('\n')
    clean_text_list = []
    for text in text_list:
        if line_determiner.match(text):
            clean_text_list.append(text)
    return clean_text_list


def json_text_to_dict(text: str) -> dict[str, str]:
    """
    Args:
        text(str): is JSON fromat of "sura name": "url" with new line with
            Example:
            {
            "002": "https//example.com.002.mp3",
            "004": "https//example.com.004.mp3"
            }
    """
    if text:
        return json.loads(text)
    return {}


BASE_DIR = Path('DEMO_DIR')
RECITER_POOL_FILE = BASE_DIR / 'reciter_pool.jsonl'
MOSHAF_POOL_FILE = BASE_DIR / 'moshaf_pool.jsonl'
DOWNLOAD_PATH = BASE_DIR / 'Downloads'
DATASET_PATH = BASE_DIR / 'dataset'
DOWNLOAD_LOCK_FILE = BASE_DIR / 'download.lock'

REQUIRED_MOSHAF_FIELDS = [
    'name',
    'reciter_id',
    'sources',
    'specific_sources',
    'is_sura_parted',
    'publisher',
    'comments',
    'rewaya',
    'madd_monfasel_len',
    'madd_mottasel_len',
    'madd_aared_len',
    'madd_mottasel_mahmooz_aared_len',
    'madd_alayn_lazem_len',
    'tasheel_or_madd',
    'daaf_harka',
    'idghaam_nakhlqkm',
    'noon_tamanna',
]


MOSHAF_FIELD_FUNCS_AFTER_SUBMIT = {
    'sources': text_to_list,
    'specific_sources': json_text_to_dict,
}
