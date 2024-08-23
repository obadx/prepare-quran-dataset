from pathlib import Path
import re


def text_to_list(text: str, line_determiner: re.Pattern = re.compile(r'^(http)|\d+')):
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


BASE_DIR = Path('DEMO_DIR')
RECITER_POOL_FILE = BASE_DIR / 'reciter_pool.jsonl'
MOSHAF_POOL_FILE = BASE_DIR / 'moshaf_pool.jsonl'
DOWNLOAD_PATH = BASE_DIR / 'Downloads'
DATASET_PATH = BASE_DIR / 'dataset'

REQUIRED_MOSHAF_FIELDS = [
    'name',
    'reciter_id',
    'sources',
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
    'sources': text_to_list
}
