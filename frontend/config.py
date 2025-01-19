from pathlib import Path
import re


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


def parse_link_in_list(links: list[str], line_determiner: re.Pattern = re.compile(r'^http')):
    """Creates a text area with each line is a sperate item identified by `line_determiner`

    Args:
        line_determiner (str): the re pattern that each line should be with EX: re.compile(r'^http')
    """
    clean_text_list = []
    for text in links:
        if line_determiner.match(text):
            clean_text_list.append(text)
    return clean_text_list


POPUP_MSG_ICONS: dict[str, str] = {
    'success': '✅',
    'error': '❌',
    'warn': '⚠️',
    'info': 'ℹ️',
}

BASE_DIR = Path('../../../quran-dataset')
RECITER_POOL_FILE = BASE_DIR / 'reciter_pool.jsonl'
MOSHAF_POOL_FILE = BASE_DIR / 'moshaf_pool.jsonl'
DOWNLOAD_PATH = BASE_DIR / 'Downloads'
DATASET_PATH = BASE_DIR / 'dataset'
DOWNLOAD_LOCK_FILE = BASE_DIR / 'download.lock'
DOWNLOAD_ERROR_LOG = BASE_DIR / 'download_error.json'

REQUIRED_MOSHAF_FIELDS = [
    'is_annotated',
    'name',
    'reciter_id',
    'sources',
    'specific_sources',
    'segmented_by',
    'publisher',
    'comments',
    'rewaya',
    'takbeer',
    'madd_monfasel_len',
    'madd_mottasel_len',
    'madd_mottasel_waqf',
    'madd_aared_len',
    'madd_alleen_len',
    'ghonna_lam_and_raa',
    'meem_aal_imran',
    'madd_yaa_alayn_alharfy',
    'saken_before_hamz',
    'sakt_iwaja',
    'sakt_marqdena',
    'sakt_man_raq',
    'sakt_bal_ran',
    'sakt_maleeyah',
    'between_anfal_and_tawba',
    'noon_and_yaseen',
    'yaa_ataan',
    'start_with_ism',
    'yabsut',
    'bastah',
    'almusaytirun',
    'bimusaytir',
    'tasheel_or_madd',
    'yalhath_dhalik',
    'irkab_maana',
    'noon_tamnna',
    'harakat_daaf',
    'alif_salasila',
    'idgham_nakhluqkum',
    'raa_firq',
    'raa_alqitr',
    'raa_misr',
    'raa_nudhur',
    'raa_yasr',
]

EXCLUDED_MSHAF_ITEM_FIELDS_IN_VIEW = [
    'recitation_files',
]


MOSHAF_FIELD_FUNCS_AFTER_SUBMIT = {
    'sources': parse_link_in_list,
    # 'specific_sources': validate_sura_index,
}
