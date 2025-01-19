import re
import urllib
from typing import Literal

from .quran_data_utils import SUAR_LIST, SURA_TO_AYA_COUNT, normalize_text


def get_aya_standard_name(name: str | int) -> str:
    """gets the standard name of the aya

    * We only support https://everyayah.com/ format as "xxxyyy.mp3"
        where xxx is the sura index starting form 1
        and yyy is the aya index starting from 0 to the total aya count for sura, where 0 for استعاذة or بسملة  not an independet aya
        Example: name = `002023` is the same as `2023` representing: the verse (aya) 23 of sura number 2
    * We also accept strings: ['audhubillah', 'bismillah']  for استعاذة and بسملة
    """
    error_str = f'Aya should be with everyayah.com format as: `xxxyyy` where `xxx` is the sura index form(1) to (114) and `yyy` is the ayah index from(0) to max_aya count for sura. Example: `002100` is equvilent to `2100` where sura idx is 2 and aya index is 100'
    try:
        int_name = int(name)
    except ValueError:
        if name in ['audhubillah', 'bismillah']:
            return name
        else:
            raise AssertionError(error_str)

    sura_idx = int_name // 1000
    aya_idx = int_name % 1000

    assert sura_idx <= 114 and sura_idx >= 1, (
        f'{error_str}.  got sura_idx=`{sura_idx}` of name=`{name}`')

    assert aya_idx >= 0 and aya_idx <= SURA_TO_AYA_COUNT[sura_idx], (
        f'{error_str}. Max Aya count for sura({sura_idx}) is ({SURA_TO_AYA_COUNT[sura_idx]}) got aya_idx=`{aya_idx}` of input=`{name}`')

    return f'{int_name:0{6}}'


def get_sura_standard_name(name: str) -> str:
    """Returns the standard name of the sura represnted by the Sura's Index i.e("001")
    Args:
        name (str): the name of the sura represnted by:
        * the sura Arabic name i.e("البقرة")
        * or by the sura's index i.e ("002") or ("2")
        * or by bothe i.e("البقرة 2")

    Returns:
        (str): the sura's index as a standard name i.e("002")
    """

    # searching for the Arabic name of the sura
    name_normalized = normalize_text(name)
    suar_list = SUAR_LIST
    chosen_idx = None
    for idx, sura_name in enumerate(suar_list):
        sura_name_normalized = normalize_text(sura_name)
        if re.search(sura_name_normalized, name_normalized):
            # save the longest sura name
            if chosen_idx:
                if len(sura_name) > len(suar_list[chosen_idx]):
                    chosen_idx = idx
            else:
                chosen_idx = idx
    if chosen_idx is not None:
        return f'{chosen_idx + 1:0{3}}'

    # search first for numbers "002", or "2"
    # TODO: refine this regs to be specific
    re_result = re.search(r'\d+', name)
    if re_result:
        num = int(re_result.group())
        if num >= 1 and num <= 114:
            return f'{num:0{3}}'
        else:
            raise AssertionError(
                f'sura index should be between 1 and 114 got: ({num}) of name: ({name})')

    raise AssertionError(
        f'Sura name is not handeled in this case. name="{name}"')


def get_file_name(
    filename: str,
    segmented_by: Literal['sura', 'aya', 'none'] = 'sura',
) -> str:
    """Returns the filename is a valid  sura or aya name

    Raises:
        AssertetionError: if the file is not a valid sura or aya
    """
    filename = urllib.parse.unquote(filename)

    splits = filename.split('.')
    assert len(splits) == 2, (
        f'The filename ({filename}) does not has an extention ex:(.mp3) or have more than one dot (.)')
    name = splits[0]
    ext = splits[1]

    match segmented_by:
        case 'sura':
            name = get_sura_standard_name(name)
        case 'aya':
            name = get_aya_standard_name(name)

    return f'{name}.{ext}'
