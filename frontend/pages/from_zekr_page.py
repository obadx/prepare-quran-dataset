import json
from pathlib import Path

import streamlit as st
from menu import menu_with_redirect

from prepare_quran_dataset.construct.utils import extract_sura_from_zekr


def from_zekr():
    st.header('Extract recitation from zekr.online')
    st.markdown(
        '![zekr](https://zekr.online/about/images/footer-logo.png)'
        'Extracting recitations for every sura fro a [zekr.online](https://zekr.online/) moshaf like:'
        ' [https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403](https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403)'
        ' as a JSON'
    )
    st.code(
        """
        {
            "sura index": "url of the sura",
        }
        """, language='json',
    )
    url = st.text_input(
        'Enter Zekr moshaf Url',
        help='EX: https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403')
    if url:
        out = extract_sura_from_zekr(url)
        st.code(json.dumps(out, indent=4, ensure_ascii=False), language='json')


# displays sidebar menu & redirect to main page if not initialized
menu_with_redirect()
from_zekr()
