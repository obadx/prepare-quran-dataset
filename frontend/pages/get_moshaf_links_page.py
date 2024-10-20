import json
from pathlib import Path

import streamlit as st
from menu import menu_with_redirect

from prepare_quran_dataset.construct.utils import (
    extract_sura_from_zekr,
    extract_suar_from_mp3quran,
)


def from_zekr():
    st.subheader('zekr.online')
    st.markdown(
        '![zekr](https://zekr.online/about/images/footer-logo.png)'
        'Extracting recitations for every sura for a [zekr.online](https://zekr.online/) moshaf like:'
        ' [https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403](https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403)'
        ' as a JSON'
    )
    url = st.text_input(
        'Enter Zekr moshaf Url',
        help='EX: https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403')
    if url:
        out = extract_sura_from_zekr(url)
        st.code(json.dumps(out, indent=4, ensure_ascii=False), language='json')


def from_mp3quran():
    st.subheader('mp3quran')
    st.markdown(
        '![mp3quran](https://mp3quran.net/uploads/apps/lg/mp3-quran-maine-2-1-1.png)')
    st.markdown(
        'Extracting recitations for every sura for a [mp3quran.net](https://mp3quran.net) moshaf like:'
        ' [https://mp3quran.net/ar/Aamer](https://mp3quran.net/ar/Aamer)'
        ' as a JSON'
    )
    url = st.text_input(
        'Enter Zekr moshaf Url',
        help='EX: https://mp3quran.net/ar/Aamer')
    if url:
        out = extract_suar_from_mp3quran(url)
        st.code(json.dumps(out, indent=4, ensure_ascii=False), language='json')


def suar_links_main():
    st.header(
        'Extract recitation from Holy Quran Recitations webpages')
    st.markdown(
        'Extract Holy Quran\'s recitations for specific moshaf from webpages as'
        ' JSON to be used in `specific_sources` in moshaf parameters.')
    st.markdown('JSON Schema: ')
    st.code(
        """
        {
            "sura index": "url of the sura",
        }
        """, language='json',
    )
    st.markdown('Example:')
    st.code(
        """
    {
        "110": "https://server10.mp3quran.net/download/Aamer/110.mp3",
        "111": "https://server10.mp3quran.net/download/Aamer/111.mp3",
        "112": "https://server10.mp3quran.net/download/Aamer/112.mp3",
        "113": "https://server10.mp3quran.net/download/Aamer/113.mp3",
        "114": "https://server10.mp3quran.net/download/Aamer/114.mp3"
    }
        """
    )

    webpage = st.selectbox(
        'Please Chose Webpage to extract recitations for Reciter\'s Moshaf',
        ['zekr.online', 'mp3quran'],
    )
    match webpage:
        case 'zekr.online':
            from_zekr()
        case 'mp3quran':
            from_mp3quran()


# displays sidebar menu & redirect to main page if not initialized
menu_with_redirect()
suar_links_main()
