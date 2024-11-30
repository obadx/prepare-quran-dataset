import streamlit as st
from menu import menu_with_redirect

from prepare_quran_dataset.construct.utils import (
    extract_sura_from_zekr,
    extract_suar_from_mp3quran,
    extract_suar_from_archive,
    dump_yaml,
)


def sources_view():
    st.markdown(
        'Extract Holy Quran\'s recitations\' links for specific moshaf from webpages as'
        'to fill `sources` attributes in moshaf parameters.')
    st.markdown('Example:')
    st.code(
        """
        https://server10.mp3quran.net/download/Aamer/109.mp3
        https://server10.mp3quran.net/download/Aamer/110.mp3
        https://server10.mp3quran.net/download/Aamer/111.mp3
        https://server10.mp3quran.net/download/Aamer/112.mp3
        """,
        language=None,
    )


def specific_sources_view():
    st.markdown(
        'Extract Holy Quran\'s recitations for specific moshaf from webpages as'
        ' `yaml` to be used in `specific_sources` in moshaf parameters.')
    st.markdown('yaml Schema: ')
    st.code(
        """
        "sura integer index from 1 to 114": "the url for the sura"
        """, language='yaml',
    )
    st.markdown('Example:')
    st.code(
        """
    18: https://cdns1.zekr.online/quran/5403/18/32.mp3
    19: https://cdns1.zekr.online/quran/5403/19/32.mp3
    20: https://cdns1.zekr.online/quran/5403/20/32.mp3
    21: https://cdns1.zekr.online/quran/5403/21/32.mp3
    22: https://cdns1.zekr.online/quran/5403/22/32.mp3
    23: https://cdns1.zekr.online/quran/5403/23/32.mp3
    """, language='yaml'
    )


def from_zekr():
    st.subheader('zekr.online')
    specific_sources_view()
    st.markdown(
        '![zekr](https://zekr.online/about/images/footer-logo.png)'
        'Extracting recitations for every sura for a [zekr.online](https://zekr.online/) moshaf like:'
        ' [https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403](https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403)'
        ' as a yaml'
    )
    url = st.text_input(
        'Enter Zekr moshaf Url',
        help='EX: https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403')
    if url:
        out = extract_sura_from_zekr(url)
        st.code(dump_yaml(out), language='yaml')


def from_mp3quran():
    st.subheader('mp3quran')
    specific_sources_view()
    st.markdown(
        '![mp3quran](https://mp3quran.net/uploads/apps/lg/mp3-quran-maine-2-1-1.png)')
    st.markdown(
        'Extracting recitations for every sura for a [mp3quran.net](https://mp3quran.net) moshaf like:'
        ' [https://mp3quran.net/ar/Aamer](https://mp3quran.net/ar/Aamer)'
        ' as a yaml'
    )
    url = st.text_input(
        'Enter mp3quran moshaf Url',
        help='EX: https://mp3quran.net/ar/Aamer')
    if url:
        out = extract_suar_from_mp3quran(url)
        st.code(dump_yaml(out), language='yaml')


def from_archive():
    st.subheader('Internet Archive')
    sources_view()
    st.markdown(
        '![Internet Archive logo](https://archive.org/services/img/ia-logos_dup/full/pct:200/0/default.jpg)')
    st.markdown(
        'Extracting recitations for every sura for a [https://archive.org/](https://archive.org/) moshaf like:'
        ' [https://archive.org/details/Al-Husary](https://archive.org/details/Al-Husary)'
        ' track link for every line'
    )
    url = st.text_input(
        'Enter Zekr moshaf Url',
        help='EX: https://archive.org/details/Al-Husary')
    if url:
        out = extract_suar_from_archive(url)
        st.code('\n'.join(out), language=None)


def suar_links_main():
    st.header(
        'Extract recitation from Holy Quran Recitations webpages')

    webpage = st.selectbox(
        'Please Chose Webpage to extract recitations for Reciter\'s Moshaf',
        ['zekr.online', 'mp3quran', 'archive.org'],
    )
    match webpage:
        case 'zekr.online':
            from_zekr()
        case 'mp3quran':
            from_mp3quran()
        case 'archive.org':
            from_archive()


if __name__ == '__main__':
    # displays sidebar menu & redirect to main page if not initialized
    menu_with_redirect()
    suar_links_main()
