import os

import streamlit as st

from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf
from prepare_quran_dataset.construct.database import ReciterPool, MoshafPool
from menu import menu
import config as conf


def set_up() -> None:
    """Initialize ReciterPool and Moshaf_pool"""

    if 'started' not in st.session_state:
        st.session_state.started = True

    if 'reciter_pool' not in st.session_state:
        if not conf.RECITER_POOL_FILE.is_file():
            os.makedirs(conf.BASE_DIR, exist_ok=True)
            conf.RECITER_POOL_FILE.touch()
        st.session_state.reciter_pool = ReciterPool(conf.RECITER_POOL_FILE)

    if 'moshaf_pool' not in st.session_state:
        if not conf.MOSHAF_POOL_FILE.is_file():
            os.makedirs(conf.BASE_DIR, exist_ok=True)
            conf.MOSHAF_POOL_FILE.touch()
        st.session_state.moshaf_pool = MoshafPool(
            reciter_pool=st.session_state.reciter_pool,
            metadata_path=conf.MOSHAF_POOL_FILE,
            download_path=conf.DOWNLOAD_PATH,
            dataset_path=conf.DATASET_PATH)

    if 'switch_to_view_reciters' not in st.session_state:
        st.session_state.switch_to_view_reciters = False

    if 'switch_to_view_moshaf_pool' not in st.session_state:
        st.session_state.switch_to_view_moshaf_pool = False

    if 'switch_to_download_page' not in st.session_state:
        st.session_state.switch_to_download_page = False

    if conf.DOWNLOAD_LOCK_FILE.is_file():
        st.switch_page('pages/download_page.py')


def main():
    menu()
    set_up()
    st.header('Recitation Database Manager')


if __name__ == '__main__':
    main()
