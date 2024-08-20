import os
from pathlib import Path

import streamlit as st

from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf
from prepare_quran_dataset.construct.database import ReciterPool, MoshafPool
from menu import menu


BASE_DIR = Path('DEMO_DIR')
RECITER_POOL_FILE = BASE_DIR / 'reciter_pool.jsonl'
MOSHAF_POOL_FILE = BASE_DIR / 'moshaf_pool.jsonl'
DOWNLOAD_PATH = BASE_DIR / 'Downloads'
DATASET_PATH = BASE_DIR / 'dataset'


def set_up() -> None:
    """Initialize ReciterPool and Moshaf_pool"""

    if 'started' not in st.session_state:
        st.session_state.started = True

    if 'reciter_pool' not in st.session_state:
        if not RECITER_POOL_FILE.is_file():
            os.makedirs(BASE_DIR, exist_ok=True)
            RECITER_POOL_FILE.touch()
        st.session_state.reciter_pool = ReciterPool(RECITER_POOL_FILE)

    if 'moshaf_pool' not in st.session_state:
        if not MOSHAF_POOL_FILE.is_file():
            os.makedirs(BASE_DIR, exist_ok=True)
            MOSHAF_POOL_FILE.touch()
        st.session_state.moshaf_pool = MoshafPool(
            reciter_pool=st.session_state.reciter_pool,
            metadata_path=MOSHAF_POOL_FILE,
            download_path=DOWNLOAD_PATH,
            dataset_path=DATASET_PATH)

    if 'switch_to_view_reciters' not in st.session_state:
        st.session_state.switch_to_view_reciters = False

    if 'switch_to_view_moshaf_pool' not in st.session_state:
        st.session_state.switch_to_view_moshaf_pool = False


def main():
    menu()
    set_up()
    st.header('Recitation Database Manager')

    # insert_reciter_page = st.Page(
    #     "pages/insert_reciter_page.py",
    #     title="Insert Reciter",
    #     icon=":material/add_circle:")
    #
    # view_reciters_page = st.Page(
    #     "pages/view_reciters_page.py",
    #     title="View Reciters",
    #     icon=":material/add_circle:")
    #
    # update_reciter_page = st.Page(
    #     "pages/update_reciter_page.py",
    #     title="Update Reciter",
    #     icon=":material/update:")
    #
    # pg = st.navigation(
    #     [view_reciters_page, insert_reciter_page, update_reciter_page])
    # st.set_page_config(page_title="Recitation Database Manager", page_icon="📖")
    # pg.run()


if __name__ == '__main__':
    main()
