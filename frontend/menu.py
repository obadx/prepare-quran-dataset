import os
import time

import streamlit as st

from utils import (
    save_pools_with_confirmation,
    download_all_moshaf_pool,
    PopupMessage,
)
from prepare_quran_dataset.construct.database import ReciterPool, MoshafPool
import config as conf


def set_up(reset=False) -> None:
    """Initialize ReciterPool and Moshaf_pool"""

    if 'started' not in st.session_state:
        st.session_state.started = True

    if 'reciter_pool' not in st.session_state or reset:
        if not conf.RECITER_POOL_FILE.is_file():
            os.makedirs(conf.BASE_DIR, exist_ok=True)
            conf.RECITER_POOL_FILE.touch()
        st.session_state.reciter_pool = ReciterPool(conf.RECITER_POOL_FILE)

    if 'moshaf_pool' not in st.session_state or reset:
        st.session_state.moshaf_pool = MoshafPool(
            reciter_pool=st.session_state.reciter_pool,
            base_path=conf.BASE_DIR)

    if 'switch_to_view_reciters' not in st.session_state:
        st.session_state.switch_to_view_reciters = False

    if 'switch_to_view_moshaf_pool' not in st.session_state:
        st.session_state.switch_to_view_moshaf_pool = False

    if 'popup_messages' not in st.session_state:
        st.session_state.popup_messages: list[PopupMessage] = []

    if 'refresh_moshaf_filters' not in st.session_state:
        st.session_state.refresh_moshaf_filters = False

    if conf.DOWNLOAD_LOCK_FILE.is_file() or conf.DOWNLOAD_ERROR_LOG.is_file():
        st.switch_page('pages/download_page.py')


def menu():

    st.set_page_config(page_title="Recitation Database Manager", page_icon="📖")
    st.sidebar.page_link(
        'streamlit_app.py', label='Home', icon=':material/home:')
    st.sidebar.page_link(
        'pages/view_reciters_page.py', label='View Reciters', icon="🧔")
    st.sidebar.page_link(
        'pages/view_moshaf_pool_page.py', label='View Moshaf Pool', icon="📖")
    st.sidebar.page_link(
        'pages/insert_reciter_page.py', label='🧔 Insert Reciter', icon=':material/add_circle:')
    st.sidebar.page_link(
        'pages/insert_moshaf_page.py', label='📖 Insert Moshaf Item', icon=':material/add_circle:')
    st.sidebar.page_link(
        'pages/download_page.py', label='📖 Download Page', icon='⬇️')

    if st.sidebar.button('💾 Save Pools', use_container_width=True):
        save_pools_with_confirmation()

    if st.sidebar.button('⬇️  Download All Moshaf Pool', use_container_width=True):
        download_all_moshaf_pool()

    st.sidebar.page_link(
        'pages/get_moshaf_links_page.py', label='Suar Links from Online Websites', icon='🌲')


def menu_with_redirect(reset=False):
    if 'started' not in st.session_state or reset:
        set_up(reset=reset)
        st.switch_page('streamlit_app.py')
    else:
        menu()

    # displaying popupmessages
    if st.session_state.popup_messages:
        for msg in st.session_state.popup_messages:
            msg.show()
        st.session_state.popup_messages = []
