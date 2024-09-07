import os

import streamlit as st

from utils import save_pools_with_confirmation, pop_up_message, download_all_moshaf_pool
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

    if conf.DOWNLOAD_LOCK_FILE.is_file():
        st.switch_page('pages/download_page.py')


# def style_buttons():
#     st.markdown("""
# <style>
#     .stButton button {
#         width: 100%;
#         text-align: left;
#         padding: 10px;
#         background-color: #f0f2f6;
#         color: #000000;
#         font-weight: normal;
#         border: none;
#         border-radius: 4px;
#         margin-bottom: 5px;
#     }
#     .stButton button:hover {
#         background-color: #e0e2e6;
#     }
#     .stButton button:focus {
#         background-color: #d0d2d6;
#         font-weight: bold;
#         box-shadow: none;
#     }
# </style>
#     """, unsafe_allow_html=True)


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

    st.sidebar.button(
        '💾 Save Pools', on_click=save_pools_with_confirmation, use_container_width=True)
    st.sidebar.button(
        '⬇️  Download All Moshaf Pool', on_click=download_all_moshaf_pool, use_container_width=True)


def menu_with_redirect(reset=False):
    if 'started' not in st.session_state or reset:
        set_up(reset=reset)
        st.switch_page('streamlit_app.py')
    else:
        menu()
