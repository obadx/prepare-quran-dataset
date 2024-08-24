import streamlit as st

from utils import save_pools_with_confirmation, pop_up_message, download_all_moshaf_pool


def style_buttons():
    st.markdown("""
<style>
    .stButton button {
        width: 100%;
        text-align: left;
        padding: 10px;
        background-color: #f0f2f6;
        color: #000000;
        font-weight: normal;
        border: none;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    .stButton button:hover {
        background-color: #e0e2e6;
    }
    .stButton button:focus {
        background-color: #d0d2d6;
        font-weight: bold;
        box-shadow: none;
    }
</style>
    """, unsafe_allow_html=True)


def menu():

    st.set_page_config(page_title="Recitation Database Manager", page_icon="📖")
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


def menu_with_redirect():
    if 'started' not in st.session_state:
        st.switch_page('streamlit_app.py')
    else:
        menu()
