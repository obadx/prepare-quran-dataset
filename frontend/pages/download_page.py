import time
import json

import streamlit as st

import config as conf
from menu import menu_with_redirect
from utils import get_download_lock_log, cancel_download_with_confirmation


def show_error():
    """displays errors within the download page"""
    with open(conf.DOWNLOAD_ERROR_LOG, 'r') as f:
        error_log = json.load(f)
    for id, error in error_log.items():
        st.error(f'Error while downloading Moshaf ID={id}')
        st.code(error, language='log')


def show_progress():
    """shows progress while downloading"""
    log = get_download_lock_log(conf.DOWNLOAD_LOCK_FILE)
    if log.total_count == 1:
        st.info(f'Downloading Moshaf ID={log.current_moshaf_id}')
        st.markdown(
            '<img src="https://upload.wikimedia.org/wikipedia/commons/a/ad/YouTube_loading_symbol_3_%28transparent%29.gif" alt="Girl in a jacket" width="50" >', unsafe_allow_html=True)
    else:
        st.progress(log.finished_count / log.total_count,
                    text=f'Downloading Moshaf ID={log.current_moshaf_id}, '
                    f'finished: {log.finished_moshaf_ids}, Reamaining: '
                    f'{set(log.moshaf_ids) -
                       set(log.finished_moshaf_ids) - set(log.error_ids)}, '
                    f'Error IDs: {log.error_ids}'
                    )
        st.markdown(
            '<img src="https://upload.wikimedia.org/wikipedia/commons/a/ad/YouTube_loading_symbol_3_%28transparent%29.gif" alt="Girl in a jacket" width="50" >', unsafe_allow_html=True)


def download_page():
    st.header('Download Moshaf Page')

    if conf.DOWNLOAD_LOCK_FILE.is_file():
        show_progress()

        if conf.DOWNLOAD_ERROR_LOG.is_file():
            show_error()

        # cancel download
        if st.button('Cancel Download'):
            st.session_state.cancel_download = True

        # to recheck if the download if finished
        # we need NO st.rerun in cancelation because rerun stops the
        # cancelation function and start over again
        if 'cancel_download' in st.session_state:
            cancel_download_with_confirmation(
                get_download_lock_log(conf.DOWNLOAD_LOCK_FILE).process_pid)
            del st.session_state['cancel_download']
        else:
            time.sleep(2)
            st.rerun()

    elif conf.DOWNLOAD_ERROR_LOG.is_file():
        show_error()

    else:
        st.info('No Moshaf to download')
        time.sleep(2)
        menu_with_redirect(reset=True)


if __name__ == '__main__':
    # displays sidebar menu & redirect to main page if not initialized
    menu_with_redirect()

    download_page()
