import time

import streamlit as st

import config as conf
from menu import menu_with_redirect
from utils import get_download_lock_log


# BUG: switch to download page after download
def download_page():
    st.header('Download Moshaf Page')

    if conf.DOWNLOAD_ERROR_LOG.is_file():
        with open(conf.DOWNLOAD_ERROR_LOG, 'r') as f:
            st.error('Download Error')
            st.code(f.read(), language='log')

    elif conf.DOWNLOAD_LOCK_FILE.is_file():
        log = get_download_lock_log(conf.DOWNLOAD_LOCK_FILE)
        if log.total_count == 1:
            st.info(f'Downloading Moshaf ID={log.current_moshaf_id}')
            st.markdown(
                '<img src="https://upload.wikimedia.org/wikipedia/commons/a/ad/YouTube_loading_symbol_3_%28transparent%29.gif" alt="Girl in a jacket" width="50" >', unsafe_allow_html=True)
        else:
            st.progress(log.finished_count / log.total_count,
                        text=f'Downloading Moshaf ID={log.current_moshaf_id}, '
                        f'finished: {log.finished_moshaf_ids}, reamaining: '
                        f'{set(log.moshaf_ids) - set(log.finished_moshaf_ids)}'
                        )
            st.markdown(
                '<img src="https://upload.wikimedia.org/wikipedia/commons/a/ad/YouTube_loading_symbol_3_%28transparent%29.gif" alt="Girl in a jacket" width="50" >', unsafe_allow_html=True)

        # to recheck if the download if finished
        time.sleep(2)
        st.rerun()

    else:
        st.info('No Moshaf to download')
        time.sleep(2)
        menu_with_redirect(reset=True)


# displays sidebar menu & redirect to main page if not initialized
menu_with_redirect()

download_page()
