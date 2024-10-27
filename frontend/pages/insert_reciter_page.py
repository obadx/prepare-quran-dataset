import time
import streamlit as st

from prepare_quran_dataset.construct.data_classes import Reciter
from utils import insert_or_update_item_in_pool
from menu import menu_with_redirect


def insert_reciter():
    required_fields = [
        'arabic_name',
        'english_name',
        'country_code'
    ]

    st.header("Insert New Reciter")
    insert_or_update_item_in_pool(
        model=Reciter,
        pool=st.session_state.reciter_pool,
        required_field_names=required_fields,
        key_prefix='reciter_',
        states_values_after_submit={'switch_to_view_reciters': True},
    )

    if st.session_state.switch_to_view_reciters:
        time.sleep(1)  # wait to let the app display the states
        st.switch_page('pages/view_reciters_page.py')


if __name__ == '__main__':
    # displays sidebar menu & redirect to main page if not initialized
    menu_with_redirect()
    insert_reciter()
