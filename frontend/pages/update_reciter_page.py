import time

import streamlit as st

from prepare_quran_dataset.construct.data_classes import Reciter
from prepare_quran_dataset.construct.base import Pool
from utils import insert_or_update_item_in_pool
from menu import menu_with_redirect


def after_reciter_update(pool: Pool, reciter: Reciter):
    """update the reciter related field into every Moshaf
    which are: `reciter_arabic_name`, `reciter_english_name`
    """
    for moshaf_id in reciter.moshaf_set_ids:
        moshaf = st.session_state.moshaf_pool[moshaf_id]
        moshaf.reciter_arabic_name = reciter.arabic_name
        moshaf.reciter_english_name = reciter.english_name

        st.session_state.moshaf_pool.update(moshaf)


def update_reciter():
    required_fields = [
        'arabic_name',
        'english_name',
        'country_code',
    ]

    if 'updated_reciter' in st.session_state:
        st.subheader(
            f"Update Reciter (ID={st.session_state.updated_reciter.id}, {st.session_state.updated_reciter.arabic_name})")
        insert_or_update_item_in_pool(
            model=Reciter,
            pool=st.session_state.reciter_pool,
            required_field_names=required_fields,
            key_prefix='reciter_',
            item_name_in_session_state='updated_reciter',
            states_values_after_submit={'switch_to_view_reciters': True},
            after_submit_callback=after_reciter_update,
        )
    else:
        st.error('No Reciter No Update')

    if st.session_state.switch_to_view_reciters:
        time.sleep(1)  # wait to let the app display the states
        st.switch_page('pages/view_reciters_page.py')


if __name__ == '__main__':
    # displays sidebar menu & redirect to main page if not initialized
    menu_with_redirect()
    update_reciter()
