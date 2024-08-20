import time

import streamlit as st

from prepare_quran_dataset.construct.database import Moshaf
from utils import insert_or_update_item_in_pool
from menu import menu_with_redirect


def update_moshaf_item():
    required_fields = [
        'name',
        'reciter_id',
    ]

    if 'updated_moshaf' in st.session_state:
        st.subheader(
            f"Update Moshaf (ID={st.session_state.updated_moshaf.id}, Reciter={st.session_state.updated_moshaf.reciter_arabic_name})")
        insert_or_update_item_in_pool(
            model=Moshaf,
            pool=st.session_state.moshaf_pool,
            required_field_names=required_fields,
            key_prefix='moshaf_',
            item_name_in_session_state='updated_moshaf',
            states_values_after_submit={'switch_to_view_moshaf_pool': True},
        )
    else:
        st.error('No Reciter No Update')

    if st.session_state.switch_to_view_reciters:
        time.sleep(1)  # wait to let the app display the states
        st.switch_page('pages/view_reciters_page.py')


# displays sidebar menu & redirect to main page if not initialized
menu_with_redirect()
update_moshaf_item()
