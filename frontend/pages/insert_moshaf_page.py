import time
import streamlit as st

from prepare_quran_dataset.construct.data_classes import Moshaf
from utils import insert_or_update_item_in_pool
from menu import menu_with_redirect


def insert_moshaf_item():
    required_fields = [
        'name',
        'reciter_id',
        # 'sources',
        'publisher',
        'comments',
        'rewaya',
        'madd_monfasel_len',
        'madd_mottasel_len',
        'madd_aared_len',
        'madd_mottasel_mahmooz_aared_len',
        'madd_alayn_lazem_len',
        'tasheel_or_madd',
        'daaf_harka',
        'idghaam_nakhlqkm',
        'noon_tamanna',
    ]

    st.header("Insert New Moshf Item")
    insert_or_update_item_in_pool(
        model=Moshaf,
        pool=st.session_state.moshaf_pool,
        required_field_names=required_fields,
        key_prefix='moshaf_',
        states_values_after_submit={'switch_to_view_moshaf_pool': True},
    )

    if st.session_state.switch_to_view_moshaf_pool:
        time.sleep(1)  # wait to let the app display the states
        st.switch_page('pages/view_moshaf_pool_page.py')


# displays sidebar menu & redirect to main page if not initialized
menu_with_redirect()
insert_moshaf_item()
