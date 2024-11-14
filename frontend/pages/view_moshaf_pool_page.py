from pathlib import Path

import streamlit as st
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool

from menu import menu_with_redirect
import config as conf
from utils import (
    get_field_name,
    delete_item_from_pool_with_confirmation,
    download_all_moshaf_pool,
    get_arabic_attributes,
    filter_moshaf_pool,
    MoshafFilter,
)


def filter_moshaf_pool_view(
    moshaf_pool: MoshafPool,
    reciter_pool: ReciterPool,
    download_error_file: Path = conf.DOWNLOAD_ERROR_LOG
):
    """Applying filter to chose moshaf using `MoshafFilter`
    """
    if 'chosen_moshaf_list' not in st.session_state:
        st.session_state.chosen_moshaf_list = [m for m in moshaf_pool]

    level1 = st.columns(2)
    level2 = st.columns(2)

    with level1[0]:
        filters_list = st.multiselect(
            'Please Select filter to apply',
            MoshafFilter.get_boolean_fields(),
        )

    with level1[1]:
        reciters = st.multiselect(
            'Please Selsec Reciter',
            [r for r in reciter_pool],
            format_func=lambda r: f"{r.id} / {r.arabic_name}",
        )

    with level2[1]:
        if st.button('Apply', use_container_width=True):
            bools = {
                f: True if f in filters_list else False for f in MoshafFilter.get_boolean_fields()}
            st.session_state.chosen_moshaf_list = filter_moshaf_pool(
                moshaf_pool,
                MoshafFilter(reciters=reciters, **bools),
                download_error_file=download_error_file,
            )


def view_moshaf_pool():
    st.header("View Moshaf Pool")
    filter_moshaf_pool_view(
        st.session_state.moshaf_pool,
        st.session_state.reciter_pool,
    )
    for moshaf in st.session_state.chosen_moshaf_list:
        expander = st.expander(
            f"{moshaf.name} / {moshaf.reciter_arabic_name} / (ID: {moshaf.id})")
        with expander:
            for field, value in moshaf.model_dump().items():
                if field not in conf.EXCLUDED_MSHAF_ITEM_FIELDS_IN_VIEW:
                    label = get_field_name(
                        field, moshaf.model_fields[field])
                    arabic_attributes = get_arabic_attributes(
                        moshaf.model_fields[field])
                    if arabic_attributes:
                        value = arabic_attributes[value]
                    st.markdown(
                        f'<span style="color: orange; font-weight: bold;">{label}: </span>{value}', unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if st.button("Recitations", key=f"play_{moshaf.id}", use_container_width=True, help='Play Recitation Files'):
                    st.session_state.played_moshaf_item = moshaf
                    st.switch_page("pages/play_recitations_page.py")

            with col2:
                if st.button("Update", key=f"update_{moshaf.id}", use_container_width=True):
                    st.session_state.updated_moshaf = moshaf
                    st.switch_page("pages/update_moshaf_page.py")
            with col3:
                if st.button(
                    "Download",
                        key=f"download_{moshaf.id}", use_container_width=True):
                    download_all_moshaf_pool(moshaf_ids=[moshaf.id])

            with col4:
                if st.button(
                    "Refresh",
                    key=f"refresh_{moshaf.id}", use_container_width=True,
                    help='1. Deletes the moshaf_item directory\n'
                    '2. Reload rectiation files from `Downloads` directory\n'
                    '3. Redownload if neccssary\n'
                    '4. This is not a redownload'
                ):
                    download_all_moshaf_pool(
                        moshaf_ids=[moshaf.id], refresh=True)

            with col5:
                if st.button("Delete", key=f"delete_{moshaf.id}", use_container_width=True):
                    delete_item_from_pool_with_confirmation(
                        st.session_state.moshaf_pool, moshaf)

    # if conf.DOWNLOAD_LOCK_FILE.is_file():
    #     st.switch_page('pages/download_page.py')


if __name__ == '__main__':
    # displays sidebar menu & redirect to main page if not initialized
    menu_with_redirect()

    st.session_state.switch_to_view_moshaf_pool = False
    view_moshaf_pool()
