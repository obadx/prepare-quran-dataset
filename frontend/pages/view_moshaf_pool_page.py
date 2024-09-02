import streamlit as st

from menu import menu_with_redirect
import config as conf

from utils import (
    get_field_name,
    delete_item_from_pool_with_confirmation,
    download_single_msohaf,
)


def view_moshaf_pool():
    st.header("View Moshaf Pool")
    for moshaf in st.session_state.moshaf_pool:
        expander = st.expander(
            f"{moshaf.name} / {moshaf.reciter_arabic_name} / (ID: {moshaf.id})")
        with expander:
            for field, value in moshaf.model_dump().items():
                label = get_field_name(
                    field, moshaf.model_fields[field])
                st.markdown(
                    f'<span style="color: orange; font-weight: bold;">{label}: </span>{value}', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Update", key=f"update_{moshaf.id}", use_container_width=True):
                    st.session_state.updated_moshaf = moshaf
                    st.switch_page("pages/update_moshaf_page.py")
            with col2:
                st.button(
                    "Download",
                    key=f"download_{moshaf.id}", use_container_width=True,
                    on_click=download_single_msohaf,
                    args=(moshaf.id,),
                )
            with col3:
                st.button(
                    "Refresh",
                    key=f"refresh_{moshaf.id}", use_container_width=True,
                    on_click=download_single_msohaf,
                    args=(moshaf.id,),
                    kwargs={'refresh': True},
                    help='1. Deletes the moshaf_item directory\n'
                    '2. Reload rectiation files from `Downloads` directory\n'
                    '3. Redownload if neccssary\n'
                    '4. This is not a redownload'

                )

            with col4:
                st.button(
                    "Delete", key=f"delete_{moshaf.id}", use_container_width=True,
                    on_click=delete_item_from_pool_with_confirmation,
                    kwargs={'pool': st.session_state.moshaf_pool, 'item': moshaf})

    if conf.DOWNLOAD_LOCK_FILE.is_file():
        st.switch_page('pages/download_page.py')


# displays sidebar menu & redirect to main page if not initialized
menu_with_redirect()

st.session_state.switch_to_view_moshaf_pool = False
view_moshaf_pool()
