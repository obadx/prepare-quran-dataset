import streamlit as st

from menu import menu_with_redirect

from utils import (
    get_field_name,
    delete_item_from_pool_with_confirmation,
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

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Update", key=f"update_{moshaf.id}", use_container_width=True):
                    st.session_state.updated_moshaf = moshaf
                    st.switch_page("pages/update_moshaf_page.py")
            with col2:
                if st.button("Download", key=f"download_{moshaf.id}", use_container_width=True):
                    pass

            with col3:
                st.button(
                    "Delete", key=f"delete_{moshaf.id}", use_container_width=True,
                    on_click=delete_item_from_pool_with_confirmation,
                    kwargs={'pool': st.session_state.moshaf_pool, 'item': moshaf})


# displays sidebar menu & redirect to main page if not initialized
menu_with_redirect()

st.session_state.switch_to_view_moshaf_pool = False
view_moshaf_pool()
