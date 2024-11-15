import streamlit as st

from menu import menu_with_redirect

from utils import (
    get_field_name,
    delete_item_from_pool_with_confirmation,
    popup_message,
)


def view_reciters():
    st.header("View Reciters")
    for reciter in st.session_state.reciter_pool:
        expander = st.expander(f"{reciter.arabic_name} (ID: {reciter.id})")
        with expander:
            for field, value in reciter.model_dump().items():
                label = get_field_name(
                    field, reciter.model_fields[field])
                st.markdown(
                    f'<span style="color: orange; font-weight: bold;">{label}: </span>{value}', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update", key=f"update_{reciter.id}", use_container_width=True):
                    st.session_state.updated_reciter = reciter
                    st.switch_page("pages/update_reciter_page.py")
            with col2:
                if st.button("Delete", key=f"delete_{reciter.id}", use_container_width=True):
                    if len(reciter.moshaf_set_ids) > 0:
                        popup_message(
                            (
                                f'Can not **delete** reciter: **{reciter.id} / {reciter.arabic_name}** as it has the following moshaf ids: **{reciter.moshaf_set_ids}**'),
                            'warn')
                    else:
                        delete_item_from_pool_with_confirmation(
                            st.session_state.reciter_pool, reciter)


if __name__ == '__main__':
    # displays sidebar menu & redirect to main page if not initialized
    menu_with_redirect()

    st.session_state.switch_to_view_reciters = False
    view_reciters()
