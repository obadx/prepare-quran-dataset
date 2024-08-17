import streamlit as st

from utils import (
    get_field_name,
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
                if st.button("Update", key=f"update_{reciter.id}"):
                    st.session_state.updated_reciter = reciter
                    st.switch_page("pages/update_reciter_page.py")
            with col2:
                if st.button("Delete", key=f"delete_{reciter.id}"):
                    if st.confirm(f"Are you sure you want to delete {reciter.arabic_name}?"):
                        st.session_state.reciter_pool.delete(reciter.id)
                        st.success("Reciter deleted successfully")


st.session_state.switch_to_view_reciters = False
view_reciters()
