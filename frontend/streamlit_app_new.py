from pathlib import Path
import os
import re
from typing import Type, Literal, Optional, Any, get_args, get_origin

import streamlit as st
from pydantic import BaseModel
from pydantic.fields import FieldInfo, PydanticUndefined

from prepare_quran_dataset.construct.database import ReciterPool, MoshafPool, Pool
from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf


BASE_DIR = Path('DEMO_DIR')
RECITER_POOL_FILE = BASE_DIR / 'reciter_pool.jsonl'
MOSHAF_POOL_FILE = BASE_DIR / 'moshaf_pool.jsonl'
DOWNLOAD_PATH = BASE_DIR / 'Downloads'
DATASET_PATH = BASE_DIR / 'dataset'


def main():
    set_up()
    st.title("Quran Dataset Management")

    st.markdown("""
<style>
    .stButton button {
        width: 100%;
        text-align: left;
        padding: 10px;
        background-color: #f0f2f6;
        color: #000000;
        font-weight: normal;
        border: none;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    .stButton button:hover {
        background-color: #e0e2e6;
    }
    .stButton button:focus {
        background-color: #d0d2d6;
        font-weight: bold;
        box-shadow: none;
    }
</style>
    """, unsafe_allow_html=True)
    menu = [
        "View Reciters",
        "View Moshafs",
        "Insert Reciter",
        "Insert Moshaf",
        "Save Pools",
        "Download All MoshafPool",
    ]
    # choice = st.sidebar.selectbox("Select Operation", menu)

    # Initialize session state to store the current choice
    if 'choice' not in st.session_state:
        st.session_state.choice = menu[0]

    # Create buttons for each menu item
    for item in menu:
        if st.sidebar.button(item, f"menu_{item}"):
            st.session_state.choice = item

    match st.session_state.choice:
        case "View Reciters":
            view_reciters()
        case "View Moshafs":
            view_moshafs()
        case "Insert Reciter":
            insert_reciter()
        case "Insert Moshaf":
            insert_moshaf()
        case "Save Pools":
            save_pools()
        case "Download All MoshafPool":
            download_all_moshaf_pool()


def set_up() -> None:
    """Initialize ReciterPool and Moshaf_pool"""
    if 'reciter_pool' not in st.session_state:
        if not RECITER_POOL_FILE.is_file():
            os.makedirs(BASE_DIR, exist_ok=True)
            RECITER_POOL_FILE.touch()
        st.session_state.reciter_pool = ReciterPool(RECITER_POOL_FILE)

    if 'moshaf_pool' not in st.session_state:
        if not MOSHAF_POOL_FILE.is_file():
            os.makedirs(BASE_DIR, exist_ok=True)
            MOSHAF_POOL_FILE.touch()
        st.session_state.moshaf_pool = MoshafPool(
            reciter_pool=st.session_state.reciter_pool,
            metadata_path=MOSHAF_POOL_FILE,
            download_path=DOWNLOAD_PATH,
            dataset_path=DATASET_PATH)


def download_all_moshaf_pool():
    ...


def save_pools():
    save_pools_confirmation()


@st.dialog("Save Pools?")
def save_pools_confirmation():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", key="confirm_save"):
            try:
                st.session_state.reciter_pool.save()
                st.session_state.moshaf_pool.save()
                st.success("All data saved successfully!")
            except Exception as e:
                st.error(f"Error saving data: {str(e)}")
    with col2:
        if st.button("No", key="cancel_save"):
            st.info("Save operation cancelled.")


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
                    update_reciter(reciter)
            with col2:
                if st.button("Delete", key=f"delete_{reciter.id}"):
                    if st.confirm(f"Are you sure you want to delete {reciter.arabic_name}?"):
                        st.session_state.reciter_pool.delete(reciter.id)
                        st.success("Reciter deleted successfully")


def view_moshafs():
    ...
    # st.header("View Moshafs")
    # for moshaf in st.session_state.moshaf_pool:
    #     expander = st.expander(f"ID: {moshaf.id}, Name: {moshaf.name}, Reciter ID: {
    #         moshaf.reciter_id}, Reciter Name: {moshaf.reciter_arabic_name}")
    #     with expander:
    #         for field, value in moshaf.model_dump().items():
    #             label = get_field_name(
    #                 field, st.session_state.moshaf_pool.model_fields[field])
    #             st.write(f"{label}: {value}")
    #
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             if st.button("Update", key=f"update_{moshaf.id}"):
    #                 st.session_state.update_moshaf = moshaf
    #                 st.rerun()
    #         with col2:
    #             if st.button("Delete", key=f"delete_{moshaf.id}"):
    #                 if st.confirm(f"Are you sure you want to delete this Moshaf?"):
    #                     st.session_state.moshaf_pool.delete(moshaf.id)
    #                     st.success("Moshaf deleted successfully")
    #                     st.experimental_rerun()


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
        required_field_names=required_fields
    )


def insert_moshaf():
    required_fields = [
    ]
    insert_or_update_item_in_pool(
        model=Moshaf,
        pool=st.session_state.moshaf_pool,
        required_field_names=required_fields
    )


def update_reciter(reciter: Reciter):
    required_fields = [
        'arabic_name',
        'english_name',
        'country_code',
    ]

    st.subheader(f"Update Reciter (ID={reciter.id}, {reciter.arabic_name})")
    insert_or_update_item_in_pool(
        model=Reciter,
        pool=st.session_state.reciter_pool,
        required_field_names=required_fields,
        old_item=reciter,
    )


def insert_or_update_item_in_pool(
    model: Type[BaseModel],
    pool: Pool,
    required_field_names: list[str],
    old_item: Optional[BaseModel] = None,
):

    with st.form("insert_or_update_form"):
        form_data = {}
        for field_name in required_field_names:
            field_info = model.model_fields[field_name]
            field_input = create_input_for_field(
                field_name, field_info,
                default_value=old_item.model_dump()[field_name] if old_item else None)
            if field_input is not None:
                form_data[field_name] = field_input

        if st.form_submit_button(f"Confirm {model.__name__}"):
            try:
                new_item = model(**form_data)
                print(new_item)
                if old_item is None:
                    pool.insert(new_item)
                else:
                    for field_name in required_field_names:
                        setattr(
                            old_item, field_name, getattr(new_item. field_name))
                    print(old_item)
                    pool.update(old_item)
                st.success("Operation is successfully!")
            except Exception as e:
                st.error(f"Error in the update/insert: {str(e)}")


def create_input_for_field(
    field_name: str,
    field_info: FieldInfo,
    default_value=None,
):

    # Extract Arabic name from field description if available
    label = get_field_name(field_name, field_info)

    if default_value is None:
        if field_info.default != PydanticUndefined:
            default_value = field_info.default

    # the args of a Literal typs > 0 EX: Literal[3, 4]
    if get_origin(field_info.annotation) is Literal:
        choices = list(get_args(field_info.annotation))
        return st.selectbox(label, choices, index=choices.index(default_value) if default_value else 0)

    if field_info.annotation == str:
        return st.text_input(label, value=default_value or "")
    elif field_info.annotation == int:
        return st.number_input(label, value=default_value or 0, step=1)
    elif field_info.annotation == str:
        return st.number_input(label, value=default_value or 0.0, step=0.1)
    elif field_info.annotation == bool:
        return st.checkbox(label, value=default_value or False)

    st.warning(f"Unsupported field type for {label}: {field_info.annotaion}")
    return None


def get_arabic_name(field_info: FieldInfo) -> str:
    """get the Arabic name out of the field description
    """
    if field_info.description:
        match = re.search(
            r'ArabicName\(\s*(\w+(?:\s+\w+)*)\s*\)',
            field_info.description, re.UNICODE)
        if match:
            return match.group(1)
    return ''


def get_field_name(field_name: str, field_info: FieldInfo) -> str:
    """Recturn the Arabic name of the field if applicable else the field_name
    """
    label = field_name
    arabic_name = get_arabic_name(field_info)
    if arabic_name:
        label = f"{arabic_name} ({field_name})"

    return label


if __name__ == "__main__":
    main()
