import re
from typing import Type, Literal, Optional, Any, Callable, get_args, get_origin
import time

import streamlit as st
from pydantic import BaseModel
from pydantic.fields import FieldInfo, PydanticUndefined

from prepare_quran_dataset.construct.database import Pool


@st.dialog("Delete Item")
def delete_item():
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


@st.dialog("Save Pools?")
def save_pools_with_confirmation():
    col1, col2 = st.columns(2)
    placeholder = st.empty()

    with placeholder.container():
        with col1:
            if st.button("Yes", use_container_width=True,):
                save_pools(placeholder)
        with col2:
            if st.button("No", key='save_canceled', use_container_width=True):
                placeholder.info("Save Operation is canceled")
                time.sleep(1)
                st.rerun()


def save_pools(placeholder):
    try:
        st.session_state.reciter_pool.save()
        st.session_state.moshaf_pool.save()

        placeholder.success("All data saved successfully!")
    except Exception as e:
        placeholder.error(f"Error saving data: {str(e)}", 'error')


def pop_up_message(msg: str, msg_type: str = 'success'):
    @ st.dialog(msg)
    def display_popup():
        match msg_type:
            case 'success':
                st.success(msg)
            case 'error':
                st.error(msg)
            case 'info':
                st.info(msg)
            case _:
                raise ValueError(
                    "Not valid selection, vlaid selections ['success', 'error', 'info']")

    display_popup()


def insert_or_update_item_in_pool(
    model: Type[BaseModel],
    pool: Pool,
    required_field_names: list[str],
    key_prefix='model_',
    item_name_in_session_state: str = None,
    states_values_after_submit: Optional[dict[str, Any]] = {},
):

    with st.form("insert_or_update_form"):
        for field_name in required_field_names:
            field_info = model.model_fields[field_name]
            create_input_for_field(
                field_name, field_info, key_prefix=key_prefix,
                default_value=getattr(st.session_state[item_name_in_session_state], field_name) if item_name_in_session_state else None)

        st.form_submit_button(
            f"Confirm {model.__name__}",
            on_click=insert_update_form_submit,
            kwargs={
                'model': model,
                'pool': pool,
                'key_prefix': key_prefix,
                'required_field_names': required_field_names,
                'item_name_in_session_state': item_name_in_session_state,
                'states_values_after_submit': states_values_after_submit,
            }
        )


def insert_update_form_submit(
    model: Type[BaseModel] = None,
    pool: Pool = None,
    key_prefix='model_',
    required_field_names: list[str] = [],
    item_name_in_session_state: str = None,
    states_values_after_submit: Optional[dict[str, Any]] = {},
):
    # Saveing values in dict
    form_data = {}
    for field_name in required_field_names:
        form_data[field_name] = st.session_state[key_prefix + field_name]

    try:
        # Insertion Operation
        if item_name_in_session_state is None:
            new_item = model(**form_data)
            pool.insert(new_item)
            pop_up_message("Insertion is successfully!", msg_type='success')

        # Update Operation
        else:
            for field_name, val in form_data.items():
                setattr(
                    st.session_state[item_name_in_session_state], field_name, val)
            pool.update(st.session_state[item_name_in_session_state])
            pop_up_message("Update is successfully!", msg_type='success')
    except Exception as e:
        st.error(f"Error in the update/insert: {str(e)}")
        raise e

    # cleaning session_state after ooperation
    if item_name_in_session_state is not None:
        del st.session_state[item_name_in_session_state]
    for field_name in required_field_names:
        del st.session_state[key_prefix + field_name]

    for key, val in states_values_after_submit.items():
        st.session_state[key] = val


def create_input_for_field(
    field_name: str,
    field_info: FieldInfo,
    default_value=None,
    key_prefix='model_',
):

    # Extract Arabic name from field description if available
    label = get_field_name(field_name, field_info)

    if default_value is None:
        if field_info.default != PydanticUndefined:
            default_value = field_info.default

    key = key_prefix + field_name

    # the args of a Literal typs > 0 EX: Literal[3, 4]
    if get_origin(field_info.annotation) is Literal:
        choices = list(get_args(field_info.annotation))
        return st.selectbox(
            label,
            choices,
            index=choices.index(default_value) if default_value else 0,
            key=key,
        )

    if field_info.annotation == str:
        return st.text_input(label, value=default_value or "", key=key)
    elif field_info.annotation == int:
        return st.number_input(label, value=default_value or 0, step=1, key=key)
    elif field_info.annotation == str:
        return st.number_input(label, value=default_value or 0.0, step=0.1, key=key)
    elif field_info.annotation == bool:
        return st.checkbox(label, value=default_value or False, key=key)

    raise ValueError(
        f"Unsupported field type for {label}: {field_info.annotaion}")


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
