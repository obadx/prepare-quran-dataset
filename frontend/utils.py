import re
from typing import Type, Literal, Optional, Any, Callable, get_args, get_origin
import time
import multiprocessing
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

import streamlit as st
from pydantic import BaseModel
from pydantic.fields import FieldInfo, PydanticUndefined

from prepare_quran_dataset.construct.database import Pool, MoshafPool
from prepare_quran_dataset.construct.utils import get_suar_list
import config as conf


@dataclass
class DownloadLog:
    current_moshaf_id: str
    finished_count: int
    total_count: int
    moshaf_ids: list[str]
    finished_moshaf_ids: list[str] = field(default_factory=[])


@st.cache_data
def get_suar_names() -> list[str]:
    return get_suar_list()


def write_to_download_lock_log(log: DownloadLog, filepath: Path):
    with open(filepath, 'w+') as f:
        json.dump(asdict(log), f)


def get_download_lock_log(filepath: Path) -> DownloadLog:
    with open(filepath, 'r') as f:
        log = json.load(f)
    return DownloadLog(**log)


def download_all_moshaf_pool():
    """Download All moshaf pool callback
    """
    def download_all_moshaf_task(moshaf_pool: MoshafPool, to_download_ids: list[str], lockfile_path: Path):
        finished_ids = []
        for id in to_download_ids:
            log = DownloadLog(
                current_moshaf_id=id,
                finished_count=len(finished_ids),
                total_count=len(to_download_ids),
                finished_moshaf_ids=finished_ids,
                moshaf_ids=to_download_ids,
            )
            write_to_download_lock_log(log, lockfile_path)
            moshaf_pool.download_moshaf(
                id, refresh=False, save_on_disk=True)
            finished_ids.append(id)

        # End of download -> delete the download_lockfile
        lockfile_path.unlink()

    if not conf.DOWNLOAD_LOCK_FILE.is_file():
        to_download_ids = [
            m.id for m in st.session_state.moshaf_pool if not m.is_downloaded]
        if len(to_download_ids) == 0:
            pop_up_message('All Moshaf Pool is downloaded', 'info')
            return
        p = multiprocessing.Process(
            target=download_all_moshaf_task,
            args=(st.session_state.moshaf_pool, to_download_ids, conf.DOWNLOAD_LOCK_FILE))
        p.start()  # Start the process
    else:
        pop_up_message('There is a download is already running...', 'warn')


def download_single_msohaf(moshaf_id: str, refresh=False):
    def download_moshaf_task(moshaf_pool: MoshafPool, moshaf_id: str, lockfile_path: Path):
        log = DownloadLog(
            current_moshaf_id=moshaf_id,
            finished_count=0,
            total_count=1,
            finished_moshaf_ids=[],
            moshaf_ids=[moshaf_id]
        )
        write_to_download_lock_log(log, lockfile_path)
        moshaf_pool.download_moshaf(
            moshaf_id, save_on_disk=True, refresh=refresh)
        # delete the download_lockfile
        lockfile_path.unlink()

    if not conf.DOWNLOAD_LOCK_FILE.is_file():
        moshaf_item = st.session_state.moshaf_pool[moshaf_id]
        if moshaf_item.is_downloaded and not refresh:
            pop_up_message('The moshaf is already Dwonloaded', 'info')
            return
        p = multiprocessing.Process(
            target=download_moshaf_task,
            args=(st.session_state.moshaf_pool, moshaf_id, conf.DOWNLOAD_LOCK_FILE))
        p.start()  # Start the process
    else:
        pop_up_message('There is a download is already running...', 'warn')


@st.dialog("Delete Item?")
def delete_item_from_pool_with_confirmation(pool: Pool, item: BaseModel):
    st.warning(f'Are you want to to delete ID={item.id} ?')
    col1, col2 = st.columns(2)
    placeholder = st.empty()

    with placeholder.container():
        with col1:
            if st.button("Yes", use_container_width=True,):
                delete_item_from_pool(pool, item, placeholder)
                time.sleep(1)
                st.rerun()
        with col2:
            if st.button("No", use_container_width=True):
                placeholder.info("Delete Operation is canceled")
                time.sleep(1)
                st.rerun()


def delete_item_from_pool(pool: Pool, item: BaseModel, placeholder):
    try:
        pool.delete(item.id)
        placeholder.success(
            f"ID={item.id} is deleted successfully")
    except Exception as e:
        placeholder.error(f"Error Deleteing item: {str(e)}", 'error')
        raise e


@st.dialog("Save Pools?")
def save_pools_with_confirmation():
    col1, col2 = st.columns(2)
    placeholder = st.empty()

    with placeholder.container():
        with col1:
            if st.button("Yes", use_container_width=True,):
                save_pools(placeholder)
        with col2:
            if st.button("No", use_container_width=True):
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
        raise e


def pop_up_message(msg: str, msg_type: str = 'success'):
    @ st.dialog(msg)
    def display_popup():
        match msg_type:
            case 'success':
                st.success(msg)
            case 'error':
                st.error(msg)
            case 'warn':
                st.warning(msg)
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
    field_funcs_after_submit: Optional[dict[str, Callable[[Any], Any]]] = {},
):
    """
    Args:
        field_funcs_after_submit ( Optional[dict[str, Callable[[Any], Any]]]):
            dict[str, func], dict to aggregate field value after from
            submition, Ex: {'id': lambda x: str(x)}
    """

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
                'field_funcs_after_submit': field_funcs_after_submit,
            }
        )


def insert_update_form_submit(
    model: Type[BaseModel] = None,
    pool: Pool = None,
    key_prefix='model_',
    required_field_names: list[str] = [],
    item_name_in_session_state: str = None,
    states_values_after_submit: Optional[dict[str, Any]] = {},
    field_funcs_after_submit: Optional[dict[str, Callable[[Any], Any]]] = {},
):
    """
    Args:
        field_funcs_after_submit ( Optional[dict[str, Callable[[Any], Any]]]):
            dict[str, func], dict to aggregate field value after from
            submition, Ex: {'id': lambda x: str(x)}
    """

    # Saveing values in dict
    form_data = {}
    for field_name in required_field_names:
        if field_name in field_funcs_after_submit:
            func = field_funcs_after_submit[field_name]
            form_data[field_name] = func(
                st.session_state[key_prefix + field_name])
        else:
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

    if field_name == 'reciter_id':
        reciters = {r.id: r for r in st.session_state.reciter_pool}
        reciters_ids = list(reciters.keys())
        return st.selectbox(
            label,
            reciters_ids,
            key=key,
            format_func=lambda id: f'{reciters[id].arabic_name}, ID={id}',
            index=reciters_ids.index(
                default_value) if default_value is not None else 0,
        )

    # the args of a Literal typs > 0 EX: Literal[3, 4]
    if get_origin(field_info.annotation) is Literal:
        choices = list(get_args(field_info.annotation))
        arabic_attributes = get_arabic_attributes(field_info)
        return st.selectbox(
            label,
            choices,
            format_func=lambda x: arabic_attributes[x] if arabic_attributes else x,
            index=choices.index(default_value) if default_value else 0,
            key=key,
        )

    if field_info.annotation in [str, Optional[str]]:
        return st.text_input(label, value=default_value or "", key=key)
    elif field_info.annotation in [int, Optional[int]]:
        return st.number_input(label, value=default_value or 0, step=1, key=key)
    elif field_info.annotation in [float, Optional[float]]:
        return st.number_input(label, value=default_value or 0.0, step=0.1, key=key)
    elif field_info.annotation in [bool, Optional[bool]]:
        return st.checkbox(label, value=default_value or False, key=key)
    elif field_info.annotation in [list[str], Optional[list[str]]]:
        return st.text_area(
            label,
            key=key,
            value='\n'.join(default_value) if default_value else '',
        )
    elif field_info.annotation in [dict[str, str], Optional[dict[str, str]]]:
        help = (
            'Place the specifc downloads as a '
            f'JSON\n{json.dumps({'002': 'https://example.com/003.mp3'}, indent=4)}')
        return st.text_area(
            label,
            key=key,
            value=json.dumps(default_value, indent=4) if default_value else '',
            help=help,
            placeholder=help,
        )

    raise ValueError(
        f"Unsupported field type for {label}: {field_info.annotation}")


def get_arabic_name(field_info: FieldInfo) -> str:
    """get the Arabic name out of the field description
    """
    if field_info.description:
        match = re.search(
            r'ArabicName\((.*)\)',
            field_info.description, re.UNICODE)
        if match:
            return match.group(1)
    return ''


def get_arabic_attributes(field_info: FieldInfo) -> dict[str, str] | None:
    """get the Arabic attributes for `Literal` type fields

    Returns:
        dict[str: str] if found else `None`
    """
    if field_info.description:
        match = re.search(
            r'ArabicAttr\((.*?)\)',
            field_info.description, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    return None


def get_field_name(field_name: str, field_info: FieldInfo) -> str:
    """Recturn the Arabic name of the field if applicable else the field_name
    """
    label = field_name
    arabic_name = get_arabic_name(field_info)
    if arabic_name:
        label = f"{arabic_name} ({field_name})"

    return label
