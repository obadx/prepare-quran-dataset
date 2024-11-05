import re
from typing import Type, Literal, Optional, Any, Callable, get_args, get_origin
import time
import multiprocessing
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import json
import traceback
import os
import copy

import streamlit as st
from pydantic import BaseModel
from pydantic.fields import FieldInfo, PydanticUndefined

from prepare_quran_dataset.construct.database import Pool, MoshafPool
from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.utils import get_suar_list, kill_process
import config as conf


@dataclass
class MoshafFilter:
    complete: bool = True
    not_complete: bool = True
    downloaded: bool = True
    not_downloaded: bool = True
    download_error: bool = True
    not_download_error: bool = True
    reciters: list[Reciter] = None

    @classmethod
    def get_boolean_fields(cls) -> list[str]:
        # Filter and list field names with type bool
        return [field.name for field in fields(cls) if field.type == bool]


def filter_moshaf_pool(
    moshaf_pool: MoshafPool,
    filter: MoshafFilter,
    download_error_file: Path = conf.DOWNLOAD_ERROR_LOG
) -> list[Moshaf]:
    """Filter Moshaf with moshaf pool with filters see MoshafFilter
        Default is to chose all moshaf

        When a filter with its complement is set or not i.e (0, 0) or (1, 1)
        the element is chosen
    """
    def chose(chose, not_chose, var) -> bool:
        """
        | not_chose | chose | var || Ouput |
        | ----------|-------|-----||-------|
        |     0     |   0   |   0 ||   1   |
        |     0     |   0   |   1 ||   1   |
        |     0     |   1   |   0 ||   0   |
        |     0     |   1   |   1 ||   1   |
        |     1     |   0   |   0 ||   1   |
        |     1     |   0   |   1 ||   0   |
        |     1     |   1   |   0 ||   1   |
        |     1     |   1   |   1 ||   1   |
        """
        return (
            (chose and not_chose) or
            (not chose and not var) or
            (not not_chose and var))

    comp_flag = True
    selected_moshaf = []
    reciter_ids = {r.id for r in filter.reciters} if filter.reciters else None
    error_ids = set()
    download_error_file = Path(download_error_file)
    if download_error_file.is_file():
        with open(download_error_file, 'r') as f:
            error_log = json.load(f)
            error_ids = set(error_log.keys())

    for moshaf in moshaf_pool:
        # complete nd not_complete
        comp_flag = comp_flag and chose(
            filter.complete, filter.not_complete, moshaf.is_complete)

        # download and not download
        comp_flag = comp_flag and chose(
            filter.downloaded, filter.not_downloaded, moshaf.is_downloaded)

        # downlaod_error
        if error_ids:
            comp_flag = comp_flag and chose(
                filter.download_error, filter.not_download_error, moshaf.id in error_ids)

        # selected reciters
        if reciter_ids:
            comp_flag = comp_flag and (moshaf.reciter_id in reciter_ids)

        if comp_flag:
            selected_moshaf.append(moshaf)
    return selected_moshaf


@dataclass
class DownloadLog:
    process_pid: int
    current_moshaf_id: str
    finished_count: int
    total_count: int
    moshaf_ids: list[str]
    finished_moshaf_ids: list[str] = field(default_factory=[])
    error_ids: list[str] = field(default_factory=[])


@st.cache_data
def get_suar_names() -> list[str]:
    return get_suar_list()


@st.dialog("Cancel Download?")
def cancel_download_with_confirmation(pid: int):
    col1, col2 = st.columns(2)
    placeholder = st.empty()

    with placeholder.container():
        with col1:
            if st.button("Yes", use_container_width=True,):
                try:
                    kill_process(pid)
                    conf.DOWNLOAD_LOCK_FILE.unlink()  # remove the download.lock file
                    placeholder.success('Download is canceled')
                except Exception as e:
                    placeholder.error(f'Error While canceling Download: {e}')
                    raise e
                st.rerun()
        with col2:
            if st.button("No", use_container_width=True):
                placeholder.info("Aborting ...")
                time.sleep(1)
                st.rerun()


def write_to_download_lock_log(log: DownloadLog, filepath: Path):
    with open(filepath, 'w+') as f:
        json.dump(asdict(log), f)


@st.cache_data(ttl=1)
def get_download_lock_log(filepath: Path) -> DownloadLog:
    with open(filepath, 'r') as f:
        log = json.load(f)
    return DownloadLog(**log)


def download_all_moshaf_pool(moshaf_ids: list[str] = None, refresh=False):
    """Download All moshaf pool callback
    Args:
        moshaf_ids (list[str]): if None will download all moshaf objects with
            attribute `is_downloaded` is True else will download the moshaf ids
        refresh (bool): refresh delete and recreate moshaf directory and redownload if necessary
    """
    if not conf.DOWNLOAD_LOCK_FILE.is_file():
        if moshaf_ids is None:
            to_download_ids = [
                m.id for m in st.session_state.moshaf_pool if not m.is_downloaded or refresh]
        elif isinstance(moshaf_ids, list):
            to_download_ids = [
                id for id in moshaf_ids if not st.session_state.moshaf_pool[id].is_downloaded or refresh]
        else:
            raise ValueError(
                f'"moshaf_ids should be list[str] but got: {type(moshaf_ids)}')

        if len(to_download_ids) == 0:
            if isinstance(moshaf_ids, list):
                if len(moshaf_ids) == 1:
                    pop_up_message(
                        f'Moshaf id={moshaf_ids[0]} is downloaded', 'info')
            else:
                pop_up_message('All Moshaf Pool is downloaded', 'info')
            return

        # saving moshaf & reciter pool before start downloading
        st.session_state.moshaf_pool.save()
        st.session_state.reciter_pool.save()

        # start a new INDEPENDENT download process aks: `spawn`
        # to be easy for termination & run in background
        ctx = multiprocessing.get_context('spawn')
        p = ctx.Process(
            target=download_all_moshaf_task,
            kwargs={
                'moshaf_pool': copy.deepcopy(st.session_state.moshaf_pool),
                'to_download_ids': to_download_ids,
                'lockfile_path': conf.DOWNLOAD_LOCK_FILE,
                'download_error_path': conf.DOWNLOAD_ERROR_LOG,
                'refresh': refresh,
            })
        # Detach the process to prevent it from being terminated with the main program
        p.daemon = True
        p.start()  # Start the process
        pop_up_message(
            'Download has started switch to **Download Page** or refresh the Page', 'success')
    else:
        pop_up_message('There is a download is already running...', 'warn')


def download_all_moshaf_task(
    moshaf_pool: MoshafPool,
    to_download_ids: list[str],
    lockfile_path: Path,
    download_error_path: Path,
    refresh=False,
):
    # remove error file
    if download_error_path.is_file():
        download_error_path.unlink()

    finished_ids = []
    error_ids = []
    error_logs = {}
    pid = os.getpid()
    for id in to_download_ids:
        log = DownloadLog(
            process_pid=pid,
            current_moshaf_id=id,
            finished_count=len(finished_ids),
            total_count=len(to_download_ids),
            finished_moshaf_ids=finished_ids,
            moshaf_ids=to_download_ids,
            error_ids=error_ids,
        )
        write_to_download_lock_log(log, lockfile_path)

        try:
            moshaf_pool.download_moshaf(
                id, refresh=refresh, save_on_disk=True)
            finished_ids.append(id)
        except Exception as e:
            error_logs[id] = traceback.format_exc()
            error_ids.append(id)

        # write logs for each iteration
        if error_logs:
            with open(download_error_path, 'w+') as f:
                json.dump(error_logs, f, indent=4)

    # End of download -> delete the download_lockfile
    lockfile_path.unlink()


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
        placeholder.error(f"Error Deleteing item: {str(e)}")
        # placeholder.error(f"Error Deleteing item: {str(e)}", 'error')
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
        # TODO: help should be input for every field
        help = (
            'Please enter every link in seprate line i.e: hit enter. Example:'
            '\nhttps://example.com/003.mp3\nhttps://example.com/004.mp3'
        )

        return st.text_area(
            label,
            key=key,
            value='\n'.join(default_value) if default_value else '',
            help=help,
            placeholder=help,
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
