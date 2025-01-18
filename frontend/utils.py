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

from prepare_quran_dataset.construct.database import Pool, MoshafPool, ReciterPool
from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.utils import (
    kill_process,
    load_yaml,
    dump_yaml,
)
from prepare_quran_dataset.construct.quran_data_utils import get_suar_list, SURA_TO_AYA_COUNT
from prepare_quran_dataset.construct.docs_utils import (
    get_arabic_attributes,
    get_arabic_name,
)

import config as conf


@dataclass
class PopupMessage:
    msg: str
    msg_type: str

    def post_init(self):
        if self.msg_type not in conf.POPUP_MSG_ICONS:
            raise ValueError(
                f'`msg_type` should be one of {list(conf.POPUP_MSG_ICONS.keys())} got {self.msg_type}')

    def show(self):
        popup_message(
            self.msg,
            msg_type=self.msg_type, icons_dict=conf.POPUP_MSG_ICONS)


def popup_message(msg: str, msg_type: str = 'success', icons_dict=conf.POPUP_MSG_ICONS):
    """Displays a popup message on the right side as `st.toast`
    Args:
        msg (str): the message to display
        msg_type (str): either ['success', 'error', 'warn', 'info']
    """
    if msg_type not in icons_dict:
        raise ValueError(
            f'`msg_type` should be one of {list(icons_dict.keys())} got {msg_type}')
    st.toast(msg, icon=icons_dict[msg_type])


def popup_message_rerun(msg: str, msg_type: str = 'success'):
    """Displays a popup message after rerun"""
    st.session_state.popup_messages.append(PopupMessage(msg, msg_type))


@dataclass
class MoshafFilter:
    complete: bool = True
    not_complete: bool = True
    downloaded: bool = True
    not_downloaded: bool = True
    download_error: bool = True
    not_download_error: bool = True
    annotated: bool = True
    not_annotated: bool = True
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

    selected_moshaf = []
    reciter_ids = {r.id for r in filter.reciters} if filter.reciters else None
    error_ids = set()
    download_error_file = Path(download_error_file)
    if download_error_file.is_file():
        with open(download_error_file, 'r') as f:
            error_log = json.load(f)
            error_ids = set(error_log.keys())

    for moshaf in moshaf_pool:
        comp_flag = True
        # complete nd not_complete
        comp_flag = comp_flag and chose(
            filter.complete, filter.not_complete, moshaf.is_complete)

        # download and not download
        comp_flag = comp_flag and chose(
            filter.downloaded, filter.not_downloaded, moshaf.is_downloaded)

        # annotated & not annotated
        comp_flag = comp_flag and chose(
            filter.annotated, filter.not_annotated, moshaf.is_annotated)

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
def get_sura_to_aya_count() -> list[str]:
    return SURA_TO_AYA_COUNT


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

        # all moshaf is downloaded -> (Redownload popup)
        if len(to_download_ids) == 0:
            if moshaf_ids:
                to_download_ids = moshaf_ids
            else:
                # Check to redownload the whole Moshaf Pool
                to_download_ids = [m.id for m in st.session_state.moshaf_pool]
            redownload_confirmation(
                st.session_state.moshaf_pool, to_download_ids)
            return

        start_download(to_download_ids, refresh)

    else:
        popup_message('There is a download is already running...', 'warn')


@st.dialog("Already downloaded!")
def redownload_confirmation(
    moshaf_pool: MoshafPool,
    moshaf_ids: list[str] = [],
):
    """
    Returns:
        bool: True if re download, False otherwise
    """
    if not isinstance(moshaf_ids, list):
        raise ValueError(
            f'`moshaf_ids` should be list of strings not: {type(moshaf_ids)}')
    st.warning('The following Moshaf Items are downloaded:')
    str_ids = ""
    for idx in moshaf_ids:
        str_ids += (
            f'* {idx} -> {moshaf_pool[idx].name} / {moshaf_pool[idx].reciter_arabic_name}\n')
    st.markdown(str_ids)
    st.warning('Are you sure you want to redownload them ?')

    col1, col2 = st.columns(2)
    placeholder = st.empty()

    with placeholder.container():
        with col1:
            if st.button("Yes", use_container_width=True,):
                start_download(moshaf_ids, redownload=True)
        with col2:
            if st.button("No", use_container_width=True):
                st.rerun()


def start_download(
    to_download_ids: list[str] = None,
    refresh=False,
    redownload=False,
):
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
            'redownload': redownload,
        })
    # Detach the process to prevent it from being terminated with the main program
    p.daemon = True
    p.start()  # Start the process
    popup_message_rerun(
        'Download has started switch to **Download Page** or refresh the Page',
        msg_type='success',
    )
    st.rerun()


def download_all_moshaf_task(
    moshaf_pool: MoshafPool,
    to_download_ids: list[str],
    lockfile_path: Path,
    download_error_path: Path,
    refresh=False,
    redownload=False,
):
    print('Start Download ...........................')
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
                id,
                refresh=refresh, redownload=redownload, save_on_disk=True)
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
                if isinstance(pool, MoshafPool):
                    popup_message_rerun(
                        f'Delete Operation of **{item.id} / {item.name}** is canceled!', 'info')
                elif isinstance(pool, ReciterPool):
                    popup_message_rerun(
                        f'Delete Operation of **{item.id} / {item.arabic_name}** is canceled!', 'info')
                st.rerun()


def delete_item_from_pool(pool: Pool, item: BaseModel, placeholder):
    try:
        pool.delete(item.id)
        pool.save()

        if isinstance(pool, MoshafPool):
            popup_message_rerun(
                f'**{item.id} / {item.name}** is deleted successfully!' 'success')
        elif isinstance(pool, ReciterPool):
            popup_message_rerun(
                f'**{item.id} / {item.arabic_name}** is deleted successfully!', 'success')
    except Exception as e:
        if isinstance(pool, MoshafPool):
            placeholder.error(
                f"Error Deleteing Moshaf: **{item.id} / {item.name}**: {str(e)}")
        elif isinstance(pool, ReciterPool):
            placeholder.error(
                f"Error Deleteing Reciter: **{item.id} / {item.arabic_name}**: {str(e)}")
        raise e


@st.dialog("Save Pools?")
def save_pools_with_confirmation():
    col1, col2 = st.columns(2)
    placeholder = st.empty()

    with placeholder.container():
        with col1:
            if st.button("Yes", use_container_width=True,):
                save_pools(placeholder)
                st.rerun()
        with col2:
            if st.button("No", use_container_width=True):
                popup_message_rerun("Save Operation is canceled", 'info')
                st.rerun()


def save_pools(placeholder):
    try:
        st.session_state.reciter_pool.save()
        st.session_state.moshaf_pool.save()
        popup_message_rerun("All data saved successfully!", 'success')
    except Exception as e:
        popup_message_rerun(f"Error saving data: {str(e)}", 'error')
        raise e


def insert_or_update_item_in_pool(
    model: Type[BaseModel],
    pool: Pool,
    required_field_names: list[str],
    key_prefix='model_',
    item_name_in_session_state: str = None,
    states_values_after_submit: Optional[dict[str, Any]] = {},
    field_funcs_after_submit: Optional[dict[str, Callable[[Any], Any]]] = {},
    after_submit_callback: Optional[Callable[[Pool, BaseModel], None]] = None,
):
    """Insert or update item in either `MoshafPool` or `ReciterPool` form with submit button

    Args:
        field_funcs_after_submit ( Optional[dict[str, Callable[[Any], Any]]]):
            dict[str, func], dict to aggregate field value after from
            submition, Ex: {'id': lambda x: str(x)}
        after_submit_callbak: (Optional[Callable[[Pool, BaseModel]]]): a function is called
            after the submitions runs successfully
            def ex_callback(pool, new_item):
                print(pool)
                print(new_item)
    """
    with st.form("insert_or_update_form"):
        form_data = {}
        for field_name in required_field_names:
            field_info = model.model_fields[field_name]
            val = create_input_for_field(
                field_name, field_info, key_prefix=key_prefix,
                default_value=getattr(st.session_state[item_name_in_session_state], field_name) if item_name_in_session_state else None)

            if field_name in field_funcs_after_submit:
                func = field_funcs_after_submit[field_name]
                form_data[field_name] = func(val)
            else:
                form_data[field_name] = val

        if st.form_submit_button(f"Confirm {model.__name__}"):

            try:
                new_item: BaseModel = None

                # Insertion Operation
                if item_name_in_session_state is None:
                    new_item = model(**form_data)
                    pool.insert(new_item)
                    popup_message("Insertion is successfully!",  'success')

                # Update Operation
                else:
                    for field_name, val in form_data.items():
                        setattr(
                            st.session_state[item_name_in_session_state], field_name, val)
                    pool.update(st.session_state[item_name_in_session_state])
                    new_item = st.session_state[item_name_in_session_state]
                    popup_message("Update is successfully!", 'success')
            except Exception as e:
                st.error(f"Error in the update/insert: {str(e)}")
                raise e

            # cleaning session_state after ooperation
            if item_name_in_session_state is not None:
                del st.session_state[item_name_in_session_state]
            for field_name in required_field_names:
                del st.session_state[key_prefix + field_name]

            # setting keys after submit
            for key, val in states_values_after_submit.items():
                st.session_state[key] = val

            if after_submit_callback is not None:
                after_submit_callback(pool, new_item)


def create_input_for_field(
    field_name: str,
    field_info: FieldInfo,
    default_value: Any = None,
    key_prefix='model_',
    help: str = None,
) -> Any:
    """Created an input filed given a pydantic `field_name`, `fidel_info`
    Args:
        default_value (Any): overwrites the `field_info.default` if default_value is None. Otherwise we ues `field_info.default`
        help (str): overwrites the `field_info.description` if help is None. Otherwise we ues `field_info.description` as a help text for input_filed
    """

    # Extract Arabic name from field description if available
    label = get_field_name(field_name, field_info)

    if default_value is None:
        if field_info.default != PydanticUndefined:
            default_value = field_info.default

    key = key_prefix + field_name

    if help is None:
        help = field_info.description

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
            help=help,
        )

    if field_info.annotation in [str, Optional[str]]:
        return st.text_input(label, value=default_value or "", key=key, help=help)
    elif field_info.annotation in [int, Optional[int]]:
        return st.number_input(label, value=default_value or 0, step=1, key=key, help=help)
    elif field_info.annotation in [float, Optional[float]]:
        return st.number_input(label, value=default_value or 0.0, step=0.1, key=key, help=help)
    elif field_info.annotation in [bool, Optional[bool]]:
        return st.checkbox(label, value=default_value or False, key=key, help=help)
    elif field_info.annotation in [list[str], Optional[list[str]]]:
        help = (
            'Please enter every link in seprate line i.e: hit enter. Example:'
            '\nhttps://example.com/003.mp3\nhttps://example.com/004.mp3\n'
        ) + help

        text = st.text_area(
            label,
            key=key,
            value='\n'.join(default_value) if default_value else '',
            help=help,
            placeholder=help,
        )
        return text.split('\n')
    elif field_info.annotation in [dict[str, str], Optional[dict[str, str]], dict[int, str], Optional[dict[int, str]]]:
        help = (
            'Place the specifc downloads as yaml:'
            f'\n{dump_yaml({3: 'https://example.com/003.mp3',
                           4: 'https://example.com/004.mp3'})}\n'
        ) + help
        yaml_text = st.text_area(
            label,
            key=key,
            value=dump_yaml(default_value) if default_value else '',
            help=help,
            placeholder=help,
        )
        return load_yaml(yaml_text) if yaml_text else {}

    raise ValueError(
        f"Unsupported field type for {label}: {field_info.annotation}")


def get_field_name(field_name: str, field_info: FieldInfo) -> str:
    """Recturn the Arabic name of the field if applicable else the field_name
    """
    label = field_name
    arabic_name = get_arabic_name(field_info)
    if arabic_name:
        label = f"{arabic_name} ({field_name})"

    return label
