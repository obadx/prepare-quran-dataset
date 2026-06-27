from pathlib import Path
from random import randint
from dataclasses import dataclass
import re
from typing import Literal
import json

import streamlit as st
from datasets import load_dataset, Dataset
import numpy as np
import io
import wave


from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool
from prepare_quran_dataset.construct.quran_data_utils import (
    SUAR_LIST,
    SURA_TO_AYA_COUNT,
)
from prepare_quran_dataset.annotate.edit import EditConfig, MoshafEditConfig, Operation
from prepare_quran_dataset.annotate.main import FIRST_CHANNEL_ONLY_MOSHAF
from quran_transcript import normalize_aya


def float_equal_n(a, b, n):
    return round(a, n) == round(b, n)


POPUP_MSG_ICONS: dict[str, str] = {
    "success": "✅",
    "error": "❌",
    "warn": "⚠️",
    "info": "ℹ️",
}


def popup_message(msg: str, msg_type: str = "success", icons_dict=POPUP_MSG_ICONS):
    """Displays a popup message on the right side as `st.toast`
    Args:
        msg (str): the message to display
        msg_type (str): either ['success', 'error', 'warn', 'info']
    """
    if msg_type not in icons_dict:
        raise ValueError(
            f"`msg_type` should be one of {list(icons_dict.keys())} got {msg_type}"
        )
    st.toast(msg, icon=icons_dict[msg_type])


@dataclass
class PopupMessage:
    msg: str
    msg_type: str

    def post_init(self):
        if self.msg_type not in POPUP_MSG_ICONS:
            raise ValueError(
                f"`msg_type` should be one of {list(POPUP_MSG_ICONS.keys())} got {self.msg_type}"
            )

    def show(self):
        popup_message(self.msg, msg_type=self.msg_type, icons_dict=POPUP_MSG_ICONS)


def popup_message_rerun(msg: str, msg_type: str = "success"):
    """Displays a popup message after rerun"""
    st.session_state.popup_messages.append(PopupMessage(msg, msg_type))


def save_to_config_to_disk():
    # Save OPerations
    st.session_state.edit_config.to_yaml(st.session_state.edit_config_path)
    st.session_state.moshaf_to_seg_to_ops = (
        st.session_state.edit_config.to_moshaf_dict()
    )


def delete_moshaf_operation(moshaf_id: str, op: Operation):
    st.session_state.moshaf_id_to_config[moshaf_id].delete_operation(op)
    save_to_config_to_disk()


def save_moshaf_operation(moshaf_id: str, op_or_ops: Operation | list[Operation]):
    if isinstance(op_or_ops, Operation):
        op_or_ops = [op_or_ops]

    if moshaf_id not in st.session_state.moshaf_id_to_config:
        st.session_state.moshaf_id_to_config[moshaf_id] = MoshafEditConfig(
            id=moshaf_id, operations=op_or_ops
        )
        st.session_state.edit_config.configs.append(
            st.session_state.moshaf_id_to_config[moshaf_id]
        )
    else:
        for op in op_or_ops:
            st.session_state.moshaf_id_to_config[moshaf_id].add_operation(op)

    save_to_config_to_disk()


def is_qlqla_kobra(text) -> bool:
    """Whethr the aya has قلقة كبرى or not"""

    qlqla = "قطبجد"
    shadda = "ّ"
    text = re.sub(r"\s+", "", text)  # remvoe spaces
    if re.search(f"[{qlqla}]{shadda}.$", text):
        return True
    return False


def is_hams_end(text) -> bool:
    """Whethr the aya has همس متطرف or not"""

    hams = "فحثهشخصسكت"
    shadda = "ّ"
    haraka = "ًٌٍَُِْ"
    text = re.sub(r"\s+", "", text)  # remvoe spaces
    if re.search(f"[{hams}]{shadda}?[{haraka}]$", text):
        return True
    return False


def is_hams_end_for_char(text, char="ت") -> bool:
    """Whethr the aya has همس متطرف or not"""

    shadda = "ّ"
    haraka = "ًٌٍَُِْ"
    text = re.sub(r"\s+", "", text)  # remvoe spaces
    if re.search(f"[{char}]{shadda}?[{haraka}]$", text):
        return True
    return False


def is_qlqla_kobra_qaf(text) -> bool:
    """Whethr the aya has قاف مقلقة متطرفة مشددة"""

    qaf = "ق"
    shadda = "ّ"
    text = re.sub(r"\s+", "", text)  # remvoe spaces
    if re.search(f"[{qaf}]{shadda}.$", text):
        return True
    return False


def add_operation(item: dict, operation_type: Literal["insert", "update"]):
    with st.form("update_form"):
        st.write(f"segment_index: {item['segment_index']}")
        st.write(f"Sura Index: {item['sura_or_aya_index']}")
        new_tarteel_transcript = st.text_input(
            "tarteel_transcript",
            value=item["tarteel_transcript"][0],
        )
        new_start_seconds = st.number_input(
            "new_start_seconds",
            value=item["timestamp_seconds"][0],
            step=0.001,
        )
        new_end_seconds = st.number_input(
            "new_end_seconds",
            value=item["timestamp_seconds"][1],
            step=0.001,
        )
        new_audio_file = st.text_input("audio_file")

        if st.form_submit_button("موافق", use_container_width=True):
            operation = Operation(
                type=operation_type,
                segment_index=item["segment_index"],
                new_tarteel_transcript=new_tarteel_transcript
                if new_tarteel_transcript != item["tarteel_transcript"][0]
                else None,
                new_audio_file=new_audio_file if new_audio_file else None,
                new_start_seconds=new_start_seconds
                if not float_equal_n(new_start_seconds, item["timestamp_seconds"][0], 3)
                else None,
                new_end_seconds=new_end_seconds
                if not float_equal_n(new_end_seconds, item["timestamp_seconds"][1], 3)
                else None,
            )
            save_moshaf_operation(item["moshaf_id"], operation)
            if operation_type == "insert":
                popup_message(
                    f"تم إضافة عنصر بنجاح: **{item['segment_index']}**",
                    "success",
                )
            elif operation_type == "update":
                popup_message(
                    f"تم إضافة التعديل ل: **{item['segment_index']}**",
                    "success",
                )
            st.rerun()


@st.dialog("إضافة عنصر")
def insert_with_confirmation(item: dict):
    add_operation(item, "insert")


@st.dialog("عدل العنصر")
def update_with_confirmation(item: dict):
    add_operation(item, "update")


def view_operation(item: dict, op: Operation):
    st.write(f"**segment_index:** {item['segment_index']}")
    st.write(f"**Sura Index:** {item['sura_or_aya_index']}")
    st.write(f"**Reciter Arabic Name:** {item['reciter_arabic_name']}")

    if op.new_tarteel_transcript:
        st.write(f"**New Tarteel Transcript:** {op.new_tarteel_transcript}")

    if (
        op.new_audio_file is not None
        or op.new_start_seconds is not None
        or op.new_end_seconds is not None
    ):
        try:
            base_path = st.session_state.update_wave_files_path
            media_files_path = (
                st.session_state.original_quran_dataset_path
                / f"dataset/{item['moshaf_id']}"
            )
            new_item = op.to_hf(
                item["timestamp_seconds"],
                fixes_base_path=base_path,
                media_files_path=media_files_path,
                first_channel_only=item["moshaf_id"] in FIRST_CHANNEL_ONLY_MOSHAF,
            )

            wav_bytes = numpy_to_wav_bytes(new_item["audio"]["array"], 16000)
            st.write(f"**New aduio File Path:** {op.new_audio_file}")
            st.write(f"**New Timestamp:** {new_item['timestamp_seconds']}")
            st.audio(wav_bytes, format="audio/wav")
        except Exception as e:
            st.write(f"⚠️Error while loading file:, {e}")
            popup_message(
                f"Error while loading file: {base_path / op.new_audio_file}", "error"
            )


@st.dialog("عرض التعديل")
def view_update_operation(item: dict, op: Operation):
    view_operation(item, op)


@st.dialog("عرض الإضافة")
def view_insert_operation(item: dict, op: Operation):
    view_operation(item, op)


@st.dialog("احذف العنصر?")
def delete_item_with_confirmation(item: dict):
    st.warning(f"هل ترغب في حذف العنصر: **{item['segment_index']}** ?")
    col1, col2 = st.columns(2)
    placeholder = st.empty()

    with placeholder.container():
        with col1:
            if st.button(
                "نعم",
                use_container_width=True,
            ):
                operation = Operation(
                    type="delete", segment_index=item["segment_index"]
                )
                save_moshaf_operation(item["moshaf_id"], operation)
                popup_message(
                    "تم إضافة عملية الحذف بنجاح",
                    "success",
                )
                st.rerun()

        with col2:
            if st.button("لا", use_container_width=True):
                popup_message(
                    "إلغائ عملية الحذف",
                    "info",
                )
                st.rerun()


@st.dialog("هل أنت متأكد من حذف العملية?")
def abort_opearation_with_confirmation(item: dict, op: Operation):
    st.warning(
        f"هل أنت متأكد من حذف عملية: **{op.type}** > **{item['segment_index']}** ?"
    )
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "نعم",
            use_container_width=True,
        ):
            delete_moshaf_operation(item["moshaf_id"], op)
            popup_message(
                f"تم حذف العملية: **{op.type}** > **{item['segment_index']}**",
                "success",
            )
            st.rerun()

    with col2:
        if st.button("لا", use_container_width=True):
            popup_message(
                "تم الإلغاء",
                "info",
            )
            st.rerun()


@st.dialog("ظبط زمن الهمس المتطرف")
def adjust_hams_end_duration(ds: Dataset):
    with st.form("hams_form"):
        hams_pad_ms = st.number_input("مدة الزيادة ms", value=0)

        if st.form_submit_button("موافق"):
            if hams_pad_ms:
                with st.spinner("الضبط حارٍٍ"):
                    filter_ds = ds.filter(
                        lambda ex: is_hams_end(ex["tarteel_transcript"][-1]),
                        num_proc=16,
                    )

                    if len(filter_ds) > 0:
                        new_operations = []
                        moshaf_id = filter_ds[0]["moshaf_id"]
                        for item in filter_ds:
                            new_operations.append(
                                Operation(
                                    type="update",
                                    segment_index=item["segment_index"],
                                    new_end_seconds=item["timestamp_seconds"][1]
                                    + hams_pad_ms / 1000,
                                )
                            )
                        save_moshaf_operation(moshaf_id, new_operations)
                        popup_message("تم طبط زمن الهمس المتطرف بنجاح", "success")
                        st.rerun()
                    else:
                        popup_message("الزمن لا بد أن يكون أكبر من الصفر !", "error")


@st.dialog("ظبط زمن القلقة المتطرفة")
def adjust_qlqla_duration(ds: Dataset):
    with st.form("qlqal_form"):
        qlqala_pad_ms = st.number_input("مدة الزيادة ms", value=0)

        if st.form_submit_button("موافق"):
            if qlqala_pad_ms:
                with st.spinner("الضبط حارٍٍ"):
                    filter_ds = ds.filter(
                        lambda ex: is_qlqla_kobra(ex["tarteel_transcript"][-1]),
                        num_proc=16,
                    )

                    if len(filter_ds) > 0:
                        new_operations = []
                        moshaf_id = filter_ds[0]["moshaf_id"]
                        for item in filter_ds:
                            new_operations.append(
                                Operation(
                                    type="update",
                                    segment_index=item["segment_index"],
                                    new_end_seconds=item["timestamp_seconds"][1]
                                    + qlqala_pad_ms / 1000,
                                )
                            )
                        save_moshaf_operation(moshaf_id, new_operations)
                        popup_message(
                            "تم طبط زمن القلقة للقاف المشددة المتطرفة بنجاح", "success"
                        )
                        st.rerun()
                    else:
                        popup_message("لا يوجد قاف مقلقة متطرفة !", "error")


# Convert NumPy array to WAV bytes
def numpy_to_wav_bytes(audio_array, sample_rate):
    # Normalize to 16-bit range (-32768 to 32767)
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        audio_array = (audio_array * 32767).astype(np.int16)
    elif audio_array.dtype != np.int16:
        audio_array = audio_array.astype(np.int16)  # Convert if not float or int16

    # Create in-memory bytes buffer
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 2 bytes (16-bit)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())
        return wav_buffer.getvalue()


def display_audio_file(
    item: dict,
    key_prefix="",
    ignore_load_button=False,
    expanded=False,
):
    """Displayes an audio file with download button"""
    expander = st.expander(f"**{item['segment_index']}**", expanded=expanded)
    with expander:
        keys = set(item.keys()) - {"audio"}
        for key in keys:
            st.write(f"**{key}:** {item[key]}")
        m_id = item["moshaf_id"]
        if m_id in st.session_state.moshaf_to_seg_to_tasmeea:
            if item["segment_index"] in st.session_state.moshaf_to_seg_to_tasmeea[m_id]:
                st.write(
                    st.session_state.moshaf_to_seg_to_tasmeea[m_id][
                        item["segment_index"]
                    ]
                )

        # view operations on this item
        if item["moshaf_id"] in st.session_state.moshaf_to_seg_to_ops:
            if (
                item["segment_index"]
                in st.session_state.moshaf_to_seg_to_ops[item["moshaf_id"]]
            ):
                st.write("**التعديلات:**")
                for op in st.session_state.moshaf_to_seg_to_ops[item["moshaf_id"]][
                    item["segment_index"]
                ]:
                    op_cols = st.columns(5)
                    if op.type == "delete":
                        with op_cols[0]:
                            st.write("DELETE ❌")
                    elif op.type == "update":
                        with op_cols[0]:
                            if st.button(
                                "Update ℹ️",
                                key=f"{key_prefix}_track_{item['segment_index']}_view_update",
                            ):
                                view_update_operation(item, op)
                    elif op.type == "insert":
                        with op_cols[0]:
                            if st.button(
                                "Insert 📥",
                                key=f"{key_prefix}_track_{item['segment_index']}_view_insert",
                            ):
                                view_insert_operation(item, op)

                    with op_cols[-1]:
                        if st.button(
                            "Abort 🔄️❌",
                            key=f"{key_prefix}_track_{item['segment_index']}_abort_op_{op.type}",
                        ):
                            abort_opearation_with_confirmation(item, op)

        if ignore_load_button:
            wav_bytes = numpy_to_wav_bytes(item["audio"]["array"], 16000)
            st.audio(wav_bytes, format="audio/wav")

        elif st.button(
            "Load File",
            key=f"{key_prefix}_track_{item['segment_index']}",
            use_container_width=True,
        ):
            wav_bytes = numpy_to_wav_bytes(item["audio"]["array"], 16000)
            st.audio(wav_bytes, format="audio/wav")
            # with open(conf.BASE_DIR / file_info.path, "rb") as file:
            #     st.download_button(
            #         label="Download File",
            #         data=file,
            #         file_name=file_info.name,
            #     )

        left_col, middle_col, right_col = st.columns(3)
        with left_col:
            if st.button(
                "احذف ❌",
                use_container_width=True,
                key=f"{key_prefix}_track_{item['segment_index']}_delete",
            ):
                delete_item_with_confirmation(item)

        with middle_col:
            if st.button(
                "إضافة 📥",
                use_container_width=True,
                key=f"{key_prefix}_track_{item['segment_index']}_add",
            ):
                insert_with_confirmation(item)

        with right_col:
            if st.button(
                "عدل ✏️",
                use_container_width=True,
                key=f"{key_prefix}_track_{item['segment_index']}_update",
            ):
                update_with_confirmation(item)


def update_sura_start_idx():
    st.session_state.sura_view_info["start_idx"] = (
        st.session_state.sura_start_idx_number_input
    )


def display_sura(ds, sura_idx, moshaf_id):
    """
    Args:
        sura_idx (int): Absolute sura index
    """
    # Reset
    if "sura_view_info" not in st.session_state:
        st.session_state.sura_view_info = {
            "sura_idx": sura_idx,
            "moshaf_id": moshaf_id,
            "start_idx": 0,
        }
    if (
        st.session_state.sura_view_info["sura_idx"] != sura_idx
        or st.session_state.sura_view_info["moshaf_id"] != moshaf_id
    ):
        st.session_state.sura_view_info = {
            "sura_idx": sura_idx,
            "moshaf_id": moshaf_id,
            "start_idx": 0,
        }

    range_columns = st.columns(5)
    with range_columns[0]:
        count_per_page = st.number_input("عدد العناصر بالصفحة", value=20)
    with range_columns[-1]:
        start_idx = st.number_input(
            "بداية الصفحة",
            value=st.session_state.sura_view_info["start_idx"],
            on_change=update_sura_start_idx,
            key="sura_start_idx_number_input",
        )

    sura_ids = []
    for seg in st.session_state.moshaf_to_seg_to_idx[moshaf_id]:
        if int(seg.split(".")[0]) == sura_idx:
            sura_ids.append(st.session_state.moshaf_to_seg_to_idx[moshaf_id][seg])
    for idx in range(start_idx, min(start_idx + count_per_page, len(sura_ids))):
        ds_idx = sura_ids[idx]
        display_audio_file(ds[ds_idx], ignore_load_button=True)

    pages_columns = st.columns(2)

    # Moving pages
    with pages_columns[1]:
        if len(sura_ids) > (start_idx + count_per_page):
            if st.button("الصفحة التالية", use_container_width=True):
                st.session_state.sura_view_info["start_idx"] += count_per_page
    with pages_columns[0]:
        if (start_idx - count_per_page) >= 0:
            if st.button("الصفحة السابقة", use_container_width=True):
                st.session_state.sura_view_info["start_idx"] -= count_per_page


def display_higher_durations(ds: Dataset, threshold: float):
    """
    Args:
        sura_idx (int): Absolute sura index
    """
    f_ds = ds.filter(lambda ex: ex["duration_seconds"] > threshold, num_proc=16)
    for item in f_ds:
        display_audio_file(item, key_prefix="high")


def display_small_durations(ds: Dataset, threshold: float):
    """
    Args:
        sura_idx (int): Absolute sura index
    """
    f_ds = ds.filter(lambda ex: ex["duration_seconds"] < threshold, num_proc=16)
    for item in f_ds:
        display_audio_file(
            item, key_prefix="small", ignore_load_button=True, expanded=True
        )


def display_qlqla_kobra(ds):
    f_ds = ds.filter(
        lambda ex: is_qlqla_kobra(ex["tarteel_transcript"][-1]), num_proc=16
    )
    for item in f_ds:
        display_audio_file(item, key_prefix="small", ignore_load_button=True)


def display_hams_end(ds):
    char = st.selectbox("اختر حرف الهمس", list("فحثهشخصسكت"))
    f_ds = ds.filter(
        lambda ex: is_hams_end_for_char(ex["tarteel_transcript"][-1], char), num_proc=16
    )
    for item in f_ds:
        display_audio_file(item, key_prefix="small", ignore_load_button=True)


def display_suar_beginning(ds):
    f_ds = ds.filter(
        lambda ex: int(ex["segment_index"].split(".")[1]) == 0, num_proc=16
    )
    for item in f_ds:
        display_audio_file(item, key_prefix="begin", ignore_load_button=True)


def display_suar_end(ds):
    seg_to_idx = ds["segment_index"]
    chose_ids = []
    for idx in range(len(seg_to_idx) - 1):
        if seg_to_idx[idx].split(".")[0] != seg_to_idx[idx + 1].split(".")[0]:
            chose_ids.append(idx)
    chose_ids.append(len(seg_to_idx) - 1)

    for idx in chose_ids:
        display_audio_file(
            ds[idx], key_prefix="end", ignore_load_button=True, expanded=True
        )


def display_empty_trans(ds):
    f_ds = ds.filter(
        lambda ex: len(normalize_aya(ex["tarteel_transcript"][-1])) <= 1,
        num_proc=16,
    )
    for item in f_ds:
        display_audio_file(item, key_prefix="begin", ignore_load_button=True)


def display_long_trans(ds):
    length = st.number_input("طول المقطع بالحروف", 100)
    f_ds = ds.filter(
        lambda ex: any(
            len(
                normalize_aya(
                    s,
                    remove_tashkeel=True,
                    ignore_hamazat=True,
                    ignore_alef_maksoora=True,
                )
            )
            >= length
            for s in ex["tarteel_transcript"]
        ),
        num_proc=16,
    )
    for item in f_ds:
        display_audio_file(item, key_prefix="begin", ignore_load_button=True)


def display_tasmeea_errors(ds):
    m_id = ds[0]["moshaf_id"]

    error_segs = []
    for sura_idx in st.session_state.tasmeea_errors[m_id]:
        if "nones" in st.session_state.tasmeea_errors[m_id][sura_idx]:
            error_segs += [
                x["segment_index"]
                for x in st.session_state.tasmeea_errors[m_id][sura_idx]["nones"]
            ]

    for seg_idx in sorted(error_segs):
        idx = st.session_state.moshaf_to_seg_to_idx[m_id][seg_idx]
        display_audio_file(ds[idx], key_prefix="begin", ignore_load_button=True)


def find_nearest_tasmeea_results(moshaf_id: str, sura_idx: int, aya_idx: int, winodw=1):
    seg_ids = []
    start_aya_idx = max(1, aya_idx - winodw)
    end_aya_idx = aya_idx + winodw
    if moshaf_id in st.session_state.moshaf_to_sura_to_tasmeea:
        if sura_idx in st.session_state.moshaf_to_sura_to_tasmeea[moshaf_id]:
            for tasmeea_info in st.session_state.moshaf_to_sura_to_tasmeea[moshaf_id][
                sura_idx
            ]:
                for search_aya_idx in range(start_aya_idx, end_aya_idx + 1):
                    if "start_span" in tasmeea_info:
                        if tasmeea_info["start_span"] is not None:
                            if (
                                sura_idx == tasmeea_info["start_span"]["sura_idx"]
                                and search_aya_idx
                                == tasmeea_info["start_span"]["aya_idx"]
                            ):
                                seg_ids.append(tasmeea_info["segment_index"])
    return seg_ids


def display_tasmeea_missings(ds):
    m_id = ds[0]["moshaf_id"]

    missings = []
    for sura_idx, sura_erros in st.session_state.tasmeea_errors[m_id].items():
        if "missings" in sura_erros:
            for item_idx, item in enumerate(sura_erros["missings"]):
                related_seg_ids = find_nearest_tasmeea_results(
                    moshaf_id=m_id,
                    sura_idx=item["start_span"]["sura_idx"],
                    aya_idx=item["start_span"]["aya_idx"],
                )
                missings.append((sura_idx, item_idx))

    st.write(f"عدد الأجزاء الناقصة: {len(missings)}")
    if len(missings) == 0:
        st.success("لا يوجد آيات ناقصة")
    else:
        missing_idx = st.number_input(
            "اختر رقم الخطأ", min_value=0, max_value=len(missings) - 1
        )
        sura_idx, idx_in_sura = missings[missing_idx]
        item = st.session_state.tasmeea_errors[m_id][sura_idx]["missings"][idx_in_sura]
        related_seg_ids = find_nearest_tasmeea_results(
            moshaf_id=m_id,
            sura_idx=item["start_span"]["sura_idx"],
            aya_idx=item["start_span"]["aya_idx"],
        )
        st.write(item)
        for seg_idx in related_seg_ids:
            idx = st.session_state.moshaf_to_seg_to_idx[m_id][seg_idx]
            display_audio_file(
                ds[idx],
                key_prefix=f"tasmeea_missings",
                ignore_load_button=True,
            )


def display_sakt(ds):
    m_id = ds[0]["moshaf_id"]
    hams_positions = [(18, 1), (18, 2), (36, 52), (75, 27), (83, 14)]
    seg_ids = []
    for sura_idx, aya_idx in hams_positions:
        seg_ids += find_nearest_tasmeea_results(m_id, sura_idx, aya_idx, winodw=0)

    for seg_idx in seg_ids:
        idx = st.session_state.moshaf_to_seg_to_idx[m_id][seg_idx]
        display_audio_file(
            ds[idx],
            key_prefix=f"hams_",
            ignore_load_button=True,
        )


def display_ghayn(ds):
    m_id = ds[0]["moshaf_id"]
    hams_positions = [(18, 64)]
    seg_ids = []
    for sura_idx, aya_idx in hams_positions:
        seg_ids += find_nearest_tasmeea_results(m_id, sura_idx, aya_idx, winodw=0)

    for seg_idx in seg_ids:
        idx = st.session_state.moshaf_to_seg_to_idx[m_id][seg_idx]
        display_audio_file(
            ds[idx],
            key_prefix=f"hams_",
            ignore_load_button=True,
        )


def display_hafs_ways(ds):
    m_id = ds[0]["moshaf_id"]
    hafs_way_to_poses = {
        "المد المنفصل, والمتصل": [(2, 21), (101, 10)],
        "المد المنفصل وقفا والمد العارض": [(3, 5), (3, 6)],
        "مد اللين": [(106, 1), (106, 2)],
        "ميم آل عمران": [(3, 1), (3, 2)],
        "المد الازم الحرفي للعين": [(19, 1), (42, 2)],
        "السكت عند هوجا": [(18, 1)],
        "السكت عند مرقدنا": [(36, 52)],
        "السكت عند من راق": [(75, 27)],
        "السكت عند بل ران": [(83, 14)],
        "أوجه ماليه هلك": [(69, 28), (69, 29)],
        "بين الانفال والتوبة": [(8, 75), (9, 1)],
        "الإدغام والمد في نون وياسين": [(68, 1), (36, 1)],
        "آتان بالنمل": [(27, 36)],
        "سين يبسط": [(2, 245)],
        "سين بسطة": [(7, 69)],
        "سين المسيطرون": [(52, 37)],
        "سين بمسيطر": [(88, 22)],
        "التسهيل ولامد": [(10, 59), (27, 59), (6, 143), (6, 144), (10, 51), (10, 91)],
        "يلهث ذلك": [(7, 176)],
        "اركب معنا": [(11, 42)],
        "تأمنا": [(12, 11)],
        "ضعف": [(30, 54)],
        "سلاسل": [(76, 4)],
        "نخلقكم": [(77, 20)],
        "فرق": [(26, 63)],
        "القطر": [(34, 12)],
        "مصر": [(10, 87), (12, 21), (12, 99), (43, 51)],
        "نذر": [(54, 16), (54, 18)],
        "يسر": [(89, 4)],
        "الميم المخفاة": [(2, 52)],
    }
    hafs_way_to_seg_ids = {}
    for way in hafs_way_to_poses:
        if way not in hafs_way_to_seg_ids:
            hafs_way_to_seg_ids[way] = []
        for sura_idx, aya_idx in hafs_way_to_poses[way]:
            hafs_way_to_seg_ids[way] += find_nearest_tasmeea_results(
                m_id, sura_idx, aya_idx, winodw=0
            )

    for way in hafs_way_to_seg_ids:
        st.subheader(way)
        for seg_idx in hafs_way_to_seg_ids[way]:
            idx = st.session_state.moshaf_to_seg_to_idx[m_id][seg_idx]
            display_audio_file(
                ds[idx],
                key_prefix=way,
                ignore_load_button=True,
            )


def display_moshaf(ds_path: Path, moshaf: Moshaf):
    ds = load_dataset(
        str(ds_path), name=f"moshaf_{moshaf.id}", split="train", num_proc=32
    )
    if "moshaf_to_seg_to_idx" not in st.session_state:
        st.session_state.moshaf_to_seg_to_idx = {}

    if moshaf.id not in st.session_state.moshaf_to_seg_to_idx:
        segment_ids = ds["segment_index"]
        seg_to_idx = {seg: idx for idx, seg in enumerate(segment_ids)}
        st.session_state.moshaf_to_seg_to_idx[moshaf.id] = seg_to_idx

    st.write(f"عدد المقاطع: {len(ds)}")
    st.write(moshaf.reciter_arabic_name)

    col1, col2, col3, col4 = st.columns(4)
    stat_coumns = st.columns(4)
    qlqal_columns = st.columns(3)
    hams_columns = st.columns(3)
    sura_stat_columns = st.columns(4)
    empty_transcript = st.columns(4)
    tasmeea_columns = st.columns(4)
    sakt_columns = st.columns(2)
    hafa_ways_columns = st.columns(2)
    ghayn_columns = st.columns(2)

    with col4:
        if st.button("اختر عينة عشاوئية", use_container_width=True):
            rand_idx = randint(0, len(ds) - 1)
            st.session_state.rand_idx = rand_idx

    if "rand_idx" in st.session_state:
        st.subheader("عينة عشوائية")
        display_audio_file(ds[st.session_state.rand_idx], key_prefix="rand")

    avaiable_suar = [int(r.name.split(".")[0]) for r in moshaf.recitation_files]
    avaiable_suar = sorted(avaiable_suar)
    with col1:
        sura_idx = st.selectbox(
            "اختر السورة",
            avaiable_suar,
            format_func=lambda x: f"{x} / {SUAR_LIST[x - 1]}",
        )
        st.write(f"عدد الآيات بالسورة: {SURA_TO_AYA_COUNT[sura_idx]}")

    with col2:
        if st.button("اعرض التعديلات", use_container_width=True):
            st.session_state.display_edits = True

    with col3:
        if st.button("اخف التعديلات", use_container_width=True):
            st.session_state.display_edits = False

    if "display_edits" in st.session_state:
        if st.session_state.display_edits:
            st.subheader("التعديلات")
            # view operations on this item
            if moshaf.id in st.session_state.moshaf_to_seg_to_ops:
                edited_ds = ds.filter(
                    lambda ex: ex["segment_index"]
                    in st.session_state.moshaf_to_seg_to_ops[moshaf.id],
                    keep_in_memory=True,
                    num_proc=16,
                )
                for item in edited_ds:
                    display_audio_file(item, key_prefix="edit")

    with stat_coumns[3]:
        if st.button("اظهر المقاطع القصيرة", use_container_width=True):
            st.session_state.display_short = True
    with stat_coumns[2]:
        if st.button("اخف المقاطع القصيرة", use_container_width=True):
            st.session_state.display_short = False

    with stat_coumns[1]:
        if st.button("اظهر المقاطع الطويلة", use_container_width=True):
            st.session_state.display_long = True
    with stat_coumns[0]:
        if st.button("اخف المقاطع الطويلة", use_container_width=True):
            st.session_state.display_long = False

    if "display_short" in st.session_state:
        if st.session_state.display_short:
            st.subheader("المقاطع القصيرة")
            small_duration = st.number_input("ادخل المدة بالثواني", value=3.0)
            display_small_durations(ds, small_duration)

    if "display_long" in st.session_state:
        if st.session_state.display_long:
            st.subheader("المقاطع الطويلة")
            long_duration = st.number_input("ادخل المدة بالثواني", value=30.0)
            display_higher_durations(ds, long_duration)

    with qlqal_columns[2]:
        if st.button("اظهر القلقة الكبرى", use_container_width=True):
            st.session_state.display_qlqla = True
    with qlqal_columns[0]:
        if st.button("اخف القلقة الكبرى", use_container_width=True):
            st.session_state.display_qlqla = False
    with qlqal_columns[1]:
        if st.button("اضبط زمن المقلقة المتطرفة", use_container_width=True):
            adjust_qlqla_duration(ds)

    if "display_qlqla" in st.session_state:
        if st.session_state.display_qlqla:
            st.subheader("القلقة الكبرى")
            display_qlqla_kobra(ds)

    with hams_columns[2]:
        if st.button("اظهر الهمس المتطرف", use_container_width=True):
            st.session_state.display_hams = True
    with hams_columns[0]:
        if st.button("اخف الهمس المتطرف", use_container_width=True):
            st.session_state.display_hams = False
    with hams_columns[1]:
        if st.button("اضبط زمن الهمس المتطرف", use_container_width=True):
            adjust_hams_end_duration(ds)

    if "display_hams" in st.session_state:
        if st.session_state.display_hams:
            st.subheader("الهمس المتطرف")
            display_hams_end(ds)

    with sura_stat_columns[3]:
        if st.button("أظهر أوائل السور", use_container_width=True):
            st.session_state.view_suar_beginning = True
    with sura_stat_columns[2]:
        if st.button("اخف أوائل السور", use_container_width=True):
            st.session_state.view_suar_beginning = False
    with sura_stat_columns[1]:
        if st.button("أظهر أواخر السور", use_container_width=True):
            st.session_state.view_suar_end = True
    with sura_stat_columns[0]:
        if st.button("اخف أواخر السور", use_container_width=True):
            st.session_state.view_suar_end = False

    if "view_suar_beginning" in st.session_state:
        if st.session_state.view_suar_beginning:
            st.subheader("أوائل السور")
            display_suar_beginning(ds)
    if "view_suar_end" in st.session_state:
        if st.session_state.view_suar_end:
            st.subheader("أواخر السور")
            display_suar_end(ds)

    with empty_transcript[3]:
        if st.button("أظهر النصوص الفارعة", use_container_width=True):
            st.session_state.show_empty_trans = True
    with empty_transcript[2]:
        if st.button("اخف النصوص الفارعة", use_container_width=True):
            st.session_state.show_empty_trans = False
    if "show_empty_trans" in st.session_state:
        if st.session_state.show_empty_trans:
            st.subheader("النصوص الفارغة")
            display_empty_trans(ds)
    with empty_transcript[1]:
        if st.button("أظهر النصوص الطويلة", use_container_width=True):
            st.session_state.show_long_trans = True
    with empty_transcript[0]:
        if st.button("اخف النصوص الطويلة", use_container_width=True):
            st.session_state.show_long_trans = False
    if "show_long_trans" in st.session_state:
        if st.session_state.show_long_trans:
            st.subheader("النصوص الطويلة")
            display_long_trans(ds)

    with tasmeea_columns[3]:
        if st.button("أظهر أخطاء التسميع", use_container_width=True):
            st.session_state.show_tasmeea_errors = True
    with tasmeea_columns[2]:
        if st.button("اخف أخطاء التسميع", use_container_width=True):
            st.session_state.show_tasmeea_errors = False
    with tasmeea_columns[1]:
        if st.button("أظهر الآيات الناقصصة", use_container_width=True):
            st.session_state.show_tasmeea_missings = True
    with tasmeea_columns[0]:
        if st.button("اخف الآيات الناقصصة", use_container_width=True):
            st.session_state.show_tasmeea_missings = False
    if "show_tasmeea_errors" in st.session_state:
        if st.session_state.show_tasmeea_errors:
            st.header("أخطاء التسميع")
            display_tasmeea_errors(ds)
    if "show_tasmeea_missings" in st.session_state:
        if st.session_state.show_tasmeea_missings:
            st.header("أخطاء الآيات الناقصة")
            display_tasmeea_missings(ds)

    with sakt_columns[1]:
        if st.button("أظهر السكت", use_container_width=True):
            st.session_state.show_sakt = True
    with sakt_columns[0]:
        if st.button("أخف السكت", use_container_width=True):
            st.session_state.show_sakt = False
    if "show_sakt" in st.session_state:
        if st.session_state.show_sakt:
            st.subheader("الموضع المحتلمة السكت")
            display_sakt(ds)

    with hafa_ways_columns[1]:
        if st.button("أظهر أوجه حفص", use_container_width=True):
            st.session_state.show_hafs_ways = True
    with hafa_ways_columns[0]:
        if st.button("اخف أوجه حفص", use_container_width=True):
            st.session_state.show_hafs_ways = False
    if "show_hafs_ways" in st.session_state:
        if st.session_state.show_hafs_ways:
            st.subheader("أوجه حفص")
            display_hafs_ways(ds)

    with ghayn_columns[1]:
        if st.button("أظهر الغين المكسورة وقفا", use_container_width=True):
            st.session_state.show_ghayn = True
    with ghayn_columns[0]:
        if st.button("أخف الغين المكسورة وقفا", use_container_width=True):
            st.session_state.show_ghayn = False
    if "show_ghayn" in st.session_state:
        if st.session_state.show_ghayn:
            st.subheader("العين المكسورة وقفا")
            display_ghayn(ds)

    st.subheader("مقاطع السورة")
    display_sura(ds, sura_idx, moshaf.id)


if __name__ == "__main__":
    ds_path = "/cluster/users/shams035u1/data/mualem-recitations-annotated"
    original_quran_dataset_path = "/cluster/users/shams035u1/data/quran-dataset"
    update_wave_files_path = (
        "/cluster/users/shams035u1/data/mualem-recitations-annotated/moshaf-fixes"
    )

    # ds_path = "../out-quran-ds/"
    # original_quran_dataset_path = "../quran-dataset"
    # update_wave_files_path = "../moshaf-fixes/"

    edit_config_path = "./configs/edit/edit_config.yml"

    if "edit_config" not in st.session_state:
        st.session_state.edit_config = EditConfig.from_yaml(edit_config_path)
        st.session_state.moshaf_id_to_config: dict[str, MoshafEditConfig] = {
            c.id: c for c in st.session_state.edit_config.configs
        }
        st.session_state.edit_config_path = edit_config_path
        st.session_state.moshaf_to_seg_to_ops = (
            st.session_state.edit_config.to_moshaf_dict()
        )

        st.session_state.update_wave_files_path = Path(update_wave_files_path)
        st.session_state.original_quran_dataset_path = Path(original_quran_dataset_path)

        # for tasmeea

        # dict{moahaf_id: tasmeea}
        st.session_state.moshaf_to_sura_to_tasmeea = {}
        st.session_state.moshaf_to_seg_to_tasmeea = {}
        # dict{moahaf_id: tasmeea}
        st.session_state.tasmeea_errors = {}

    ds_path = Path(ds_path)
    reciter_pool = ReciterPool(ds_path / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, ds_path)

    sel_moshaf_id = st.selectbox(
        "اختر مصحفا",
        [m.id for m in moshaf_pool],
    )
    if sel_moshaf_id not in st.session_state.moshaf_to_seg_to_tasmeea:
        tasmeea_dir = Path(ds_path) / f"tasmeea/{sel_moshaf_id}"
        tasmeea_file = tasmeea_dir / "tasmeea.json"
        tasmeea_errors_file = tasmeea_dir / "errors.json"
        if tasmeea_file.is_file():
            with open(tasmeea_file, "r", encoding="utf-8") as f:
                tasmeea = json.load(f)
            st.session_state.moshaf_to_sura_to_tasmeea[sel_moshaf_id] = {
                int(k): v for k, v in tasmeea.items()
            }
            seg_to_tasmeea_data = {}
            for sura_tasmeea in tasmeea.values():
                for tasmeea_info in sura_tasmeea:
                    seg_to_tasmeea_data[tasmeea_info["segment_index"]] = tasmeea_info
            st.session_state.moshaf_to_seg_to_tasmeea[sel_moshaf_id] = (
                seg_to_tasmeea_data
            )

        if tasmeea_errors_file.is_file():
            with open(tasmeea_errors_file, "r", encoding="utf-8") as f:
                st.session_state.tasmeea_errors[sel_moshaf_id] = json.load(f)

    display_moshaf(ds_path, moshaf_pool[sel_moshaf_id])
