from pathlib import Path
from random import randint
from dataclasses import dataclass

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


def save_moshaf_operation(moshaf_id: str, op: Operation):
    if moshaf_id not in st.session_state.moshaf_id_to_config:
        st.session_state.moshaf_id_to_config[moshaf_id] = MoshafEditConfig(
            id=moshaf_id, operations=[op]
        )
        st.session_state.edit_config.configs.append(
            st.session_state.moshaf_id_to_config[moshaf_id]
        )
    else:
        st.session_state.moshaf_id_to_config[moshaf_id].operations.append(op)

    # Save OPerations
    st.session_state.edit_config.to_yaml(st.session_state.edit_config_path)
    st.session_state.moshaf_to_seg_to_ops = (
        st.session_state.edit_config.to_moshaf_dict()
    )


@st.dialog("إضافة عنصر")
def insert_with_confirmation(item: dict):
    with st.form("add_form"):
        st.write(f"segment_index: {item['segment_index']}")
        st.write(f"Sura Index: {item['sura_or_aya_index']}")
        new_tarteel_transcript = st.text_input(
            "tarteel_transcript",
        )
        new_audio_file = st.text_input("audio_file")

        if st.form_submit_button("موافق", use_container_width=True):
            if (
                new_tarteel_transcript != item["tarteel_transcript"][0]
                and new_audio_file
            ):
                operation = Operation(
                    type="insert",
                    segment_index=item["segment_index"],
                    new_tarteel_transcript=new_tarteel_transcript,
                    new_audio_file=new_audio_file,
                )
                save_moshaf_operation(item["moshaf_id"], operation)
                popup_message(
                    f"تم إضافة عنصر بنجاح: **{item['segment_index']}**",
                    "success",
                )
                st.rerun()

            else:
                popup_message(
                    "يجب ملأ كل الحقول",
                    "info",
                )


@st.dialog("عدل العنصر")
def update_with_confirmation(item: dict):
    with st.form("update_form"):
        st.write(f"segment_index: {item['segment_index']}")
        st.write(f"Sura Index: {item['sura_or_aya_index']}")
        new_tarteel_transcript = st.text_input(
            "tarteel_transcript",
            value=item["tarteel_transcript"][0],
        )
        new_audio_file = st.text_input("audio_file")

        if st.form_submit_button("موافق", use_container_width=True):
            if (
                new_tarteel_transcript != item["tarteel_transcript"][0]
                or new_audio_file
            ):
                operation = Operation(
                    type="update",
                    segment_index=item["segment_index"],
                    new_tarteel_transcript=new_tarteel_transcript
                    if new_tarteel_transcript != item["tarteel_transcript"][0]
                    else None,
                    new_audio_file=new_audio_file if new_audio_file else None,
                )
                save_moshaf_operation(item["moshaf_id"], operation)
                popup_message(
                    f"تم إضافة التعديل ل: **{item['segment_index']}**",
                    "success",
                )
                st.rerun()

            else:
                popup_message(
                    "لا يوجد تعديل",
                    "info",
                )


def view_operation(item: dict, op: Operation):
    st.write(f"**segment_index:** {item['segment_index']}")
    st.write(f"**Sura Index:** {item['sura_or_aya_index']}")
    st.write(f"**Reciter Arabic Name:** {item['reciter_arabic_name']}")

    if op.new_tarteel_transcript:
        st.write(f"**New Tarteel Transcript:** {op.new_tarteel_transcript}")

    if op.new_audio_file:
        st.write(f"**New aduio File Path:** {op.new_audio_file}")
        base_path = st.session_state.update_wave_files_path
        try:
            wav_bytes = numpy_to_wav_bytes(op.to_hf(base_path)["audio"]["array"], 16000)
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
):
    """Displayes an audio file with download button"""
    expander = st.expander(f"**{item['segment_index']}**")
    with expander:
        keys = set(item.keys()) - {"audio"}
        for key in keys:
            st.write(f"**{key}:** {item[key]}")

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
                    if op.type == "delete":
                        st.write("DELETE ❌")
                    elif op.type == "update":
                        if st.button(
                            "Update ℹ️",
                            key=f"{key_prefix}_track_{item['segment_index']}_view_update",
                        ):
                            view_update_operation(item, op)
                    elif op.type == "insert":
                        if st.button(
                            "Insert 📥",
                            key=f"{key_prefix}_track_{item['segment_index']}_view_insert",
                        ):
                            view_insert_operation(item, op)

        if st.button(
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


def display_sura(ds: Dataset, sura_idx):
    """
    Args:
        sura_idx (int): Absolute sura index
    """
    f_ds = ds.filter(lambda ex: int(ex["sura_or_aya_index"]) == sura_idx, num_proc=16)
    for item in f_ds:
        display_audio_file(item)


def display_higher_durations(ds: Dataset, threshold: float):
    """
    Args:
        sura_idx (int): Absolute sura index
    """
    f_ds = ds.filter(lambda ex: int(ex["duration_seconds"]) > threshold, num_proc=16)
    for item in f_ds:
        display_audio_file(item, key_prefix="high")


def display_small_durations(ds: Dataset, threshold: float):
    """
    Args:
        sura_idx (int): Absolute sura index
    """
    f_ds = ds.filter(lambda ex: int(ex["duration_seconds"]) < threshold, num_proc=16)
    for item in f_ds:
        display_audio_file(item, key_prefix="small")


def display_moshaf(ds_path: Path, moshaf: Moshaf):
    ds = load_dataset(str(ds_path), name=f"moshaf_{moshaf.id}", split="train")
    st.write(f"عدد المقاطع: {len(ds)}")
    st.write(moshaf.reciter_arabic_name)

    col1, col2, col3 = st.columns(3)

    with col3:
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
        if st.button("اعرض التعديلات"):
            st.session_state.display_edits = True

    if "display_edits" in st.session_state:
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

    st.subheader("المقاطع القصيرة")
    small_duration = st.number_input("ادخل المدة بالثواني", value=3.0)
    display_small_durations(ds, small_duration)

    st.subheader("المقاطع الطويلة")
    long_duration = st.number_input("ادخل المدة بالثواني", value=30.0)
    display_higher_durations(ds, long_duration)

    st.subheader("مقاطع السورة")
    display_sura(ds, sura_idx)


if __name__ == "__main__":
    ds_path = "/cluster/users/shams035u1/data/mualem-recitations-annotated"
    # ds_path = "../out-quran-ds/"
    # update_wave_files_path = "../moshaf-fixes/"
    edit_config_path = "./edit_config.yml"
    update_wave_files_path = (
        "/cluster/users/shams035u1/data/mualem-recitations-annotated/moshaf-fixes"
    )

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

    ds_path = Path(ds_path)
    reciter_pool = ReciterPool(ds_path / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, ds_path)

    sel_moshaf_id = st.selectbox(
        "اختر مصحفا",
        [m.id for m in moshaf_pool],
    )
    display_moshaf(ds_path, moshaf_pool[sel_moshaf_id])
