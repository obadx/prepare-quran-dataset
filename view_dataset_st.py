from pathlib import Path
from random import randint
import argparse

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
            st.write(f"{key}: {item[key]}")
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
    f_ds = ds.filter(lambda ex: int(ex["duration_seconds"]) >= threshold, num_proc=16)
    for item in f_ds:
        display_audio_file(item)


def display_small_durations(ds: Dataset, threshold: float):
    """
    Args:
        sura_idx (int): Absolute sura index
    """
    f_ds = ds.filter(lambda ex: int(ex["duration_seconds"]) <= threshold, num_proc=16)
    for item in f_ds:
        display_audio_file(item)


def display_moshaf(ds_path: Path, moshaf: Moshaf):
    ds = load_dataset(str(ds_path), name=f"moshaf_{moshaf.id}", split="train")
    st.write(f"عدد المقاطع: {len(ds)}")
    st.write(moshaf.reciter_arabic_name)

    left_col, right_col = st.columns(2)

    with right_col:
        if st.button("اختر عينة عشاوئية", use_container_width=True):
            rand_idx = randint(0, len(ds) - 1)
            st.session_state.rand_idx = rand_idx

    if "rand_idx" in st.session_state:
        st.subheader("عينة عشوائية")
        display_audio_file(ds[st.session_state.rand_idx], key_prefix="rand")

    avaiable_suar = [int(r.name.split(".")[0]) for r in moshaf.recitation_files]
    avaiable_suar = sorted(avaiable_suar)
    with left_col:
        sura_idx = st.selectbox(
            "اختر السورة",
            avaiable_suar,
            format_func=lambda x: SUAR_LIST[x - 1],
        )
        st.write(f"عدد الآيات بالسورة: {SURA_TO_AYA_COUNT[sura_idx]}")

    st.subheader("المقاطع القصيرة")
    small_duration = st.number_input("ادخل المدة بالثواني", value=3.0)
    display_small_durations(ds, small_duration)

    st.subheader("المقاطع الطويلة")
    long_duration = st.number_input("ادخل المدة بالثواني", value=30.0)
    display_higher_durations(ds, long_duration)

    st.subheader("مقاطع السورة")
    display_sura(ds, sura_idx)


if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    args = parser.parse_args()

    ds_path = args.dataset_path
    reciter_pool = ReciterPool(ds_path / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, ds_path)

    sel_moshaf_id = st.selectbox(
        "اختر مصحفا",
        [m.id for m in moshaf_pool],
    )
    display_moshaf(ds_path, moshaf_pool[sel_moshaf_id])
