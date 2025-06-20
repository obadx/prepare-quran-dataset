import copy
import tempfile
import subprocess
import os

import streamlit as st
from menu import menu_with_redirect
from utils import get_suar_names, get_sura_to_aya_count
import config as conf

from prepare_quran_dataset.construct.data_classes import AudioFile


def display_audio_file(
    file_info: AudioFile,
    filename: str = None,
):
    """Displayes an audio file with download button"""
    if filename is None:
        filename = file_info.name.split(".")[0]

    ext = file_info.name.split(".")[-1]
    expander = st.expander(f"**{file_info.name} / {filename}**")
    with expander:
        st.write(f"Sample Rate={file_info.sample_rate}")
        st.write(f"File Type={ext}")
        if st.button("Load File", key=file_info.name, use_container_width=True):
            st.audio(str(conf.BASE_DIR / file_info.path), loop=False)
            with open(conf.BASE_DIR / file_info.path, "rb") as file:
                st.download_button(
                    label="Download File",
                    data=file,
                    file_name=file_info.name,
                )

        # New Download wav 16000 button
        if st.button(
            "Download wav 16000", key=f"wav_{file_info.name}", use_container_width=True
        ):
            with st.spinner("Converting to wav 16000..."):
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav", dir="./"
                    ) as tmpfile:
                        temp_path = tmpfile.name

                    # Prepare file paths
                    input_path = str(conf.BASE_DIR / file_info.path)
                    output_path = temp_path

                    # Run FFmpeg conversion
                    cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", output_path]
                    subprocess.run(
                        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )

                    # Offer download and clean up
                    with open(output_path, "rb") as f:
                        wav_data = f.read()
                    os.unlink(output_path)  # Delete temp file immediately after reading

                    # Create download button
                    base_name = file_info.name.split(".")[0]
                    new_filename = f"{base_name}_16000.wav"
                    st.download_button(
                        label="Click to download wav 16000",
                        data=wav_data,
                        file_name=new_filename,
                        key=f"download_wav_{file_info.name}",
                    )

                except subprocess.CalledProcessError as e:
                    st.error(f"Conversion failed: {e.stderr.decode()}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def play_recitations():
    if "played_moshaf_item" in st.session_state:
        st.header(
            f"Play Recitations of Moshaf ID={st.session_state.played_moshaf_item.id}"
        )
        recitations: list[AudioFile] = copy.deepcopy(
            st.session_state.played_moshaf_item.recitation_files
        )

        if st.session_state.played_moshaf_item.segmented_by == "sura":
            recitations = sorted(recitations, key=lambda item: item.name)
            for file_info in recitations:
                sura_index = file_info.name.split(".")[0]
                sura_name = ""
                sura_name = get_suar_names()[int(sura_index) - 1]
                display_audio_file(file_info, filename=sura_name)

        elif st.session_state.played_moshaf_item.segmented_by == "aya":
            # dict[sura_idx, aya_idx]
            sura_idx_to_fileinfo: dict[int, dict[int, [AudioFile]]] = {}
            named_files = []
            for info in recitations:
                name = info.name.split(".")[0]
                try:
                    sura_idx = int(name[:3])
                    aya_idx = int(name[3:])
                    if sura_idx not in sura_idx_to_fileinfo:
                        sura_idx_to_fileinfo[sura_idx] = {}
                    sura_idx_to_fileinfo[sura_idx][aya_idx] = info
                except ValueError:
                    named_files.append(info)

            # sorting keys of the dict
            sura_idx_to_fileinfo = dict(sorted(sura_idx_to_fileinfo.items()))
            for k in sura_idx_to_fileinfo:
                sura_idx_to_fileinfo[k] = dict(sorted(sura_idx_to_fileinfo[k].items()))

            # display named files
            for info in named_files:
                display_audio_file(info)

            # Displaying specific Aya of Specific Sura
            left_col, right_col = st.columns(2)
            with right_col:
                selected_sura_idx = st.selectbox(
                    "اختار السورة:",
                    list(sura_idx_to_fileinfo.keys()),
                    format_func=lambda i: get_suar_names()[i - 1],
                )
            with left_col:
                selected_aya_idx = st.selectbox(
                    "اختر الآية:",
                    list(sura_idx_to_fileinfo[selected_sura_idx].keys()),
                )
            display_audio_file(
                sura_idx_to_fileinfo[selected_sura_idx][selected_aya_idx]
            )


if __name__ == "__main__":
    # displays sidebar menu & redirect to main page if not initialized
    menu_with_redirect()
    play_recitations()
