import copy
from uuid import uuid4
import subprocess
import os
from pathlib import Path

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

        # Create a unique key for this file's conversion state
        conversion_key = f"conversion_{file_info.name}"
        download_key = f"download_{file_info.name}"

        # Initialize session state for this file
        if conversion_key not in st.session_state:
            st.session_state[conversion_key] = {
                "status": "idle",  # 'idle', 'converting', 'ready', 'error'
                "temp_path": None,
                "wav_data": None,
                "filename": None,
            }

        state = st.session_state[conversion_key]

        # New Download wav 16000 button
        if st.button(
            "Convert to wav 16000",
            key=f"wav_{file_info.name}",
            use_container_width=True,
            disabled=state["status"] != "idle",
        ):
            # Update state immediately to prevent re-triggering
            state["status"] = "converting"

            # Create temporary directory if it doesn't exist
            TEMP_DIR = Path("temp")
            TEMP_DIR.mkdir(parents=True, exist_ok=True)

            # Create temporary file path
            temp_path = TEMP_DIR / f"{uuid4().hex}.wav"
            state["temp_path"] = str(temp_path)

            # Prepare file paths
            input_path = str(conf.BASE_DIR / file_info.path)

            # Run FFmpeg conversion
            try:
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    input_path,
                    "-ar",
                    "16000",
                    "-y",  # Overwrite output file without asking
                    str(temp_path),
                ]
                subprocess.run(cmd, check=True, timeout=120)

                # Read converted file
                with open(temp_path, "rb") as f:
                    state["wav_data"] = f.read()

                # Set download filename
                base_name = file_info.name.split(".")[0]
                state["filename"] = f"{base_name}_16000.wav"
                state["status"] = "ready"

            except subprocess.TimeoutExpired:
                state["status"] = "error"
                st.error("Conversion timed out (120 seconds)")
                if temp_path.exists():
                    os.unlink(temp_path)
            except subprocess.CalledProcessError as e:
                state["status"] = "error"
                st.error(f"Conversion failed: {str(e)}")
                if temp_path.exists():
                    os.unlink(temp_path)
            except Exception as e:
                state["status"] = "error"
                st.error(f"Error: {str(e)}")
                if temp_path.exists():
                    os.unlink(temp_path)

        # Show conversion status
        if state["status"] == "converting":
            st.info("Converting to wav 16000... Please wait.")
        elif state["status"] == "ready":
            st.success("Conversion complete!")

            # Create download button
            st.download_button(
                label="Download wav 16000",
                data=state["wav_data"],
                file_name=state["filename"],
                key=download_key,
                on_click=lambda: cleanup_temp_file(state),
                use_container_width=True,
            )

            # Add button to reset conversion
            if st.button(
                "Convert another file",
                key=f"reset_{file_info.name}",
                use_container_width=True,
            ):
                cleanup_temp_file(state)
                state["status"] = "idle"
                state["wav_data"] = None
                st.experimental_rerun()

        elif state["status"] == "error":
            st.error("Conversion failed. Please try again.")


def cleanup_temp_file(state):
    """Cleanup temporary file and reset state"""
    if state.get("temp_path") and os.path.exists(state["temp_path"]):
        os.unlink(state["temp_path"])
    state["temp_path"] = None
    state["wav_data"] = None


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
