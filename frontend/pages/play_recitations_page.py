import copy
import streamlit as st

from menu import menu_with_redirect
from prepare_quran_dataset.construct.data_classes import AudioFile


def play_recitations():
    if 'played_moshaf_item' in st.session_state:
        st.header(
            f"Play Recitations of Moshaf ID={st.session_state.played_moshaf_item.id}")
        recitations: list[AudioFile] = copy.deepcopy(
            st.session_state.played_moshaf_item.recitation_files)
        recitations = sorted(recitations, key=lambda item: item.name)
        for file_info in recitations:
            expander = st.expander(f'**{file_info.name}**')
            with expander:
                ext = file_info.name.split('.')[-1]
                st.write(f'Sample Rate={file_info.sample_rate}')
                st.write(f'File Type={ext}')
                st.audio(file_info.path, loop=False)


# displays sidebar menu & redirect to main page if not initialized
menu_with_redirect()
play_recitations()
