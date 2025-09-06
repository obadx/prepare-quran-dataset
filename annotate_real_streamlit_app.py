import streamlit as st
import pandas as pd
import json
import numpy as np
from datasets import load_dataset
from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from pathlib import Path
from typing import Dict, List, Any

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "sura_idx" not in st.session_state:
    st.session_state.sura_idx = 1
if "aya_idx" not in st.session_state:
    st.session_state.aya_idx = 1
if "start_word_index" not in st.session_state:
    st.session_state.start_word_index = 0
if "num_words" not in st.session_state:
    st.session_state.num_words = 5
if "phonetic_script" not in st.session_state:
    st.session_state.phonetic_script = ""
if "sifat_df" not in st.session_state:
    st.session_state.sifat_df = pd.DataFrame()
if "last_editor_key" not in st.session_state:
    st.session_state.last_editor_key = ""
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "item_to_edit" not in st.session_state:
    st.session_state.item_to_edit = None


# Load dataset
@st.cache_resource
def load_audio_dataset():
    ds = load_dataset("obadx/ood_muaalem_test", split="train")
    rng = np.random.default_rng(seed=42)
    ids = np.arange(len(ds))
    rng.shuffle(ids)
    ids = [int(i) for i in ids]  # Convert to integers
    return ds, ids


try:
    ds, ids = load_audio_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# Cache sura information
@st.cache_data
def get_sura_info():
    sura_idx_to_name = {}
    sura_to_aya_count = {}
    start_aya = Aya()
    for sura_idx in range(1, 115):
        start_aya.set(sura_idx, 1)
        sura_idx_to_name[sura_idx] = start_aya.get().sura_name
        sura_to_aya_count[sura_idx] = start_aya.get().num_ayat_in_sura
    return sura_idx_to_name, sura_to_aya_count


sura_idx_to_name, sura_to_aya_count = get_sura_info()


# Load existing annotations
def load_annotations():
    annotations_file = Path("annotations.json")
    if annotations_file.exists():
        with open(annotations_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Initialize annotations
if not st.session_state.annotations:
    st.session_state.annotations = load_annotations()


# Function to save annotation to JSON file
def save_annotation(item_id, annotation):
    # Update session state
    st.session_state.annotations[item_id] = annotation

    # Save to file
    with open("annotations.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.annotations, f, ensure_ascii=False, indent=2)


# Get current item
current_id = ids[st.session_state.index]
item = ds[current_id]

# Check if current item has existing annotation
if item["id"] in st.session_state.annotations and not st.session_state.edit_mode:
    st.session_state.phonetic_script = st.session_state.annotations[item["id"]][
        "phonetic_script"
    ]
    st.session_state.sifat_df = pd.DataFrame(
        st.session_state.annotations[item["id"]]["sifat_table"]
    )

# App layout
st.title("Quran Audio Transcription Annotation Tool")

# Display audio and metadata
st.header("Audio Sample")
st.audio(item["audio"]["array"], sample_rate=item["audio"]["sampling_rate"])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("ID", item["id"])
with col2:
    st.metric("Source", item["source"])
with col3:
    st.metric("Original ID", item["original_id"])
with col4:
    st.metric("Progress", f"{st.session_state.index + 1}/{len(ids)}")

    # Show annotation status
    if item["id"] in st.session_state.annotations:
        st.success("✓ Annotated")
    else:
        st.info("Not annotated")

# Quran reference selection
st.header("Quran Reference")

col1, col2, col3, col4 = st.columns(4)
with col1:
    sura_idx = st.selectbox(
        "Sura",
        options=list(range(1, 115)),
        format_func=lambda x: f"{x}. {sura_idx_to_name[x]}",
        index=st.session_state.sura_idx - 1,
    )
with col2:
    max_aya = sura_to_aya_count[sura_idx]
    aya_idx = st.number_input(
        "Aya", min_value=1, max_value=max_aya, value=st.session_state.aya_idx
    )
with col3:
    start_word_index = st.number_input(
        "Start Word Index", min_value=0, value=st.session_state.start_word_index
    )
with col4:
    num_words = st.number_input(
        "Number of Words", min_value=1, max_value=20, value=st.session_state.num_words
    )

# Update session state
st.session_state.sura_idx = sura_idx
st.session_state.aya_idx = aya_idx
st.session_state.start_word_index = start_word_index
st.session_state.num_words = num_words

# Get Uthmani script
start_aya = Aya()
start_aya.set(sura_idx, aya_idx)
uthmani_script = start_aya.get_by_imlaey_words(start_word_index, num_words).uthmani

st.subheader("Uthmani Script")
st.write(uthmani_script)

# Phonetization
st.subheader("Phonetic Transcription")
if st.button("Generate Phonetic Transcription"):
    default_moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=4,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=4,
    )
    phonetizer_out = quran_phonetizer(uthmani_script, default_moshaf)
    st.session_state.phonetic_script = phonetizer_out.phonemes

    # Create sifat table
    sifat_data = []
    for sifa in phonetizer_out.sifat:
        sifat_data.append(
            {
                "phoneme": sifa.phonemes,
                "hams_or_jahr": sifa.hams_or_jahr,
                "shidda_or_rakhawa": sifa.shidda_or_rakhawa,
                "tafkheem_or_taqeeq": sifa.tafkheem_or_taqeeq,
                "itbaq": sifa.itbaq,
                "safeer": sifa.safeer,
                "qalqla": sifa.qalqla,
                "tikraar": sifa.tikraar,
                "tafashie": sifa.tafashie,
                "istitala": sifa.istitala,
                "ghonna": sifa.ghonna,
            }
        )

    st.session_state.sifat_df = pd.DataFrame(sifat_data)
    st.rerun()

# Editable phonetic script
phonetic_script = st.text_area(
    "Phonetic Script",
    value=st.session_state.phonetic_script,
    height=100,
    key="phonetic_script_editor",
)

# Editable sifat table
if not st.session_state.sifat_df.empty:
    st.subheader("Sifat Table")

    # Define options for each column
    column_options = {
        "hams_or_jahr": ["hams", "jahr"],
        "shidda_or_rakhawa": ["shadeed", "between", "rikhw"],
        "tafkheem_or_taqeeq": ["mofakham", "moraqaq", "low_mofakham"],
        "itbaq": ["monfateh", "motbaq"],
        "safeer": ["safeer", "no_safeer"],
        "qalqla": ["moqalqal", "not_moqalqal"],
        "tikraar": ["mokarar", "not_mokarar"],
        "tafashie": ["motafashie", "not_motafashie"],
        "istitala": ["mostateel", "not_mostateel"],
        "ghonna": ["maghnoon", "not_maghnoon"],
    }

    # Create a unique key for this editor instance
    editor_key = f"sifat_editor_{st.session_state.index}"

    # Use the data editor with explicit value assignment
    edited_df = st.data_editor(
        st.session_state.sifat_df,
        column_config={
            "phoneme": st.column_config.TextColumn("Phoneme", width="small"),
            **{
                col: st.column_config.SelectboxColumn(col, options=options)
                for col, options in column_options.items()
            },
        },
        use_container_width=True,
        num_rows="fixed",
        key=editor_key,
    )

    # Update the session state directly when the editor key changes
    if editor_key != st.session_state.last_editor_key:
        st.session_state.last_editor_key = editor_key
    else:
        # Only update if the dataframe has actually changed
        if not edited_df.equals(st.session_state.sifat_df):
            st.session_state.sifat_df = edited_df
            # Force a rerun to immediately reflect changes
            st.rerun()

# Navigation and saving
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Previous") and st.session_state.index > 0:
        st.session_state.index -= 1
        # Reset for the new item
        st.session_state.phonetic_script = ""
        st.session_state.sifat_df = pd.DataFrame()
        st.session_state.last_editor_key = ""
        st.session_state.edit_mode = False
        st.rerun()

with col2:
    if st.button("Save Annotation"):
        annotation = {
            "phonetic_script": phonetic_script,
            "sifat_table": st.session_state.sifat_df.to_dict(orient="records"),
        }

        # Save to JSON file
        save_annotation(item["id"], annotation)
        st.success(f"Annotation saved for ID: {item['id']}")
        st.session_state.edit_mode = False

with col3:
    # Edit button for current item (only shown if annotation exists)
    if item["id"] in st.session_state.annotations:
        if st.button("Edit Annotation", type="secondary"):
            # Load the existing annotation
            annotation = st.session_state.annotations[item["id"]]
            st.session_state.phonetic_script = annotation["phonetic_script"]
            st.session_state.sifat_df = pd.DataFrame(annotation["sifat_table"])
            st.session_state.edit_mode = True
            st.success("Annotation loaded for editing")
    else:
        st.write("")  # Empty space for layout

with col4:
    if st.button("Next") and st.session_state.index < len(ids) - 1:
        st.session_state.index += 1
        # Reset for the new item
        st.session_state.phonetic_script = ""
        st.session_state.sifat_df = pd.DataFrame()
        st.session_state.last_editor_key = ""
        st.session_state.edit_mode = False
        st.rerun()

# Export annotations
st.divider()
st.header("Annotations Management")

# Display current annotations with edit buttons
if st.session_state.annotations:
    st.subheader("Current Annotations")

    # Create a list of annotations with edit buttons
    annotations_list = []
    for item_id, annotation in st.session_state.annotations.items():
        # Find the index of this item in the dataset
        item_index = None
        for idx, original_idx in enumerate(ids):
            if ds[original_idx]["id"] == item_id:
                item_index = idx
                break

        annotations_list.append(
            {
                "id": item_id,
                "phonetic_script": annotation["phonetic_script"][:50] + "..."
                if len(annotation["phonetic_script"]) > 50
                else annotation["phonetic_script"],
                "sifat_count": len(annotation["sifat_table"]),
                "item_index": item_index,
            }
        )

    # Create a dataframe for display
    annotation_df = pd.DataFrame(annotations_list)

    # Display the dataframe with edit buttons
    for _, row in annotation_df.iterrows():
        col1, col2, col3, col4 = st.columns([2, 5, 2, 2])
        with col1:
            st.write(f"**ID:** {row['id']}")
        with col2:
            st.write(f"**Phonetic:** {row['phonetic_script']}")
        with col3:
            st.write(f"**Sifat:** {row['sifat_count']}")
        with col4:
            if st.button("Edit", key=f"edit_{row['id']}"):
                if row["item_index"] is not None:
                    st.session_state.index = row["item_index"]
                    st.session_state.edit_mode = True

                    # Load the annotation data
                    annotation = st.session_state.annotations[row["id"]]
                    st.session_state.phonetic_script = annotation["phonetic_script"]
                    st.session_state.sifat_df = pd.DataFrame(annotation["sifat_table"])

                    st.rerun()
                else:
                    st.error(f"Could not find item {row['id']} in dataset")

    # Provide download link
    with open("annotations.json", "r", encoding="utf-8") as f:
        st.download_button(
            label="Download JSON",
            data=f,
            file_name="annotations.json",
            mime="application/json",
        )
else:
    st.info("No annotations have been saved yet.")

# Add a button to clear all annotations
if st.button("Clear All Annotations"):
    if st.session_state.annotations:
        st.session_state.annotations = {}
        Path("annotations.json").unlink(missing_ok=True)
        st.session_state.phonetic_script = ""
        st.session_state.sifat_df = pd.DataFrame()
        st.success("All annotations cleared")
        st.rerun()
    else:
        st.info("No annotations to clear")
