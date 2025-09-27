import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Literal
from enum import IntEnum

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes, search, SifaOutput
from datasets import load_dataset
from pydantic import BaseModel, Field

MADD_LEN = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]


class NoonMoshaddahLen(IntEnum):
    NO_GHONNAH = 0
    PARTIAL = 1
    COMPLETE = 2


class NoonMokhfahLen(IntEnum):
    NOON = 0
    PARTIAL = 1
    COMPLETE = 2


class Qalqalah(IntEnum):
    NO_QALQALH = 0
    HAS_QALQALAH = 1


class QdataBenchItem(BaseModel):
    # premetives
    id: str = Field(description="Unique hash id for every element")
    original_id: str = Field(description="The item's id in the original dataset")
    gender: Literal["male", "female"]

    # General Tajeweed
    qalo_alif_len: MADD_LEN = Field(
        default=2, description="The lengths of the normal madd alif form word `قالوا`"
    )
    qalo_waw_len: MADD_LEN = Field(
        default=2, description="The length of the normal madd waw form word `قالوا`"
    )
    laa_alif_len: MADD_LEN = Field(
        default=2, description="The length of the normal madd alif form word `لا`"
    )
    separate_madd: MADD_LEN = Field(
        default=4, description="The length of separate madd for word `لنا إنك`"
    )
    noon_moshaddadah_len: NoonMoshaddahLen = Field(
        default=NoonMoshaddahLen.COMPLETE,
        description="The length of noon moshaddah for word `إنَّك`",
    )
    noon_mokhfah_len: NoonMokhfahLen = Field(
        default=NoonMokhfahLen.COMPLETE,
        description="The length of noon mokhfah for word ` أنت`",
    )
    allam_alif_len: MADD_LEN = Field(
        default=2, description="The lengths of the normal madd alif form word `علام`"
    )

    madd_aared_len: MADD_LEN = Field(
        default=4, description="The length of the مد العارض للسكون "
    )
    qalqalah: Qalqalah = Field(
        default=Qalqalah.HAS_QALQALAH,
        description="The existance of qalqalah for not for word `الغيوب`",
    )

    # quran-transcript
    phonetic_transcript: str = Field(
        description="The phoetic transcript using `quran-transcript` package guidlenes"
    )
    sifat: list[SifaOutput] = Field(
        description="The sifat  transcript level using `quran-transcript` package guidlenes"
    )


# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "sura_idx" not in st.session_state:
    st.session_state.sura_idx = 1
if "start_aya" not in st.session_state:
    st.session_state.start_aya = Aya(1, 1)
if "uthmani_script" not in st.session_state:
    st.session_state.uthmani_script = ""
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

# Initialize QdataBenchItem fields
if "gender" not in st.session_state:
    st.session_state.gender = "male"
if "qalo_alif_len" not in st.session_state:
    st.session_state.qalo_alif_len = 4
if "qalo_waw_len" not in st.session_state:
    st.session_state.qalo_waw_len = 4
if "laa_alif_len" not in st.session_state:
    st.session_state.laa_alif_len = 4
if "separate_madd" not in st.session_state:
    st.session_state.separate_madd = 4
if "noon_moshaddadah_len" not in st.session_state:
    st.session_state.noon_moshaddadah_len = NoonMoshaddahLen.COMPLETE
if "noon_mokhfah_len" not in st.session_state:
    st.session_state.noon_mokhfah_len = NoonMokhfahLen.COMPLETE
if "allam_alif_len" not in st.session_state:
    st.session_state.allam_alif_len = 4
if "madd_aared_len" not in st.session_state:
    st.session_state.madd_aared_len = 4
if "qalqalah" not in st.session_state:
    st.session_state.qalqalah = Qalqalah.qalqalah


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
            raw_annotations = json.load(f)
        # Convert sifat lists to SifaOutput instances
        for item_id, annotation in raw_annotations.items():
            if "sifat" in annotation:
                annotation["sifat"] = [
                    SifaOutput(**sifa) if isinstance(sifa, dict) else sifa
                    for sifa in annotation["sifat"]
                ]
        return raw_annotations
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
    annotation_data = st.session_state.annotations[item["id"]]
    # Load QdataBenchItem fields
    st.session_state.phonetic_script = annotation_data.get("phonetic_transcript", "")
    # Convert sifat to DataFrame
    sifat_list = annotation_data.get("sifat", [])
    st.session_state.sifat_df = pd.DataFrame(
        [
            sifa.model_dump() if isinstance(sifa, SifaOutput) else sifa
            for sifa in sifat_list
        ]
    )
    # Add row index column
    if not st.session_state.sifat_df.empty:
        st.session_state.sifat_df.insert(
            0, "row_index", range(1, len(st.session_state.sifat_df) + 1)
        )
    # Load other fields
    st.session_state.gender = annotation_data.get("gender", "male")
    st.session_state.qalo_alif_len = annotation_data.get("qalo_alif_len", 4)
    st.session_state.qalo_waw_len = annotation_data.get("qalo_waw_len", 4)
    st.session_state.laa_alif_len = annotation_data.get("laa_alif_len", 4)
    st.session_state.separate_madd = annotation_data.get("separate_madd", 4)
    st.session_state.noon_moshaddadah_len = annotation_data.get(
        "noon_moshaddadah_len", NoonMoshaddahLen.COMPLETE
    )
    st.session_state.noon_mokhfah_len = annotation_data.get(
        "noon_mokhfah_len", NoonMokhfahLen.COMPLETE
    )
    st.session_state.allam_alif_len = annotation_data.get("allam_alif_len", 4)
    st.session_state.madd_aared_len = annotation_data.get("madd_aared_len", 4)
    st.session_state.qalqalah = annotation_data.get("qalqalah", Qalqalah.qalqalah)

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

sura_idx = st.selectbox(
    "Sura",
    options=list(range(1, 115)),
    format_func=lambda x: f"{x}. {sura_idx_to_name[x]}",
    # index=st.session_state.sura_idx - 1,
)
max_aya = sura_to_aya_count[sura_idx]
st.session_state.start_aya.set(sura_idx, max_aya // 2)
search_text = st.text_input(
    "Enter search text",
    value=st.session_state.start_aya.set_new(sura_idx, 1).get().imlaey,
)
search_out = search(
    search_text,
    start_aya=st.session_state.start_aya,
    window=max_aya + 1,
    remove_tashkeel=True,
    ignore_hamazat=True,
)
if len(search_out) == 0:
    st.error("No search results found")
    uthmani_script = ""
else:
    uthmani_script = search_out[0].uthmani_script

st.subheader("Uthmani Script")
st.write(uthmani_script)

# Update session state
st.session_state.sura_idx = sura_idx
st.session_state.uthmani_script = uthmani_script


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
    # TODO: aligment here
    sifat_data = []
    for sifa in phonetizer_out.sifat:
        sifat_data.append(sifa.model_dump())

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

    # Add row index column if it doesn't exist
    if "row_index" not in st.session_state.sifat_df.columns:
        st.session_state.sifat_df.insert(
            0, "row_index", range(1, len(st.session_state.sifat_df) + 1)
        )

    # Add row operations
    col_add, col_add_pos, col_del = st.columns(3)
    with col_add:
        if st.button("➕ Add Row at End"):
            # Add a new empty row at the end
            new_row = {
                col: ""
                for col in st.session_state.sifat_df.columns
                if col != "row_index"
            }
            new_row["row_index"] = len(st.session_state.sifat_df) + 1
            st.session_state.sifat_df = pd.concat(
                [st.session_state.sifat_df, pd.DataFrame([new_row])], ignore_index=True
            )
            st.rerun()

    with col_add_pos:
        if len(st.session_state.sifat_df) > 0:
            insert_position = st.selectbox(
                "Insert after row",
                options=list(range(len(st.session_state.sifat_df))),
                format_func=lambda x: f"After row {x + 1}",
            )
            if st.button("➕ Insert Row"):
                # Split the dataframe and insert new row
                new_row = {
                    col: ""
                    for col in st.session_state.sifat_df.columns
                    if col != "row_index"
                }
                new_row["row_index"] = (
                    insert_position + 1.5
                )  # Placeholder, will be renumbered
                # Insert the new row
                st.session_state.sifat_df = pd.concat(
                    [
                        st.session_state.sifat_df.iloc[: insert_position + 1],
                        pd.DataFrame([new_row]),
                        st.session_state.sifat_df.iloc[insert_position + 1 :],
                    ],
                    ignore_index=True,
                )
                # Renumber all rows
                st.session_state.sifat_df["row_index"] = range(
                    1, len(st.session_state.sifat_df) + 1
                )
                st.rerun()

    with col_del:
        if len(st.session_state.sifat_df) > 0:
            row_to_delete = st.selectbox(
                "Select row to delete",
                options=list(range(len(st.session_state.sifat_df))),
                format_func=lambda x: f"Row {x + 1}",
            )
            if st.button("➖ Delete Selected Row"):
                st.session_state.sifat_df = st.session_state.sifat_df.drop(
                    st.session_state.sifat_df.index[row_to_delete]
                ).reset_index(drop=True)
                # Renumber remaining rows
                st.session_state.sifat_df["row_index"] = range(
                    1, len(st.session_state.sifat_df) + 1
                )
                st.rerun()

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
            "row_index": st.column_config.NumberColumn(
                "Row", width="small", disabled=True
            ),
            "phoneme": st.column_config.TextColumn("Phoneme", width="small"),
            **{
                col: st.column_config.SelectboxColumn(col, options=options)
                for col, options in column_options.items()
            },
        },
        use_container_width=True,
        num_rows="dynamic",  # Change to dynamic to allow adding rows through the editor
        key=editor_key,
        disabled=["row_index"],  # Make row index non-editable
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

# QdataBenchItem fields
st.header("QdataBenchItem Annotation")

col1, col2 = st.columns(2)
with col1:
    st.session_state.gender = st.radio("Gender", ["male", "female"])
with col2:
    st.session_state.qalqalah = st.selectbox(
        "Qalqalah", options=list(Qalqalah), format_func=lambda x: x.name
    )

st.subheader("Madd Lengths")
cols = st.columns(4)
with cols[0]:
    st.session_state.qalo_alif_len = st.slider(
        "Qalo Alif Len", 0, 8, st.session_state.qalo_alif_len
    )
with cols[1]:
    st.session_state.qalo_waw_len = st.slider(
        "Qalo Waw Len", 0, 8, st.session_state.qalo_waw_len
    )
with cols[2]:
    st.session_state.laa_alif_len = st.slider(
        "Laa Alif Len", 0, 8, st.session_state.laa_alif_len
    )
with cols[3]:
    st.session_state.separate_madd = st.slider(
        "Separate Madd", 0, 8, st.session_state.separate_madd
    )

cols = st.columns(3)
with cols[0]:
    st.session_state.allam_alif_len = st.slider(
        "Allam Alif Len", 0, 8, st.session_state.allam_alif_len
    )
with cols[1]:
    st.session_state.madd_aared_len = st.slider(
        "Madd Aared Len", 0, 8, st.session_state.madd_aared_len
    )
with cols[2]:
    st.session_state.noon_moshaddadah_len = st.selectbox(
        "Noon Moshaddadah Len",
        options=list(NoonMoshaddahLen),
        format_func=lambda x: x.name,
    )

st.session_state.noon_mokhfah_len = st.selectbox(
    "Noon Mokhfah Len", options=list(NoonMokhfahLen), format_func=lambda x: x.name
)

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
        # Remove the row_index column before saving
        sifat_df_to_save = st.session_state.sifat_df.drop(
            columns=["row_index"], errors="ignore"
        )
        # Convert sifat records to SifaOutput instances
        sifat_list = []
        for record in sifat_df_to_save.to_dict(orient="records"):
            # Filter out None values and empty strings
            filtered_record = {
                k: v for k, v in record.items() if v is not None and v != ""
            }
            sifat_list.append(SifaOutput(**filtered_record))

        # Create QdataBenchItem instance
        bench_item = QdataBenchItem(
            id=item["id"],
            original_id=item["original_id"],
            gender=st.session_state.gender,
            qalo_alif_len=st.session_state.qalo_alif_len,
            qalo_waw_len=st.session_state.qalo_waw_len,
            laa_alif_len=st.session_state.laa_alif_len,
            separate_madd=st.session_state.separate_madd,
            noon_moshaddadah_len=st.session_state.noon_moshaddadah_len,
            noon_mokhfah_len=st.session_state.noon_mokhfah_len,
            allam_alif_len=st.session_state.allam_alif_len,
            madd_aared_len=st.session_state.madd_aared_len,
            qalqalah=st.session_state.qalqalah,
            phonetic_transcript=phonetic_script,
            sifat=sifat_list,
        )

        # Save to JSON file
        save_annotation(item["id"], bench_item.model_dump())
        st.success(f"Annotation saved for ID: {item['id']}")
        st.session_state.edit_mode = False

with col3:
    # Edit button for current item (only shown if annotation exists)
    if item["id"] in st.session_state.annotations:
        if st.button("Edit Annotation", type="secondary"):
            # Load the existing annotation
            annotation = st.session_state.annotations[item["id"]]
            st.session_state.phonetic_script = annotation.get("phonetic_transcript", "")
            # Convert sifat to DataFrame
            sifat_list = annotation.get("sifat", [])
            st.session_state.sifat_df = pd.DataFrame(
                [
                    sifa.model_dump() if isinstance(sifa, SifaOutput) else sifa
                    for sifa in sifat_list
                ]
            )
            # Add row index column
            if not st.session_state.sifat_df.empty:
                st.session_state.sifat_df.insert(
                    0, "row_index", range(1, len(st.session_state.sifat_df) + 1)
                )
            # Load other fields
            st.session_state.gender = annotation.get("gender", "male")
            st.session_state.qalo_alif_len = annotation.get("qalo_alif_len", 4)
            st.session_state.qalo_waw_len = annotation.get("qalo_waw_len", 4)
            st.session_state.laa_alif_len = annotation.get("laa_alif_len", 4)
            st.session_state.separate_madd = annotation.get("separate_madd", 4)
            st.session_state.noon_moshaddadah_len = annotation.get(
                "noon_moshaddadah_len", NoonMoshaddahLen.COMPLETE
            )
            st.session_state.noon_mokhfah_len = annotation.get(
                "noon_mokhfah_len", NoonMokhfahLen.COMPLETE
            )
            st.session_state.allam_alif_len = annotation.get("allam_alif_len", 4)
            st.session_state.madd_aared_len = annotation.get("madd_aared_len", 4)
            st.session_state.qalqalah = annotation.get("qalqalah", Qalqalah.qalqalah)
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

        # Use the correct key 'phonetic_transcript' instead of 'phonetic_script'
        phonetic_text = annotation.get("phonetic_transcript", "")
        truncated_phonetic = (
            phonetic_text[:50] + "..." if len(phonetic_text) > 50 else phonetic_text
        )

        # Get sifat count
        sifat_count = len(annotation.get("sifat", []))

        annotations_list.append(
            {
                "id": item_id,
                "phonetic_script": truncated_phonetic,
                "sifat_count": sifat_count,
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
                    st.session_state.phonetic_script = annotation.get(
                        "phonetic_transcript", ""
                    )
                    # Convert sifat to DataFrame
                    sifat_list = annotation.get("sifat", [])
                    st.session_state.sifat_df = pd.DataFrame(
                        [
                            sifa.model_dump() if isinstance(sifa, SifaOutput) else sifa
                            for sifa in sifat_list
                        ]
                    )
                    # Add row index column
                    if not st.session_state.sifat_df.empty:
                        st.session_state.sifat_df.insert(
                            0, "row_index", range(1, len(st.session_state.sifat_df) + 1)
                        )
                    # Load other fields
                    st.session_state.gender = annotation.get("gender", "male")
                    st.session_state.qalo_alif_len = annotation.get("qalo_alif_len", 4)
                    st.session_state.qalo_waw_len = annotation.get("qalo_waw_len", 4)
                    st.session_state.laa_alif_len = annotation.get("laa_alif_len", 4)
                    st.session_state.separate_madd = annotation.get("separate_madd", 4)
                    st.session_state.noon_moshaddadah_len = annotation.get(
                        "noon_moshaddadah_len", NoonMoshaddahLen.COMPLETE
                    )
                    st.session_state.noon_mokhfah_len = annotation.get(
                        "noon_mokhfah_len", NoonMokhfahLen.COMPLETE
                    )
                    st.session_state.allam_alif_len = annotation.get(
                        "allam_alif_len", 4
                    )
                    st.session_state.madd_aared_len = annotation.get(
                        "madd_aared_len", 4
                    )
                    st.session_state.qalqalah = annotation.get(
                        "qalqalah", Qalqalah.qalqalah
                    )
                    st.session_state.edit_mode = True
                    st.session_state.index = row["item_index"]
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
