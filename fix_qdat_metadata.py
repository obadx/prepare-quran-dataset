from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset, Value
import re
import uuid
import subprocess
import os


def normalize_fienmaes(dir):
    original_ds_dir = Path(dir)
    # renaming files
    for file_path in original_ds_dir.iterdir():
        if file_path.is_file():
            # Transform the filename
            new_name = file_path.name.lower().replace("-", "_")
            new_path = file_path.parent / new_name

            # Rename the file if the new name is different
            if new_path != file_path:
                file_path.rename(new_path)
                print(f"Renamed: {file_path.name} -> {new_name}")


def downsample_and_to_wav(dir):
    directory = Path(dir)

    # Loop through all files in the directory
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() != ".wav":
            # Create the new filename with .wav extension
            new_filename = file_path.with_suffix(".wav")

            # Build the ffmpeg command
            command = [
                "ffmpeg",
                "-i",
                str(file_path),  # Input file
                "-ar",
                "16000",  # Set sample rate to 16kHz
                "-y",  # Overwrite output file if it exists
                str(new_filename),  # Output file
            ]

            try:
                # Run ffmpeg command
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Remove the original file if conversion was successful
                file_path.unlink()
                print(f"Converted and removed: {file_path.name} -> {new_filename.name}")

            except subprocess.CalledProcessError as e:
                print(f"Error converting {file_path.name}: {e}")
        elif file_path.is_file() and file_path.suffix.lower() == ".wav":
            # For existing WAV files, just convert to 16kHz
            temp_filename = file_path.with_stem(file_path.stem + "_temp")

            # Build the ffmpeg command
            command = [
                "ffmpeg",
                "-i",
                str(file_path),  # Input file
                "-ar",
                "16000",  # Set sample rate to 16kHz
                "-y",  # Overwrite output file if it exists
                str(temp_filename),  # Output file
            ]

            try:
                # Run ffmpeg command
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Replace the original file with the converted one
                file_path.unlink()
                temp_filename.rename(file_path)
                print(f"Converted: {file_path.name} to 16kHz")

            except subprocess.CalledProcessError as e:
                print(f"Error converting {file_path.name}: {e}")
                # Clean up temporary file if it was created
                if temp_filename.exists():
                    temp_filename.unlink()


metadata_path = "/home/abdullah/Downloads/qdat/original_metadata.csv"

# Read the CSV file, handling the first column correctly
df = pd.read_csv(metadata_path, skipinitialspace=True)

# Forward-fill missing values in the first column
df.iloc[:, 0] = df.iloc[:, 0].replace("", np.nan).ffill()

# Clean the first column by removing non-numeric characters and extra spaces
df.iloc[:, 0] = df.iloc[:, 0].str.replace(r"[^\d\s]", "", regex=True).str.strip()

# Fix specific data errors
df.columns = df.columns.str.strip()  # Clean column names
df["Gender"] = df["Gender"].replace(11, 1)  # Correct invalid gender value
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")  # Ensure Age is numeric

# Handle specific row issues (e.g., S160_10 row)
s160_index = df[df["title"] == "S160_10"].index
if not s160_index.empty:
    df.loc[s160_index, "Age"] = 21  # Correct Age for S160_10
    # Assign appropriate group label based on Age
    df.loc[s160_index, df.columns[0]] = "21"

# Save the cleaned DataFrame to a new CSV file
df.to_csv("fixed_qdata_metada.csv", index=False, encoding="utf-8-sig")

print("CSV file has been fixed and saved as 'fixed_file.csv'")


ds = Dataset.from_csv("./fixed_qdata_metada.csv")
print(ds)

ds = ds.remove_columns(["Unnamed: 0"])
print(ds)

ds = ds.rename_column("title", "file_name")

print(ds)

old_col_to_new = {}
for col in ds.column_names:
    new_col = col.lower()
    new_col = re.sub(r"\s", "_", new_col)
    old_col_to_new[col] = new_col


ds = ds.rename_columns(old_col_to_new)
print(ds)

ds = ds.map(lambda ex: {"file_name": ex["file_name"].lower().replace("-", "_")})

origianl_ds_file = Path("/home/abdullah/Downloads/qdat/train")
# downsample_and_to_wav(origianl_ds_file)

stem_to_file = {
    f.stem.lower().replace("-", "_"): f.name.lower().replace("-", "_")
    for f in origianl_ds_file.glob("*")
}

ds = ds.filter(lambda ex: ex["file_name"] in stem_to_file)
ds = ds.map(lambda ex: {"file_name": stem_to_file[ex["file_name"]]})

ds = ds.cast_column("the_tight_noon", Value("int32"))

ds = ds.map(lambda ex: {"id": str(uuid.uuid4())[:8]})

names = {n.split(".")[0] for n in ds["file_name"]}
print(len(names))
to_del_ids = []
curr_names = {}
for item in ds:
    if item["file_name"].split(".")[0] in curr_names:
        to_del_ids.append(item["id"])
        curr_names[item["file_name"].split(".")[0]].append(item["id"])
    else:
        curr_names[item["file_name"].split(".")[0]] = [item["id"]]

print("Duplicate Items")
id_col_to_idx = {id: idx for idx, id in enumerate(ds["id"])}
no_dup = True
for dup_file_name in curr_names:
    if len(curr_names[dup_file_name]) > 1:
        no_dup = False
        for id in curr_names[dup_file_name]:
            idx = id_col_to_idx[id]
            print(ds[idx])
        print("-" * 40)
if no_dup:
    print("No Duplicate Items in the metada")


missing_metadata_files = set(stem_to_file.values()) - set(ds["file_name"])
print(missing_metadata_files)

ds = ds.map(lambda ex: {"original_id": ex["file_name"].split(".")[0]})


print(ds)
print(ds[45])

ds = ds.to_csv("./fixed_qdata_metada.csv")
# normalize_fienmaes(origianl_ds_file)
