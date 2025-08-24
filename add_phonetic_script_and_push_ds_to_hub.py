import os
from pathlib import Path

from quran_transcript import quran_phonetizer, MoshafAttributes
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from datasets import load_dataset


def load_secrets():
    # Load environment variables from .env
    load_dotenv()

    # Retrieve tokens
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Log into HuggingFace Hub
    if hf_token:
        hf_login(token=hf_token)
    else:
        print("HuggingFace token not found!")


def add_phonetic_script_map(ex, moshaf):
    ph_out = quran_phonetizer(ex["uthmani"], moshaf)

    return {
        "phonemes": ph_out.phonemes,
        "sifat": [s.model_dump() for s in ph_out.sifat],
    }


if __name__ == "__main__":
    # loading wandb tokens ans HF login
    repo_id = "obadx/muaalem-annotated-v1"
    load_secrets()
    ds_path = Path("/cluster/users/shams035u1/data/mualem-recitations-annotated")
    moshaf_ds = load_dataset(str(ds_path), name="moshaf_metadata")
    moshaf_ds.push_to_hub(repo_id, config_name="moshaf_metadata")

    reciters_ds = load_dataset(str(ds_path), name="reciters_metadata")
    reciters_ds.push_to_hub(repo_id, config_name="reciters_metadata")

    for moshaf in moshaf_ds["train"]:
        ds_name = f"moshaf_{moshaf['id']}"
        ds = load_dataset(str(ds_path), name=ds_name, num_proc=32)
        moshaf_attr = MoshafAttributes(**moshaf)

        ds = ds.map(
            add_phonetic_script_map,
            fn_kwargs={"moshaf": moshaf_attr},
            num_proc=32,
        )

        ds.push_to_hub(repo_id, config_name=ds_name)
