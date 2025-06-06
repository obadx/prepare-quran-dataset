from pathlib import Path

from datasets import load_dataset

from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool


if __name__ == "__main__":
    # config
    ds_path = Path("/cluster/users/shams035u1/data/mualem-recitations-annotated")
    reciter_pool = ReciterPool(ds_path / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, ds_path)

    for moshaf in moshaf_pool:
        ds = load_dataset(str(ds_path), name=f"moshaf_{moshaf.id}", split="train")
        print(ds)
