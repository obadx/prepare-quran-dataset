from pathlib import Path
import json

from datasets import load_dataset

from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool

if __name__ == "__main__":
    ds_path = Path("/cluster/users/shams035u1/data/mualem-recitations-annotated")
    reciter_pool = ReciterPool(ds_path / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, ds_path)

    moshaf_to_dup = {}
    for moshaf in moshaf_pool:
        ds = load_dataset(
            str(ds_path), name=f"moshaf_{moshaf.id}", split="train", num_proc=32
        )
        seg_ids = ds["segment_index"]
        segs = set()
        for seg in seg_ids:
            if seg not in segs:
                segs.add(seg)
            else:
                if moshaf.id not in moshaf_to_dup:
                    moshaf_to_dup[moshaf.id] = []
                moshaf_to_dup[moshaf.id].append(seg)

    with open("duplicate_segs.json", "w+") as f:
        json.dump(f, moshaf_to_dup, indent=2)
