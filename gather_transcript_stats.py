from pathlib import Path
import json

from datasets import load_dataset

from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool
from quran_transcript import normalize_aya


if __name__ == "__main__":
    # config
    ds_path = Path("/cluster/users/shams035u1/data/mualem-recitations-annotated")
    reciter_pool = ReciterPool(ds_path / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, ds_path)

    lens = {}
    for moshaf in moshaf_pool:
        ds = load_dataset(ds_path, name=f"moshaf_{moshaf.id}", split="train")
        trans = ds["tarteel_transcript"]
        sura_list = ds["sura_or_aya_index"]
        lens[moshaf.id] = {}
        for sura_idx, ts in zip(trans, sura_list):
            for t in ts:
                norm_t = normalize_aya(
                    t,
                    remove_spaces=True,
                    remove_tashkeel=True,
                    ignore_alef_maksoora=True,
                    ignore_hamazat=True,
                )
                if sura_idx not in lens[moshaf.id]:
                    lens[moshaf.id][sura_idx] = []
                lens[moshaf.id][sura_idx].append(len(norm_t))

    with open("trans_lens.json", "r") as f:
        json.dump(lens, f)
