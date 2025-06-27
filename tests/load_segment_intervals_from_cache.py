from pathlib import Path

from prepare_quran_dataset.construct.database import ReciterPool, MoshafPool
from prepare_quran_dataset.annotate.main import load_segment_intervals_from_cache

if __name__ == "__main__":
    ds_path = Path("../quran-dataset")
    reciter_pool = ReciterPool(ds_path / "reciter_pool.jsonl")
    moshab_pool = MoshafPool(reciter_pool, ds_path)

    # moshaf_id = "4.0"
    # segment_to_interval = load_segment_intervals_from_cache(
    #     moshab_pool[moshaf_id],
    #     f".segment_cache/{moshaf_id}",
    #     ds_path / f"dataset/{moshaf_id}/metadata.jsonl",
    # )
    # print(len(segment_to_interval))
    # print(segment_to_interval["002.0409"])

    moshaf_id = "7.0"
    segment_to_interval = load_segment_intervals_from_cache(
        moshab_pool[moshaf_id],
        f".segment_cache/{moshaf_id}",
        ds_path / f"dataset/{moshaf_id}/metadata.jsonl",
    )
    print(len(segment_to_interval))
    print(segment_to_interval["002.0409"])
    print(segment_to_interval["021.0151"])
    print(segment_to_interval["021.0152"])

    # moshaf_id = "0.0"
    # segment_to_interval = load_segment_intervals_from_cache(
    #     moshab_pool[moshaf_id],
    #     f".segment_cache/{moshaf_id}",
    #     ds_path / f"dataset/{moshaf_id}/metadata.jsonl",
    # )
    # print(len(segment_to_interval))
    # print(segment_to_interval["002.0409"])

    # moshaf_id = "2.0"
    # segment_to_interval = load_segment_intervals_from_cache(
    #     moshab_pool[moshaf_id],
    #     f".segment_cache/{moshaf_id}",
    #     ds_path / f"dataset/{moshaf_id}/metadata.jsonl",
    # )
    # print(len(segment_to_interval))
    # print(segment_to_interval["002.0409"])
