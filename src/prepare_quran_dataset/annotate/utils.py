from pathlib import Path
import os
import gc
from typing import Optional
import logging

from datasets import IterableDataset, Dataset
from tqdm import tqdm


def load_segment_ids(ds_path: Path, segment_column="segment_index") -> set[str]:
    """Loads segment ids that was finished annotation"""
    ds_path = Path(ds_path)
    ids = []
    for parquet_file in ds_path.glob("*parquet"):
        ds = Dataset.from_parquet(parquet_file)
        ids += ds[segment_column]
        del ds
        gc.collect()

    return set(ids)


def save_to_disk_split(
    dataset: IterableDataset,
    split_name: str,
    out_path: str | Path,
    samples_per_shard: int = 128,
    annotated_segment_ids: Optional[set[str]] = None,
    segment_column="segment_index",
):
    """save an Iterable hugginfce dataset onto disk to avoid memory overfill"""
    assert isinstance(dataset, IterableDataset), (
        f"We only support IterableDatset we got {type(dataset)}"
    )

    out_path = Path(out_path)

    # create directory structure
    os.makedirs(out_path, exist_ok=True)

    # loop to save as parquet format
    cache = []
    shard_idx = 0
    for idx, item in tqdm(enumerate(dataset)):
        if annotated_segment_ids:
            if item[segment_column] not in annotated_segment_ids:
                cache.append(item)
        else:
            cache.append(item)

        if ((idx + 1) % samples_per_shard == 0) and idx != 0:
            if cache:
                shard_path = (
                    out_path / f"{split_name}/train/shard_{shard_idx:0{5}}.parquet"
                )
                shard_ds = Dataset.from_list(cache)
                shard_ds.to_parquet(shard_path)
                del shard_ds
                del cache
                gc.collect()
                cache = []
                shard_idx += 1
            else:
                logging.info(
                    f"Skipping shard: {shard_path} as it was annotated previously"
                )

    # rest of the items
    if cache:
        shard_ds = Dataset.from_list(cache)
        shard_ds.to_parquet(
            out_path / f"{split_name}/train/shard_{shard_idx:0{5}}.parquet"
        )
        del shard_ds
        del cache
        gc.collect()
        cache = []
        shard_idx += 1
