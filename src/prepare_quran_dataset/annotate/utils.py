from pathlib import Path
import os
import gc

from datasets import IterableDataset, Dataset
from tqdm import tqdm


def save_to_disk_split(
    dataset: IterableDataset,
    split_name: str,
    out_path: str | Path,
    samples_per_shard: int = 128,
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
        cache.append(item)

        if ((idx + 1) % samples_per_shard == 0) and idx != 0:
            shard_ds = Dataset.from_list(cache)
            shard_ds.to_parquet(
                out_path / f"{split_name}/train/shard_{shard_idx:0{5}}.parquet"
            )
            del shard_ds
            del cache
            gc.collect()
            cache = []
            shard_idx += 1

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
