from typing import Literal, Optional, Any, Self
from pathlib import Path
import gc
import logging
from dataclasses import dataclass
import sys

import yaml
import librosa
from pydantic import BaseModel, model_validator
from datasets import Dataset


def decode_segment_index(
    segment_index: str,
) -> tuple[int, int]:
    """
    Args:
        segment_index (str): {sura absolute index}.{part index} Example: "001.0003"

    Returns:
        tuple[sura integer index, part integer index]
    """
    parts = segment_index.split(".")
    return int(parts[0]), int(parts[1])


OPERATION_TYPE_ORDER = {"insert": 0, "delete": 1, "update": 2}


class Operation(BaseModel):
    type: Literal["insert", "update", "delete"]
    segment_index: str
    new_segment_index: Optional[str] = None
    new_audio_file: Optional[str | Path] = None
    new_tarteel_transcript: Optional[list[str] | str] = None
    sample_rate: Optional[int] = 16000

    def model_post_init(
        self,
        context: Any,
    ):
        if self.new_tarteel_transcript is not None:
            if isinstance(self.new_tarteel_transcript, str):
                self.new_tarteel_transcript = [self.new_tarteel_transcript]

    @model_validator(mode="after")
    def check_addition(self) -> Self:
        if self.type == "insert":
            assert self.new_audio_file is not None, (
                "You have to add value for `new_audio_file` in `insert` operation"
            )
            assert self.new_tarteel_transcript is not None, (
                "You have to add value for `new_tarteel_transcript` in `insert` operation"
            )

        return self

    def _to_hf_audio(self, base_path: str | Path = None) -> dict:
        assert self.new_audio_file is not None, (
            "No file supllied to convert it into hugginface array"
        )

        path = Path(self.new_audio_file)
        if base_path is not None:
            path = Path(base_path) / self.new_audio_file

        wave, _ = librosa.core.load(path, sr=self.sample_rate, mono=True)

        return {"array": wave, "sampling_rate": self.sample_rate}

    def to_hf(self):
        item = {}
        if self.new_segment_index is not None:
            item["segment_index"] = self.new_segment_index

        if self.new_audio_file is not None:
            item["audio"] = self._to_hf_audio()

        if self.new_tarteel_transcript is not None:
            item["tarteel_transcript"] = self.new_tarteel_transcript

        return item


class MoshafEditConfig(BaseModel):
    id: str
    operations: list[Operation]

    def model_post_init(
        self,
        context: Any,
    ):
        # sort oprerations
        self.operations = sorted(
            self.operations,
            key=lambda o: (o.segment_index, OPERATION_TYPE_ORDER[o.type]),
        )

    @model_validator(mode="after")
    def check_duplicate_operations(self) -> Self:
        segment_index_to_ops = {}
        for op in self.operations:
            if op.segment_index not in segment_index_to_ops:
                segment_index_to_ops[op.segment_index] = set()
            segment_index_to_ops[op.segment_index].add(op.type)

        for segment_index, ops in segment_index_to_ops.items():
            if {"update", "delete"}.issubset(ops):
                raise ValueError(
                    f"You can not update and delete the same segment_index: `{segment_index}` for moshaf: `{self.id}`"
                )
            elif {"insert", "delete"}.issubset(ops):
                raise ValueError(
                    f"Instead of doing insert and delete for the same segemnt_index just use `update` operation segment_index: `{segment_index}` for moshaf: `{self.id}`"
                )

        return self

    def sura_format(self):
        """returns a dictionary: sura_index: segment_index, list(segment_operationsj)"""
        seg_to_operations: dict[str, list] = {}
        for op in self.operations:
            if op.segment_index not in seg_to_operations:
                seg_to_operations[op.segment_index] = []
            seg_to_operations[op.segment_index].append(op)

        # sorting operation for every segment
        seg_to_operations = {
            s: sorted(ops, key=lambda o: (OPERATION_TYPE_ORDER[o.type]))
            for s, ops in seg_to_operations.items()
        }

        sura_to_seg_to_ops = {}
        for seg, ops in seg_to_operations.items():
            sura_idx, sura_seg_idx = decode_segment_index(seg)
            if sura_idx not in sura_to_seg_to_ops:
                sura_to_seg_to_ops[sura_idx] = {}
            sura_to_seg_to_ops[sura_idx][sura_seg_idx] = ops

        # sorting segments for every sura
        sura_to_seg_to_ops = {
            s: dict(sorted(d.items())) for s, d in sura_to_seg_to_ops.items()
        }

        return sura_to_seg_to_ops


class DatasetShardOperations(BaseModel):
    path: str | Path
    operations: list[Operation]


class EditConfig(BaseModel):
    configs: list[MoshafEditConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EditConfig":
        # Load YAML data from file
        path = Path(path)
        yaml_data = path.read_text(encoding="utf-8")
        raw_data = yaml.safe_load(yaml_data)

        return cls(**raw_data)


@dataclass
class MoshafDatasetInfo:
    segment_to_shard_path: dict[str, Path | str]
    shard_path_to_segment: dict[str, list[str]]
    segment_to_idx: dict[str, int]
    idx_to_segment: list[str]


def load_moshaf_dataset_info(
    ds_path: Path, segment_column="segment_index"
) -> MoshafDatasetInfo:
    """Loads segment ids that was finished annotation"""
    ds_path = Path(ds_path)
    ids = []
    shard_path_to_segment: dict[str, set[str]] = {}
    for parquet_file in sorted(ds_path.glob("*parquet")):
        logging.info(f"Loading Annoated shard: {parquet_file}")
        ds = Dataset.from_parquet(str(parquet_file))
        shard_ids = sorted(ds[segment_column])
        ids += shard_ids
        shard_path_to_segment[str(parquet_file)] = shard_ids
        del ds
        gc.collect()

    ids = sorted(ids)
    logging.info(f"Total loadded examples: {len(ids)}")

    return MoshafDatasetInfo(
        segment_to_shard_path={
            seg: shard
            for shard in shard_path_to_segment
            for seg in shard_path_to_segment[shard]
        },
        shard_path_to_segment=shard_path_to_segment,
        segment_to_idx={seg: idx for idx, seg in enumerate(ids)},
        idx_to_segment=ids,
    )


def get_segment_format(sura_idx, aya_idx):
    return f"{sura_idx:03d}.{aya_idx:04d}"


def plan_moshaf_edits(
    ds_info: MoshafDatasetInfo,
    moshaf_edit_configs: MoshafEditConfig,
) -> list[DatasetShardOperations]:
    shard_to_ops: dict[str, list[Operation]] = {}

    def insert_shard_op(op: Operation):
        shard = ds_info.segment_to_shard_path[op.segment_index]
        if shard not in shard_to_ops:
            shard_to_ops[shard] = []
        shard_to_ops[shard].append(op)

    # sura to operation for insert and delete operations
    sura_to_aya_to_ops = moshaf_edit_configs.sura_format()

    for sura_idx in sura_to_aya_to_ops:
        offset = 0
        sura_aya_ids = sorted(sura_to_aya_to_ops[sura_idx].keys())

        # aya_idx is not aya but aya like (a part of sura member)
        for idx, aya_idx in enumerate(sura_to_aya_to_ops[sura_idx]):
            # opearations can for same aya can be:
            # insert, update (insert new item in this position and update the previous item in with new updates)
            # insert
            # update
            # delete
            operation: Operation = sura_to_aya_to_ops[sura_idx][aya_idx][0]

            if operation.type == "insert":
                assert len(sura_to_aya_to_ops[sura_idx][aya_idx]) in [1, 2]
                operation.new_segment_index = get_segment_format(
                    sura_idx, aya_idx + offset
                )

                insert_shard_op(operation)
                offset += 1
                if len(sura_to_aya_to_ops[sura_idx][aya_idx]) == 2:
                    assert sura_to_aya_to_ops[sura_idx][aya_idx][1].type == "update"
                    sura_to_aya_to_ops[sura_idx][aya_idx][
                        1
                    ].new_segment_index = get_segment_format(sura_idx, aya_idx + offset)
                    insert_shard_op(sura_to_aya_to_ops[sura_idx][aya_idx][1])
                elif offset != 0:
                    insert_shard_op(
                        Operation(
                            type="update",
                            segment_index=operation.segment_index,
                            new_segment_index=get_segment_format(
                                sura_idx, aya_idx + offset
                            ),
                        )
                    )

            elif operation.type == "delete":
                offset -= 1
                insert_shard_op(operation)

            elif operation.type == "update":
                if offset != 0:
                    operation.new_segment_index = get_segment_format(
                        sura_idx, aya_idx + offset
                    )
                insert_shard_op(operation)

            # Updating rest of aya indices in the same sura
            loop_aya_idx = aya_idx + 1
            loop_segment_index = get_segment_format(sura_idx, loop_aya_idx)

            next_aya_idx = (
                sura_aya_ids[idx + 1] if (idx + 1) < len(sura_aya_ids) else sys.maxsize
            )

            # updating trailing ayas with new segment index
            while (
                (offset != 0)
                and (loop_aya_idx < next_aya_idx)
                and (loop_segment_index in ds_info.segment_to_idx)
            ):
                insert_shard_op(
                    Operation(
                        type="update",
                        segment_index=loop_segment_index,
                        new_segment_index=get_segment_format(
                            sura_idx, loop_aya_idx + offset
                        ),
                    )
                )
                # ++ step
                loop_aya_idx += 1
                loop_segment_index = get_segment_format(sura_idx, loop_aya_idx)

    # Formating results to output
    shard_operataions: list[DatasetShardOperations] = []
    for op_shard, operations in shard_to_ops.items():
        shard_operataions.append(
            DatasetShardOperations(path=op_shard, operations=operations)
        )

    return shard_operataions


def apply_edits_map(batch, segment_index_to_ops: dict[str, list[Operation]]) -> dict:
    new_batch = {k: [] for k in batch}
    if batch["segment_index"][0] in segment_index_to_ops:
        for op in segment_index_to_ops[batch["segment_index"][0]]:
            if op.type == "delete":
                ...
            elif op.type in ["update", "insert"]:
                new_item = op.to_hf()
                for k in new_batch:
                    if k in new_item:
                        new_batch[k].append(new_item[k])
                    else:
                        new_batch[k] += batch[k]

    else:
        new_batch = batch

    return new_batch


def apply_ds_shard_edits(
    ds_shard_operations: DatasetShardOperations,
    num_proc=16,
):
    ds_shard = Dataset.from_parquet(ds_shard_operations.path)
    segment_index_to_ops: {str, list[Operation]} = {}

    for op in ds_shard_operations.operations:
        if op.type in ["delete", "update"]:
            segment_index = op.segment_index
        elif op.type == "insert":
            segment_index = op.new_segment_index

        if op.segment_index not in segment_index_to_ops:
            segment_index_to_ops[segment_index] = []
        segment_index_to_ops[segment_index].append(op)

    execute_order = {"delete": 0, "update": 1, "insert": 2}
    segment_index_to_ops = {
        s: sorted(ops, key=lambda o: execute_order[o.type])
        for s, ops in segment_index_to_ops.items()
    }

    ds_shard = ds_shard.map(
        apply_edits_map,
        batched=True,
        batch_size=1,
        num_proc=num_proc,
        fn_kwargs={"segment_index_to_ops": segment_index_to_ops},
    )
    ds_shard = ds_shard.sort("segment_index")
    logging.info(f"Saveing Shard: {ds_shard_operations.path}")
    ds_shard.to_parquet(ds_shard_operations.path)
