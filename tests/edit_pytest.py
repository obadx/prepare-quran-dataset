import json

import pytest
from pydantic import ValidationError

from prepare_quran_dataset.annotate.edit import (
    MoshafEditConfig,
    Operation,
    MoshafDatasetInfo,
    DatasetShardOperations,
    plan_moshaf_edits,
)


@pytest.mark.parametrize(
    "moshaf_op_config, exp_shard_ops_list",
    [
        # update operation
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="update",
                        segment_index="112.0003",
                        new_tarteel_transcript="الحمد لله:",
                    )
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="update",
                            segment_index="112.0003",
                            new_tarteel_transcript="الحمد لله:",
                        )
                    ],
                )
            ],
        ),
        # single insert
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="insert",
                        segment_index="112.0003",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="الحمد لله:",
                        new_audio_file="hamo.wav",
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="112.0003",
                            new_segment_index="112.0003",
                            new_tarteel_transcript="الحمد لله:",
                            new_audio_file="hamo.wav",
                            new_start_seconds=3.22,
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0003",
                            new_segment_index="112.0004",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0005",
                        ),
                    ],
                )
            ],
        ),
        # single insert for last item
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="insert",
                        segment_index="112.0004",
                        new_tarteel_transcript="الحمد لله:",
                        new_audio_file="hamo.wav",
                        new_start_seconds=3.22,
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="112.0004",
                            new_segment_index="112.0004",
                            new_tarteel_transcript="الحمد لله:",
                            new_audio_file="hamo.wav",
                            new_start_seconds=3.22,
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0005",
                        ),
                    ],
                )
            ],
        ),
        # 2 inserts
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="insert",
                        segment_index="112.0001",
                        new_tarteel_transcript="الحمد لله:",
                        new_audio_file="hamo.wav",
                        new_start_seconds=3.22,
                    ),
                    Operation(
                        type="insert",
                        segment_index="112.0003",
                        new_tarteel_transcript="الحمد لله:",
                        new_audio_file="hamo.wav",
                        new_start_seconds=3.22,
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="112.0001",
                            new_segment_index="112.0001",
                            new_tarteel_transcript="الحمد لله:",
                            new_audio_file="hamo.wav",
                            new_start_seconds=3.22,
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0001",
                            new_segment_index="112.0002",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0002",
                            new_segment_index="112.0003",
                        ),
                        Operation(
                            type="insert",
                            segment_index="112.0003",
                            new_segment_index="112.0004",
                            new_tarteel_transcript="الحمد لله:",
                            new_audio_file="hamo.wav",
                            new_start_seconds=3.22,
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0003",
                            new_segment_index="112.0005",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0006",
                        ),
                    ],
                )
            ],
        ),
        # 2 inserts + 2 updates
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="insert",
                        segment_index="112.0001",
                        new_tarteel_transcript="الحمد لله:",
                        new_audio_file="hamo.wav",
                        new_start_seconds=3.22,
                    ),
                    Operation(
                        type="update",
                        segment_index="112.0001",
                        new_audio_file="hamo.wav",
                    ),
                    Operation(
                        type="insert",
                        segment_index="112.0003",
                        new_tarteel_transcript="الحمد لله:",
                        new_audio_file="hamo.wav",
                        new_start_seconds=3.22,
                    ),
                    Operation(
                        type="update",
                        segment_index="112.0004",
                        new_audio_file="hamo_update.wav",
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="112.0001",
                            new_segment_index="112.0001",
                            new_tarteel_transcript="الحمد لله:",
                            new_audio_file="hamo.wav",
                            new_start_seconds=3.22,
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0001",
                            new_audio_file="hamo.wav",
                            new_segment_index="112.0002",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0002",
                            new_segment_index="112.0003",
                        ),
                        Operation(
                            type="insert",
                            segment_index="112.0003",
                            new_segment_index="112.0004",
                            new_tarteel_transcript="الحمد لله:",
                            new_audio_file="hamo.wav",
                            new_start_seconds=3.22,
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0003",
                            new_segment_index="112.0005",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0006",
                            new_audio_file="hamo_update.wav",
                        ),
                    ],
                )
            ],
        ),
        # Sinlge Delete operation
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="delete",
                        segment_index="112.0002",
                    )
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="delete",
                            segment_index="112.0002",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0003",
                            new_segment_index="112.0002",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0003",
                        ),
                    ],
                )
            ],
        ),
        # Delete last item
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="delete",
                        segment_index="112.0004",
                    )
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="delete",
                            segment_index="112.0004",
                        ),
                    ],
                )
            ],
        ),
        # 2 delete operations
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="delete",
                        segment_index="112.0001",
                    ),
                    Operation(
                        type="delete",
                        segment_index="112.0002",
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="delete",
                            segment_index="112.0001",
                        ),
                        Operation(
                            type="delete",
                            segment_index="112.0002",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0003",
                            new_segment_index="112.0001",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0002",
                        ),
                    ],
                )
            ],
        ),
        # 2 delete operations and update
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="delete",
                        segment_index="112.0001",
                    ),
                    Operation(
                        type="delete",
                        segment_index="112.0002",
                    ),
                    Operation(
                        type="update",
                        segment_index="112.0003",
                        new_audio_file="hamo_update.wav",
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="delete",
                            segment_index="112.0001",
                        ),
                        Operation(
                            type="delete",
                            segment_index="112.0002",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0003",
                            new_segment_index="112.0001",
                            new_audio_file="hamo_update.wav",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0002",
                        ),
                    ],
                )
            ],
        ),
        # 1 insert + 1 delete
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="insert",
                        segment_index="112.0001",
                        new_audio_file="hamo_insert.wav",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="بسم الله الرحمن الرحيم",
                    ),
                    Operation(
                        type="delete",
                        segment_index="112.0003",
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="112.0001",
                            new_segment_index="112.0001",
                            new_audio_file="hamo_insert.wav",
                            new_start_seconds=3.22,
                            new_tarteel_transcript="بسم الله الرحمن الرحيم",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0001",
                            new_segment_index="112.0002",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0002",
                            new_segment_index="112.0003",
                        ),
                        Operation(
                            type="delete",
                            segment_index="112.0003",
                        ),
                    ],
                )
            ],
        ),
        # insert + isnsert + delete
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="insert",
                        segment_index="112.0001",
                        new_audio_file="hamo_insert_1.wav",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="بسم الله الرحمن الرحيم",
                    ),
                    Operation(
                        type="insert",
                        segment_index="112.0002",
                        new_audio_file="hamo_insert_2.wav",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="بسم الله الرحمن الرحيم",
                    ),
                    Operation(
                        type="delete",
                        segment_index="112.0003",
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="112.0001",
                            new_segment_index="112.0001",
                            new_audio_file="hamo_insert_1.wav",
                            new_start_seconds=3.22,
                            new_tarteel_transcript="بسم الله الرحمن الرحيم",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0001",
                            new_segment_index="112.0002",
                        ),
                        Operation(
                            type="insert",
                            segment_index="112.0002",
                            new_segment_index="112.0003",
                            new_audio_file="hamo_insert_2.wav",
                            new_start_seconds=3.22,
                            new_tarteel_transcript="بسم الله الرحمن الرحيم",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0002",
                            new_segment_index="112.0004",
                        ),
                        Operation(
                            type="delete",
                            segment_index="112.0003",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0005",
                        ),
                    ],
                )
            ],
        ),
        # insert + update + isnsert + delete + update
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="insert",
                        segment_index="112.0001",
                        new_audio_file="hamo_insert_1.wav",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="بسم الله الرحمن الرحيم",
                    ),
                    Operation(
                        type="insert",
                        segment_index="112.0002",
                        new_audio_file="hamo_insert_2.wav",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="بسم الله الرحمن الرحيم",
                    ),
                    Operation(
                        type="update",
                        segment_index="112.0002",
                        new_audio_file="hamo_update_2.wav",
                    ),
                    Operation(
                        type="delete",
                        segment_index="112.0003",
                    ),
                    Operation(
                        type="update",
                        segment_index="112.0004",
                        new_audio_file="hamo_update_4.wav",
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="112.0001",
                            new_segment_index="112.0001",
                            new_audio_file="hamo_insert_1.wav",
                            new_start_seconds=3.22,
                            new_tarteel_transcript="بسم الله الرحمن الرحيم",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0001",
                            new_segment_index="112.0002",
                        ),
                        Operation(
                            type="insert",
                            segment_index="112.0002",
                            new_segment_index="112.0003",
                            new_audio_file="hamo_insert_2.wav",
                            new_start_seconds=3.22,
                            new_tarteel_transcript="بسم الله الرحمن الرحيم",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0002",
                            new_segment_index="112.0004",
                            new_audio_file="hamo_update_2.wav",
                        ),
                        Operation(
                            type="delete",
                            segment_index="112.0003",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0005",
                            new_audio_file="hamo_update_4.wav",
                        ),
                    ],
                )
            ],
        ),
        # Multiple Shrads: insert + update + isnsert + delete + update
        (
            MoshafEditConfig(
                id="0.0",
                operations=[
                    Operation(
                        type="insert",
                        segment_index="112.0003",
                        new_audio_file="hamo_insert_1_1.wav",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="بسم الله الرحمن الرحيم",
                    ),
                    Operation(
                        type="insert",
                        segment_index="113.0001",
                        new_audio_file="hamo_insert_1.wav",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="بسم الله الرحمن الرحيم",
                    ),
                    Operation(
                        type="insert",
                        segment_index="113.0002",
                        new_audio_file="hamo_insert_2.wav",
                        new_start_seconds=3.22,
                        new_tarteel_transcript="بسم الله الرحمن الرحيم",
                    ),
                    Operation(
                        type="update",
                        segment_index="113.0002",
                        new_audio_file="hamo_update_2.wav",
                    ),
                    Operation(
                        type="delete",
                        segment_index="113.0003",
                    ),
                    Operation(
                        type="update",
                        segment_index="113.0004",
                        new_audio_file="hamo_update_4.wav",
                    ),
                ],
            ),
            [
                DatasetShardOperations(
                    path="shard_0",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="112.0003",
                            new_segment_index="112.0003",
                            new_audio_file="hamo_insert_1_1.wav",
                            new_start_seconds=3.22,
                            new_tarteel_transcript="بسم الله الرحمن الرحيم",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0003",
                            new_segment_index="112.0004",
                        ),
                        Operation(
                            type="update",
                            segment_index="112.0004",
                            new_segment_index="112.0005",
                        ),
                        Operation(
                            type="insert",
                            segment_index="113.0001",
                            new_segment_index="113.0001",
                            new_audio_file="hamo_insert_1.wav",
                            new_start_seconds=3.22,
                            new_tarteel_transcript="بسم الله الرحمن الرحيم",
                        ),
                        Operation(
                            type="update",
                            segment_index="113.0001",
                            new_segment_index="113.0002",
                        ),
                    ],
                ),
                DatasetShardOperations(
                    path="shard_1",
                    operations=[
                        Operation(
                            type="insert",
                            segment_index="113.0002",
                            new_segment_index="113.0003",
                            new_audio_file="hamo_insert_2.wav",
                            new_start_seconds=3.22,
                            new_tarteel_transcript="بسم الله الرحمن الرحيم",
                        ),
                        Operation(
                            type="update",
                            segment_index="113.0002",
                            new_segment_index="113.0004",
                            new_audio_file="hamo_update_2.wav",
                        ),
                        Operation(
                            type="delete",
                            segment_index="113.0003",
                        ),
                        Operation(
                            type="update",
                            segment_index="113.0004",
                            new_segment_index="113.0005",
                            new_audio_file="hamo_update_4.wav",
                        ),
                        Operation(
                            type="update",
                            segment_index="113.0005",
                            new_segment_index="113.0006",
                        ),
                    ],
                ),
            ],
        ),
        # # edge case insert at the end
        # (
        #     MoshafEditConfig(
        #         id="0.0",
        #         operations=[
        #             Operation(
        #                 type="insert",
        #                 segment_index="112.0005",
        #                 new_audio_file="hamo.wav",
        #               new_start_seconds=3.22,
        #                 new_tarteel_transcript="الحمد لله",
        #             )
        #         ],
        #     ),
        #     [
        #         DatasetShardOperations(
        #             path="shard_0",
        #             operations=[
        #                 Operation(
        #                     type="insert",
        #                     segment_index="112.0005",
        #                     new_segment_index="112.0005",
        #                     new_tarteel_transcript="الحمد لله",
        #                     new_audio_file="hamo.wav",
        #               new_start_seconds=3.22,
        #                 )
        #             ],
        #         )
        #     ],
        # ),
    ],
)
def test_plan_operations(
    moshaf_op_config: MoshafEditConfig, exp_shard_ops_list: list[DatasetShardOperations]
):
    sura_aya_count = {112: 5, 113: 6, 114: 7}
    ids = [
        f"{sura:03d}.{seg:04d}"
        for sura in sura_aya_count
        for seg in range(sura_aya_count[sura])
    ]
    # shrad_0: 112.0000, 112.0001, 112.0002, 112.0003, 112.0004, 113.0000, 113.0001
    # shrad_1: 113.0002, 113.0003, 113.0004, 113.0005, 114.0000, 114.0001 ... 114.0006
    shard_to_seg = {"shard_0": ids[:7], "shard_1": ids[7:]}
    seg_to_shard = {seg: shard for shard in shard_to_seg for seg in shard_to_seg[shard]}
    ds_info = MoshafDatasetInfo(
        idx_to_segment=ids,
        segment_to_idx={seg: idx for idx, seg in enumerate(ids)},
        shard_path_to_segment=shard_to_seg,
        segment_to_shard_path=seg_to_shard,
    )

    exp_shard_to_ops = {shard_op.path: shard_op for shard_op in exp_shard_ops_list}

    shard_ops_list = plan_moshaf_edits(ds_info, moshaf_op_config)

    print(
        f"Input Operations:\n{json.dumps(moshaf_op_config.model_dump(), ensure_ascii=False, indent=4)}"
    )
    print("Exepected Ouput")
    for out_shard_ops in exp_shard_ops_list:
        print(f"{json.dumps(out_shard_ops.model_dump(), ensure_ascii=False, indent=4)}")
    print("Output:")
    for out_shard_ops in shard_ops_list:
        print(f"{json.dumps(out_shard_ops.model_dump(), ensure_ascii=False, indent=4)}")

    for shard_ops in shard_ops_list:
        ex_shard_ops = exp_shard_to_ops[shard_ops.path]
        assert len(shard_ops.operations) == len(ex_shard_ops.operations)
        sorted_out_ops = sorted(shard_ops.operations, key=lambda x: x.segment_index)
        sorted_ex_ops = sorted(ex_shard_ops.operations, key=lambda x: x.segment_index)

        for out_op, ex_op in zip(sorted_out_ops, sorted_ex_ops):
            if out_op != ex_op:
                print(
                    f"Ouput Operation\n:{json.dumps(out_op.model_dump(), ensure_ascii=False, indent=4)}"
                )
                print(
                    f"Expected Operation\n:{json.dumps(ex_op.model_dump(), ensure_ascii=False, indent=4)}"
                )
                raise ValueError()


def test_invalid_insert():
    with pytest.raises(ValidationError):
        Operation(
            type="insert",
            segment_index="001.0001",
        )


def test_invalid_insert_missing_start():
    with pytest.raises(ValidationError):
        Operation(
            type="insert",
            segment_index="001.0001",
            new_audio_file="./hamo.wav",
        )


def test_invalid_insert_end_and_audio():
    with pytest.raises(ValidationError):
        Operation(
            type="insert",
            segment_index="001.0001",
            new_end_seconds=3.3,
            new_audio_file="./hamo.wav",
        )


def test_invalid_insert_start_and_end_and_aduio():
    with pytest.raises(ValidationError):
        Operation(
            type="insert",
            segment_index="001.0001",
            new_start_seconds=0,
            new_end_seconds=3.3,
            new_audio_file="./hamo.wav",
        )


def test_invalid_upate_start_and_end_and_aduio():
    with pytest.raises(ValidationError):
        Operation(
            type="update",
            segment_index="001.0001",
            new_end_seconds=3.3,
            new_start_seconds=2.1,
            new_audio_file="./hamo.wav",
        )


def test_invalid_upate_end_and_aduio():
    with pytest.raises(ValidationError):
        Operation(
            type="update",
            segment_index="001.0001",
            new_end_seconds=3.3,
            new_audio_file="./hamo.wav",
        )
