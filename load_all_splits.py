from pathlib import Path

from datasets import load_dataset
import submitit

from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter
from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool


if __name__ == "__main__":
    # config
    ds_path = Path("/cluster/users/shams035u1/data/mualem-recitations-annotated")
    reciter_pool = ReciterPool(ds_path / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, ds_path)

    # Configure Slurm
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_account="shams035",
        slurm_partition="cpu",
        slurm_time="0-4:00:00",
        slurm_ntasks_per_node=1,
        cpus_per_task=16,
    )

    for moshaf in moshaf_pool:
        executor.update_parameters(
            slurm_job_name=f"L-{moshaf.id}",
            slurm_additional_parameters={
                # "output": f"QVADcpu_{split}_%j.out"  # %j = Slurm job ID
            },
        )
        job = executor.submit(
            lambda: load_dataset(
                str(ds_path), name=f"moshaf_{moshaf.id}", split="train"
            )
        )
        print(job.job_id)
