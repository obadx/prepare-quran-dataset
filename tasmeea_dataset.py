import argparse
import json
import logging
from pathlib import Path
import concurrent.futures
import threading
import os

from datasets import load_dataset
import submitit  # Optional for SLURM support
from quran_transcript import (
    tasmeea_sura_multi_part,
    check_sura_missing_parts,
    SegmentScripts,
)


from prepare_quran_dataset.construct.database import MoshafPool, ReciterPool


# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


GLOBAL_TASMEEA_PRAMS = {
    "overlap_words": 16,
    "window_words": 32,
    "acceptance_ratio": 0.85,
    "include_istiaatha": True,
    "include_bismillah": True,
    "include_sadaka": True,
    "multi_part_truncation_words": 3,
    "remove_spaces": True,
    "ignore_hamazat": True,
    "ignore_alef_maksoora": True,
    "remove_small_alef": True,
    "remove_tashkeel": True,
    "normalize_taat": True,
}

SURA_PRAMS = {
    "055": {
        "overlap_words": 1,
    }
}

MOSHAF_PRAMS = {
    "25.0": {
        "overlap_words": 25,
        "window_words": 50,
    }
}


def segment_scripts_to_dict(seg: SegmentScripts | None) -> dict:
    if seg is None:
        return {}

    return {
        "imalaey": seg.imalaey,
        "uthmani": seg.uthmani,
        "has_istiaatha": seg.has_istiaatha,
        "has_bismillah": seg.has_bismillah,
        "has_sadaka": seg.has_sadaka,
        "has_quran": seg.has_quran,
        "start_span": {
            "sura_idx": seg.start_span[0],
            "aya_idx": seg.start_span[1],
            "imlaey": seg.start_span[2].imlaey,
            "uthmani": seg.start_span[2].uthmani,
        }
        if seg.start_span is not None
        else None,
        "end_span": {
            "sura_idx": seg.end_span[0],
            "aya_idx": seg.end_span[1],
            "imlaey": seg.end_span[2].imlaey,
            "uthmani": seg.end_span[2].uthmani,
        }
        if seg.end_span is not None
        else None,
    }


def process_sura(
    moshaf_id: str, sura_id: str, trans: list[dict[str, str]]
) -> tuple[dict, dict]:
    """
    Args:
        trans: {segment_index: str, trateel_transcript: list[str]}
    """
    tasmeea = []
    errors = {}
    nones = []
    missings = []
    params = GLOBAL_TASMEEA_PRAMS
    if moshaf_id in MOSHAF_PRAMS:
        params = params | MOSHAF_PRAMS[moshaf_id]
    if sura_id in SURA_PRAMS:
        params = params | SURA_PRAMS[sura_id]
    logging.info(f"Tasmeea Params: {params}")

    text_inputs = [t["tarteel_transcript"] for t in trans]
    segment_ids = [t["segment_index"] for t in trans]
    outputs = tasmeea_sura_multi_part(text_inputs, sura_idx=int(sura_id), **params)

    for seg_idx, o in zip(segment_ids, outputs):
        tasmeea.append(
            {"segment_index": seg_idx, "ratio": o[1]} | segment_scripts_to_dict(o[0])
        )
        if o[0] is None:
            nones.append({"segment_index": seg_idx, "ratio": o[1]})

    # Missings
    missing_segments = check_sura_missing_parts(
        sura_idx=int(sura_id), fixed_segments=[o[0] for o in outputs]
    )
    for m in missing_segments:
        missings.append(segment_scripts_to_dict(m))

    if nones:
        errors["nones"] = nones
    if missings:
        errors["missings"] = missings

    return tasmeea, errors


def process_moshaf(
    moshaf_id: str,
    tasmeea_dir: Path,
    dataset_dir: Path,
    retry_surahs: list[str] | None = None,
    max_workers: int = 16,
):
    """Process a full moshaf with optional surah retries"""

    max_workers = min(max_workers, os.cpu_count())

    all_surahs = [f"{i:03d}" for i in range(1, 115)]  # 001 to 114
    ds = load_dataset(str(dataset_dir), name=f"moshaf_{moshaf_id}", split="train")
    segment_ids = ds["segment_index"]
    tarteel_transcripts = ds["tarteel_transcript"]
    sura_to_seg_to_tarteel = {}
    for seg_idx, trt in zip(segment_ids, tarteel_transcripts):
        sura = seg_idx.split(".")[0]
        if sura not in sura_to_seg_to_tarteel:
            sura_to_seg_to_tarteel[sura] = []
        sura_to_seg_to_tarteel[sura].append(
            {"segment_index": seg_idx, "tarteel_transcript": trt}
        )

    moshaf_out_dir = tasmeea_dir / moshaf_id
    moshaf_out_dir.mkdir(parents=True, exist_ok=True)

    tasmeea_path = moshaf_out_dir / "tasmeea.json"
    errors_path = moshaf_out_dir / "errors.json"

    # Load existing data if available
    existing_tasmeea = {}
    existing_errors = {}
    if tasmeea_path.exists():
        with open(tasmeea_path, "r") as f:
            existing_tasmeea = json.load(f)
    if errors_path.exists():
        with open(errors_path, "r") as f:
            existing_errors = json.load(f)

    # Determine surahs to process
    surahs_to_process = []
    if retry_surahs:
        surahs_to_process = retry_surahs
    else:
        for s_id in all_surahs:
            if s_id not in existing_tasmeea:
                surahs_to_process.append(s_id)

    # Process each surah
    current_tasmeea = existing_tasmeea.copy()
    current_errors = existing_errors.copy()

    # Thread-safe writing mechanism
    file_lock = threading.Lock()

    def _process_and_save(sura_id):
        nonlocal current_tasmeea, current_errors
        try:
            tasmeea, errors = process_sura(
                moshaf_id, sura_id, sura_to_seg_to_tarteel[sura_id]
            )

            with file_lock:
                current_tasmeea[sura_id] = tasmeea
                current_errors[sura_id] = errors

                # Save results after each surah completion
                with open(tasmeea_path, "w") as f:
                    json.dump(current_tasmeea, f, indent=2, ensure_ascii=False)
                with open(errors_path, "w") as f:
                    json.dump(current_errors, f, indent=2, ensure_ascii=False)

            if errors:
                logging.error(f"Errors in moshaf {moshaf_id} sura {sura_id}")

        except Exception as e:
            logging.error(f"Error processing {moshaf_id}/{sura_id}: {str(e)}")
            with file_lock:
                current_errors[sura_id] = str(e)
                if sura_id in current_tasmeea:
                    del current_tasmeea[sura_id]

                with open(errors_path, "w") as f:
                    json.dump(current_errors, f, indent=2, ensure_ascii=False)

    # Process surahs with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_and_save, sura_id) for sura_id in surahs_to_process
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Raise exceptions if any occurred


def parse_retry_args(retry_list: list) -> dict:
    """Parse retry arguments into moshaf->surahs mapping"""
    retry_map = {}
    if not retry_list:
        return retry_map

    for item in retry_list:
        if ":" in item:
            moshaf_id, surahs_str = item.split(":", 1)
            surahs = [s.strip() for s in surahs_str.split(",")]
            retry_map[moshaf_id] = surahs
        else:
            all_surahs = [f"{i:03d}" for i in range(1, 115)]  # 001 to 114
            retry_map[item] = all_surahs  # Full moshaf retry
    return retry_map


def main(args):
    setup_logging()
    tasmeea_dir = args.dataset_dir / "tasmeea"
    tasmeea_dir.mkdir(parents=True, exist_ok=True)
    retry_map = parse_retry_args(args.retry)

    reciter_pool = ReciterPool(args.dataset_dir / "reciter_pool.jsonl")
    moshaf_pool = MoshafPool(reciter_pool, args.dataset_dir)
    moshaf_ids = {m.id for m in moshaf_pool}

    # Determine processing mode
    if retry_map:
        logging.info(f"Retry mode for: {retry_map}")
        for m_id in retry_map:
            assert m_id in moshaf_ids, f"Moshaf id: {m_id} deos not exist in the pool"
        # process_list = [
        #     (m.id, retry_map.get(m.id)) for m in retry_map.keys() if m in moshaf_pool
        # ]
        process_list = [
            (m.id, retry_map.get(m.id)) if m.id in retry_map else (m.id, None)
            for m in moshaf_pool
        ]

    else:
        logging.info("Processing all mosahaf")
        process_list = [(m.id, None) for m in moshaf_pool]

    # Process using SLURM or locally
    if args.slurm:
        cpus_per_task = args.threads or 16
        # Configure Slurm
        executor = submitit.AutoExecutor(folder="logs")
        executor.update_parameters(
            slurm_account="shams035",
            slurm_partition="cpu",
            slurm_time="0-18:00:00",
            slurm_ntasks_per_node=1,
            cpus_per_task=16,
        )

        jobs = []
        for moshaf_id, surahs in process_list:
            job = executor.submit(
                process_moshaf,
                moshaf_id=moshaf_id,
                tasmeea_dir=tasmeea_dir,
                dataset_dir=args.dataset_dir,
                retry_surahs=surahs,
                max_workers=16,
            )
            jobs.append(job)
            logging.info(f"Submitted job for moshaf {moshaf_id}: {job.job_id}")

    else:  # Local processing
        for moshaf_id, surahs in process_list:
            print(moshaf_id, surahs)
            process_moshaf(
                moshaf_id=moshaf_id,
                tasmeea_dir=tasmeea_dir,
                dataset_dir=args.dataset_dir,
                retry_surahs=surahs,
                max_workers=args.threads,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tasmeea Processing Pipeline")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root dataset directory with moshaf subdirectories",
    )
    parser.add_argument(
        "--retry",
        nargs="*",
        help="Retry specific mosahaf/surahs. Format: <moshaf_id> or <moshaf_id>:surah1,surah2",
    )

    # SLURM options
    parser.add_argument(
        "--slurm", action="store_true", help="Use SLURM for distributed processing"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Max threads to use per moshaf (local and SLURM). Default: 32",
    )

    args = parser.parse_args()
    main(args)
