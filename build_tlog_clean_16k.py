# WARN: Needs review
"""Build a local 16 kHz Arrow version of the ``tarteel-ai/tlog`` clean subset.

Downloads the ``clean`` config parquet shards one by one, downsamples the audio to
16 kHz mono FLAC bytes (librosa + soundfile, same loading pattern as
``train_streaming.py``), and stores the result as a ``save_to_disk``-style dataset:
one Arrow IPC file per processed shard plus ``state.json`` / ``dataset_info.json``.

Each row is enriched with metadata parsed from the audio file name (``surah_ayah_id``):

- ``surah`` (int32): the surah number, e.g. 83.
- ``ayah`` (int32): the ayah number, e.g. 2.
- ``id`` (string): the full file name stem, e.g. ``83_2_7538355371``.
- ``predicted_phonemes`` (string): joined from the per-shard inference results in
  ``tlog_inference_results_slim/clean-XXXXX-of-00411.results.jsonl``, keyed by ``id``.
  Rows without a non-empty ``predicted_phonemes`` (shard has no results file, id not
  present, or failed inference) are dropped. Shards whose results file is missing are
  skipped entirely, so their raw parquet is never downloaded.

The final dataset lives at ``$HF_HOME/datasets/tlog-clean-16k`` and is loaded with
``Dataset.load_from_disk("$HF_HOME/datasets/tlog-clean-16k")``. The Arrow files ARE
the dataset, so nothing is ever regenerated. The downloaded raw shards are kept only
in a temporary directory and deleted right after each shard is processed and saved.
Processing is resumable: already-completed shards are skipped on re-run.

Usage:
    uv run python build_tlog_clean_16k.py --test            # 1-shard end-to-end smoke test
    uv run python build_tlog_clean_16k.py                   # full run (all clean shards)
"""

import argparse
import dataclasses
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import datasets
import librosa
import pyarrow as pa
import pyarrow.compute as pc
import soundfile as sf
from datasets import Audio, Dataset, SplitInfo, load_dataset, load_from_disk
from datasets.arrow_dataset import update_metadata_with_features
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import login as hf_login
from tqdm import tqdm

REPO_ID = "tarteel-ai/tlog"
CONFIG = "clean"
TARGET_SR = 16000

DEFAULT_PHONEMES_DIR = Path(__file__).parent / "tlog_inference_results_slim"

OUTPUT_DIR_NAME = "tlog-clean-16k"
TEST_DIR_NAME = "tlog-clean-16k-test"
STATE_FILENAME = ".state.json"
LOADER_STATE_FILENAME = "state.json"
INFO_FILENAME = "dataset_info.json"

SKIPPED = object()


def load_secrets() -> str:
    """Load ``HUGGINGFACE_TOKEN`` from ``.env`` and log in to the Hugging Face Hub.

    Returns:
        The Hugging Face token.
    """
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN not found in .env")
    hf_login(token=token)
    return token


def list_clean_shards(repo_id: str, token: str) -> list[tuple[int, str]]:
    """List the parquet shards of the ``clean`` config.

    Args:
        repo_id: Hugging Face dataset repository id.
        token: Hugging Face token (needed because the dataset is gated).

    Returns:
        Sorted list of ``(shard_index, filename)`` pairs where ``filename`` is
        e.g. ``data/clean-00172-of-00411.parquet``.
    """
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset")

    def is_clean_shard(filename: str) -> bool:
        return filename.endswith(".parquet") and Path(filename).name.startswith(
            "clean-"
        )

    shard_files = sorted(f for f in files if is_clean_shard(f))

    def shard_index(filename: str) -> int:
        return int(Path(filename).stem.split("-")[1])

    return [(shard_index(f), f) for f in sorted(shard_files, key=shard_index)]


def smallest_clean_shard(repo_id: str, token: str) -> tuple[int, str]:
    """Pick the smallest ``clean`` parquet shard (fastest download for tests).

    Args:
        repo_id: Hugging Face dataset repository id.
        token: Hugging Face token.

    Returns:
        ``(shard_index, filename)`` of the smallest shard.
    """
    api = HfApi(token=token)
    candidates = []
    for f in api.list_repo_tree(
        repo_id, recursive=True, expand=True, repo_type="dataset", token=token
    ):
        rfilename = getattr(f, "rfilename", None)
        if (
            rfilename
            and rfilename.endswith(".parquet")
            and Path(rfilename).name.startswith("clean-")
            and getattr(f, "size", 0) > 0
        ):
            candidates.append(f)
    if not candidates:
        raise RuntimeError(f"No clean parquet files found in {repo_id}")
    smallest = min(candidates, key=lambda f: f.size)
    filename = smallest.rfilename
    return int(Path(filename).stem.split("-")[1]), filename


def load_state(state_path: Path) -> dict[int, int]:
    """Load the resume progress row counts.

    Args:
        state_path: Path of the JSON state file.

    Returns:
        ``shard_lengths`` mapping shard index to number of written rows. Which
        shards are finished is derived from the Arrow files on disk instead.
    """
    if not state_path.exists():
        return {}
    with open(state_path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.get("shard_lengths", {}).items()}


def save_state(
    state_path: Path, completed: set[int], shard_lengths: dict[int, int], total: int
) -> None:
    """Persist the resume progress (completed shards and their row counts).

    Args:
        state_path: Path of the JSON state file to write.
        completed: Set of completed shard indices.
        shard_lengths: Mapping of shard index to number of written rows.
        total: Total number of shards.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total": total,
        "completed": sorted(completed),
        "shard_lengths": {str(k): v for k, v in sorted(shard_lengths.items())},
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def downsample_example(
    example: dict[str, Any], target_sr: int = TARGET_SR
) -> dict[str, Any]:
    """Downsample the audio of one example to ``target_sr`` mono FLAC bytes.

    Reads the raw audio from bytes (``Audio(decode=False)``) exactly like
    ``train_streaming.py`` and re-encodes the resampled waveform as 16-bit FLAC,
    so the result round-trips through ``librosa.load(io.BytesIO(...), sr=target_sr)``.

    The ``tlog`` parquet shards carry the real audio in ``audio["bytes"]`` while
    ``audio["path"]`` is only a file name (no local file), so bytes are always
    used. Rows whose bytes are missing, empty or undecodable are left in place
    with ``audio["bytes"]`` set to ``None`` so callers can filter them out
    afterwards (this mirrors how ``tlog`` already encodes missing audio).

    Args:
        example: A dataset row whose ``audio`` column is
            ``{"bytes": <raw bytes or None>, "path": <str or None>}``.
        target_sr: Target sample rate.

    Returns:
        The example with ``audio`` replaced by ``{"bytes": <16k FLAC>, "path": None}``,
        or ``{"bytes": None, "path": <original path>}`` if the audio was unusable.
    """
    audio = example["audio"]
    if not audio["bytes"]:
        return {**example, "audio": {"bytes": None, "path": audio["path"]}}
    try:
        wav, _ = librosa.load(io.BytesIO(audio["bytes"]), sr=target_sr, mono=True)
        buf = io.BytesIO()
        sf.write(buf, wav, target_sr, format="FLAC", subtype="PCM_16")
        return {**example, "audio": {"bytes": buf.getvalue(), "path": None}}
    except Exception:
        return {**example, "audio": {"bytes": None, "path": audio["path"]}}


def has_embedded_audio(example: dict[str, Any]) -> bool:
    """Return ``True`` if the example's audio is embedded as non-empty bytes.

    Args:
        example: A dataset row with an ``audio`` column.

    Returns:
        Whether ``audio["bytes"]`` is truthy (some ``tlog`` rows carry empty
        bytes that libsndfile cannot open).
    """
    return bool(example["audio"]["bytes"])


def parse_audio_path(path: str) -> tuple[int, int, str]:
    """Parse ``surah``, ``ayah`` and ``id`` from an audio file name.

    The ``tlog`` audio file names look like ``83_2_7538355371.flac``, i.e.
    ``<surah>_<ayah>_<hash>.flac``. ``id`` is the full stem.

    Args:
        path: Audio file name (as stored in ``audio["path"]``).

    Returns:
        ``(surah, ayah, id)``.
    """
    stem = Path(path).stem
    parts = stem.split("_")
    return int(parts[0]), int(parts[1]), stem


def load_phonemes_map(shard_filename: str, phonemes_dir: Path) -> dict[str, str] | None:
    """Build the ``id -> predicted_phonemes`` reverse lookup for one shard.

    Args:
        shard_filename: Repo-relative parquet filename of the shard
            (e.g. ``data/clean-00172-of-00411.parquet``).
        phonemes_dir: Directory holding the per-shard inference results
            (e.g. ``clean-00172-of-00411.results.jsonl``).

    Returns:
        Mapping of audio id (file name stem) to ``predicted_phonemes``, or ``None``
        if the shard has no results file (in which case the caller leaves the
        ``predicted_phonemes`` column empty instead of failing).
    """
    results_name = Path(shard_filename).name.replace(".parquet", ".results.jsonl")
    results_path = phonemes_dir / results_name
    if not results_path.exists():
        print(
            f"no inference results for {results_name}, predicted_phonemes will be None"
        )
        return None
    phonemes: dict[str, str | None] = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            phonemes[Path(record["audio_path"]).stem] = record.get("predicted_phonemes")
    return phonemes


def add_path_metadata(example: dict[str, Any]) -> dict[str, Any]:
    """Add ``surah``, ``ayah`` and ``id`` columns parsed from the audio file name.

    Args:
        example: A dataset row with an ``audio`` column (raw or downsampled).

    Returns:
        The example with the ``surah``/``ayah``/``id`` keys added.
    """
    surah, ayah, audio_id = parse_audio_path(example["audio"]["path"])
    return {**example, "surah": surah, "ayah": ayah, "id": audio_id}


def add_predicted_phonemes(
    example: dict[str, Any], phonemes_map: dict[str, str] | None
) -> dict[str, Any]:
    """Add the ``predicted_phonemes`` column via the ``id -> phonemes`` lookup.

    Args:
        example: A dataset row (must already have the ``id`` column).
        phonemes_map: Reverse lookup built by ``load_phonemes_map``, or ``None``
            when the shard has no inference results.

    Returns:
        The example with ``predicted_phonemes`` set (``None`` when unavailable).
    """
    if phonemes_map is None:
        return {**example, "predicted_phonemes": None}
    return {**example, "predicted_phonemes": phonemes_map.get(example["id"])}


def _write_arrow(ds: Dataset, out_path: Path) -> None:
    """Write a dataset to an Arrow IPC file, preserving its ``huggingface`` metadata.

    ``Dataset.filter`` keeps a lazy ``_indices`` map instead of physically removing
    rows, so the filtered rows are materialized here with ``take`` before writing.

    The metadata carries the ``Audio`` features so that ``load_from_disk`` recovers
    them without relying on ``dataset_info.json``.

    Args:
        ds: Dataset to write (must fit in memory; per-shard tables are small).
        out_path: Destination path of the Arrow IPC file.
    """
    table = ds.data.table
    if ds._indices is not None:
        table = table.take(ds._indices.table.column(0))
    table = update_metadata_with_features(table, ds.features)
    with pa.OSFile(str(out_path), "wb") as sink:
        writer = pa.ipc.new_stream(sink, table.schema)
        try:
            writer.write_table(table)
        finally:
            writer.close()


def process_shard(
    shard_path: str,
    out_path: Path,
    num_proc: int,
    phonemes_map: dict[str, str] | None,
) -> int:
    """Downsample one downloaded raw shard and write its Arrow file.

    The raw rows are enriched with ``surah``/``ayah``/``id`` parsed from the audio
    file name and with ``predicted_phonemes`` from the shard's inference results.
    Rows without a non-empty ``predicted_phonemes`` are dropped.

    Args:
        shard_path: Local path of the downloaded raw parquet shard.
        out_path: Destination path of the processed Arrow shard.
        num_proc: Number of worker processes for the downsampling map.
        phonemes_map: ``id -> predicted_phonemes`` lookup, or ``None`` when the
            shard has no inference results.

    Returns:
        Number of rows written.
    """
    with tempfile.TemporaryDirectory(prefix="raw-arrow-") as tmp_cache:
        ds: Dataset = load_dataset(
            "parquet",
            data_files=[shard_path],
            split="train",
            cache_dir=tmp_cache,
        )
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.map(
            add_path_metadata,
            num_proc=num_proc,
            desc=f"adding path metadata to {Path(shard_path).name}",
        )
        if "audio" in ds.column_names:
            num_before = ds.num_rows
            ds = ds.filter(
                has_embedded_audio,
                num_proc=1,
                desc="dropping rows without embedded audio",
            )
            if ds.num_rows != num_before:
                print(f"dropped {num_before - ds.num_rows} rows without embedded audio")
        ds = ds.map(
            downsample_example,
            num_proc=num_proc,
            desc=f"downsampling {Path(shard_path).name}",
        )
        if "audio" in ds.column_names:
            num_before = ds.num_rows
            ds = ds.filter(
                has_embedded_audio,
                num_proc=1,
                desc="dropping rows that failed to decode",
            )
            if ds.num_rows != num_before:
                print(f"dropped {num_before - ds.num_rows} rows that failed to decode")
        ds = ds.map(
            add_predicted_phonemes,
            num_proc=num_proc,
            desc=f"joining predicted phonemes for {Path(shard_path).name}",
            fn_kwargs={"phonemes_map": phonemes_map},
        )
        num_before = ds.num_rows
        ds = ds.filter(
            lambda ex: bool(ex["predicted_phonemes"]),
            num_proc=1,
            desc="dropping rows without predicted phonemes",
        )
        if ds.num_rows != num_before:
            print(f"dropped {num_before - ds.num_rows} rows without predicted phonemes")
        _write_arrow(ds, out_path)
        return ds.num_rows


def process_one_shard(
    shard_index: int,
    filename: str,
    total_shards: int,
    token: str,
    output_dir: Path,
    num_proc: int,
    download_cache_root: Path,
    phonemes_dir: Path,
) -> int | None | object:
    """Download, downsample, save and clean up one shard.

    The downloaded raw shard (and its per-shard caches) lives in a temporary
    directory that is removed before returning, so no raw data is kept on disk.
    Shards without an inference results file are skipped before the raw parquet
    is downloaded (their rows would all be dropped anyway).

    Args:
        shard_index: Zero-based index of the shard.
        filename: Repo-relative parquet filename of the shard.
        total_shards: Total number of shards (used in the output name).
        token: Hugging Face token.
        output_dir: Directory where the processed Arrow shard is written.
        num_proc: Number of worker processes for the downsampling map.
        download_cache_root: Root directory for per-shard download caches.
        phonemes_dir: Directory holding the per-shard inference results.

    Returns:
        Number of rows written, ``None`` if the shard was already processed, or
        ``SKIPPED`` if the shard has no inference results and was not downloaded.
    """
    out_path = output_dir / f"data-{shard_index:05d}-of-{total_shards:05d}.arrow"
    if out_path.exists():
        return None

    phonemes_map = load_phonemes_map(filename, phonemes_dir)
    if phonemes_map is None:
        print(
            f"skipping {Path(filename).name}: no inference results (parquet not downloaded)"
        )
        return SKIPPED

    with tempfile.TemporaryDirectory(
        dir=download_cache_root, prefix="shard-"
    ) as dl_dir:
        cached_path = hf_hub_download(
            REPO_ID,
            filename,
            repo_type="dataset",
            token=token,
            cache_dir=dl_dir,
        )
        incomplete_path = output_dir / f".{out_path.name}.incomplete"
        if incomplete_path.exists():
            incomplete_path.unlink()
        rows = process_shard(cached_path, incomplete_path, num_proc, phonemes_map)
        os.replace(incomplete_path, out_path)

    return rows


def write_loader_state(
    output_dir: Path, completed: set[int], total: int, fingerprint: str | None = None
) -> None:
    """Write the ``state.json`` consumed by ``load_from_disk``.

    Lists one entry per completed shard Arrow file, sorted by shard index so that
    ``load_from_disk`` concatenates the data in order.

    Args:
        output_dir: Dataset directory.
        completed: Set of completed shard indices.
        total: Total number of shards.
        fingerprint: Optional whole-dataset fingerprint (filled in at finalize).
    """
    data_files = [
        {"filename": f"data-{i:05d}-of-{total:05d}.arrow"} for i in sorted(completed)
    ]
    state = {
        "_fingerprint": fingerprint,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": "train",
        "_data_files": data_files,
    }
    with open(output_dir / LOADER_STATE_FILENAME, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def compute_fingerprint(
    completed: set[int], total: int, shard_lengths: dict[int, int]
) -> str:
    """Compute a stable whole-dataset fingerprint from the shard files.

    Used purely as the dataset cache key stored in ``state.json``; nothing in
    ``load_from_disk`` validates it against the data.

    Args:
        completed: Set of completed shard indices.
        total: Total number of shards.
        shard_lengths: Mapping of shard index to number of rows.

    Returns:
        A deterministic SHA-1 hex digest.
    """
    h = hashlib.sha1()
    for i in sorted(completed):
        h.update(
            f"data-{i:05d}-of-{total:05d}.arrow:{shard_lengths.get(i, 0)}".encode(
                "utf-8"
            )
        )
    return h.hexdigest()


def finalize_dataset_info(
    output_dir: Path, total: int, shard_lengths: dict[int, int]
) -> None:
    """Write ``dataset_info.json`` and the finalized ``state.json``.

    Loads the first shard Arrow file for its features (recovered from the
    ``huggingface`` schema metadata) and patches in the split totals tracked
    during the run.

    Args:
        output_dir: Dataset directory.
        total: Total number of shards.
        shard_lengths: Mapping of shard index to number of written rows.
    """
    if not shard_lengths:
        raise RuntimeError("cannot finalize an empty dataset")
    completed = set(shard_lengths)
    first_index = min(completed)
    first_path = output_dir / f"data-{first_index:05d}-of-{total:05d}.arrow"
    sample = Dataset.from_file(str(first_path))
    info = sample.info.copy()

    lengths = [shard_lengths[i] for i in sorted(completed)]
    total_bytes = sum(p.stat().st_size for p in output_dir.glob("data-*.arrow"))
    info.splits = {
        "train": SplitInfo(
            name="train",
            dataset_name=OUTPUT_DIR_NAME,
            num_examples=sum(lengths),
            num_bytes=total_bytes,
            shard_lengths=lengths,
        )
    }

    with open(output_dir / INFO_FILENAME, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(info), f, indent=2, sort_keys=True)

    fingerprint = compute_fingerprint(completed, total, shard_lengths)
    write_loader_state(output_dir, completed, total, fingerprint=fingerprint)


def run_pipeline(
    token: str,
    output_dir: Path,
    num_proc: int,
    shards: list[tuple[int, str]],
    phonemes_dir: Path,
) -> int:
    """Process a list of shards into ``output_dir`` with resume support.

    Shards without an inference results file are skipped (no download); they are
    not recorded as completed, so a later run re-checks and picks them up if the
    results file appears.

    Args:
        token: Hugging Face token.
        output_dir: Directory that will hold the processed Arrow shards.
        num_proc: Number of worker processes per shard.
        shards: Sorted ``(shard_index, filename)`` pairs to process.
        phonemes_dir: Directory holding the per-shard inference results.

    Returns:
        Total number of rows written.
    """
    total = len(shards)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILENAME
    shard_lengths = load_state(state_path)

    completed: set[int] = set()
    for shard_index, _filename in shards:
        data_path = output_dir / f"data-{shard_index:05d}-of-{total:05d}.arrow"
        if data_path.exists():
            completed.add(shard_index)
            if shard_index not in shard_lengths:
                shard_lengths[shard_index] = Dataset.from_file(str(data_path)).num_rows

    pending = [s for s in shards if s[0] not in completed]
    print(
        f"total shards: {total}, already done: {total - len(pending)}, remaining: {len(pending)}"
    )

    with tempfile.TemporaryDirectory(prefix="downloads-") as download_cache_root:
        skipped = 0
        for shard_index, filename in tqdm(pending, desc="shards", unit="shard"):
            rows = process_one_shard(
                shard_index=shard_index,
                filename=filename,
                total_shards=total,
                token=token,
                output_dir=output_dir,
                num_proc=num_proc,
                download_cache_root=Path(download_cache_root),
                phonemes_dir=phonemes_dir,
            )
            if rows is SKIPPED:
                skipped += 1
                continue
            if rows is None:
                rows = shard_lengths[shard_index]
            completed.add(shard_index)
            shard_lengths[shard_index] = rows
            save_state(state_path, completed, shard_lengths, total)
            write_loader_state(output_dir, completed, total)
            print(f"saved data-{shard_index:05d}-of-{total:05d}.arrow ({rows} rows)")

    finalize_dataset_info(output_dir, total, shard_lengths)
    total_rows = sum(shard_lengths.values())
    print(
        f"dataset finalized at {output_dir}: {total_rows} rows, {len(completed)}/{total} shards "
        f"({skipped} skipped), load with Dataset.load_from_disk('{output_dir}')"
    )
    return total_rows


def verify_loading(output_dir: Path, expected_rows: int) -> None:
    """Load the finalized dataset and check that it is intact.

    Asserts the expected row count, that only Arrow data files exist (no parquet
    leftovers), that the enriched columns are present and typed, and that the
    stored audio round-trips through librosa at 16 kHz.

    Args:
        output_dir: Directory of the finalized Arrow dataset.
        expected_rows: Expected total number of rows.
    """
    stray = sorted(output_dir.glob("*.parquet"))
    if stray:
        raise AssertionError(f"unexpected parquet files in output directory: {stray}")

    ds = load_from_disk(str(output_dir))
    if ds.num_rows != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, got {ds.num_rows}")
    if "audio" not in ds.features:
        raise AssertionError("missing 'audio' feature")
    for column in ("surah", "ayah", "id", "predicted_phonemes"):
        if column not in ds.features:
            raise AssertionError(f"missing '{column}' column")

    row = ds[0]
    if not isinstance(row["surah"], int) or not isinstance(row["ayah"], int):
        raise AssertionError(
            f"surah/ayah not integers: {row['surah']!r}, {row['ayah']!r}"
        )
    if not isinstance(row["id"], str) or not row["id"]:
        raise AssertionError(f"id not a non-empty string: {row['id']!r}")
    if not isinstance(row["predicted_phonemes"], str) or not row["predicted_phonemes"]:
        raise AssertionError(
            f"predicted_phonemes not a non-empty string: {row['predicted_phonemes']!r}"
        )

    phonemes = ds.data.table.column("predicted_phonemes")
    if phonemes.null_count != 0:
        raise AssertionError(
            f"found {phonemes.null_count} rows with null predicted_phonemes"
        )
    if pc.any(pc.equal(phonemes, "")).as_py():
        raise AssertionError("found rows with empty-string predicted_phonemes")

    ds = ds.cast_column("audio", Audio(decode=False))
    audio = ds[0]["audio"]
    if not audio["bytes"]:
        raise AssertionError("first row has empty audio bytes")
    wav, sr = librosa.load(io.BytesIO(audio["bytes"]), sr=TARGET_SR, mono=True)
    if sr != TARGET_SR or wav.size == 0:
        raise AssertionError(
            f"audio did not round-trip at {TARGET_SR} Hz (got {sr} Hz, {wav.size} samples)"
        )

    print(f"PASS: load_from_disk loaded {ds.num_rows} rows from {output_dir}")
    print(
        f"  enriched columns: surah={row['surah']}, ayah={row['ayah']}, id={row['id']}, "
        f"predicted_phonemes={len(row['predicted_phonemes'])} chars"
    )
    print(f"  audio feature round-trips at {sr} Hz via librosa")
    print(f"  load it with:  ds = load_from_disk('{output_dir}')")


def main() -> None:
    """Entry point: parse arguments and run the test or full pipeline."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test", action="store_true", help="run a single-shard end-to-end smoke test"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="worker processes per shard",
    )
    parser.add_argument(
        "--cleanup-test",
        action="store_true",
        help="delete the test output directory afterwards",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="override the output directory"
    )
    parser.add_argument(
        "--phonemes-dir",
        type=Path,
        default=DEFAULT_PHONEMES_DIR,
        help="directory holding the per-shard inference results",
    )
    args = parser.parse_args()

    token = load_secrets()
    datasets_root = Path(datasets.config.HF_DATASETS_CACHE)

    if args.test:
        shard_index, filename = smallest_clean_shard(REPO_ID, token)
        test_dir = args.output_dir or datasets_root / TEST_DIR_NAME
        print(f"test shard: {filename} (index {shard_index})")
        total_rows = run_pipeline(
            token=token,
            output_dir=test_dir,
            num_proc=args.num_proc,
            shards=[(shard_index, filename)],
            phonemes_dir=args.phonemes_dir,
        )
        verify_loading(test_dir, expected_rows=total_rows)
        if args.cleanup_test:
            shutil.rmtree(test_dir, ignore_errors=True)
            print(f"removed test directory {test_dir}")
        return

    output_dir = args.output_dir or datasets_root / OUTPUT_DIR_NAME
    shards = list_clean_shards(REPO_ID, token)
    if not shards:
        raise RuntimeError(f"No 'clean' parquet shards found in {REPO_ID}")
    total_rows = run_pipeline(
        token=token,
        output_dir=output_dir,
        num_proc=args.num_proc,
        shards=shards,
        phonemes_dir=args.phonemes_dir,
    )
    verify_loading(output_dir, expected_rows=total_rows)


if __name__ == "__main__":
    main()
