"""Live-microphone cache-aware streaming for the FastConformer multi-level CTC model.

Unlike ``infer_fastconformer_streaming`` (which computes the whole file's mel
spectrogram up-front and slices it — a *simulation* of streaming), this module computes
mel frames incrementally from a growing raw-audio buffer while staying bit-identical to
the whole-file computation, and drives the model's cache-aware ``streaming_step`` chunk
by chunk as audio arrives from a browser microphone through a Gradio app.

Why incremental mel is tricky (and how it is handled here):

- The STFT is centered (``torch.stft(center=True, pad_mode="reflect")``) with
  ``n_fft=512`` / hop 160, so mel frame ``f`` covers samples ``[f*160 - 256, f*160 + 256)``.
  A frame may only be emitted mid-stream once its last sample exists — no reflection
  padding against a fake "end".
- Preemphasis special-cases the first sample of whatever signal it is given.
- Therefore each chunk's frames are computed over a window that starts ``MARGIN_FRAMES``
  frames early; the contaminated frames fall inside the discarded margin, and the
  interior frames match the whole-file mel exactly.  ``tests/test_streaming_processor.py``
  derives the contamination bound independently and puts it at 2 frames, so the margin
  of 8 is a 4x safety factor.
- Dither is forced to 0 (NeMo's own streaming does the same); otherwise the featurizer
  injects noise and results are non-deterministic.

Usage::

    python -m prepare_quran_dataset.modeling_fastconformer_cache_aware.streaming_mic app
    python -m prepare_quran_dataset.modeling_fastconformer_cache_aware.streaming_mic self-test
    python -m prepare_quran_dataset.modeling_fastconformer_cache_aware.streaming_mic probes
    python -m prepare_quran_dataset.modeling_fastconformer_cache_aware.streaming_mic post-mortem
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import time
import traceback
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import soxr
import torch
from huggingface_hub import hf_hub_download

from prepare_quran_dataset.ctc_decoding import (
    BLANK_ID,
    EOS_ID,
    collapse_carry,
    ctc_collapse,
)

from . import (
    FastConformerCacheAwareMultilevelCTC,
    FastConformerMelProcessor,
    infer_fastconformer_streaming,
)

SAMPLE_RATE = 16000
HOP = 160  # window_stride 10 ms
HALF_NFFT = 256  # n_fft // 2, the centered-STFT padding on each side
# Frames within 2 frames of a window's left edge are contaminated (reflect pad +
# preemphasis first-sample special case reach ceil(257/160) = 2 frames in).
MARGIN_FRAMES = 8

SPECIAL_IDS = (BLANK_ID, EOS_ID)
DEFAULT_MODEL_ID = "obadx/muaalem-fastconformer-base-v1"
TARGET_DBFS = -12.0  # low end of the band the training audio occupies
DEFAULT_OUT_DIR = Path("mic_session")


# ----------------------------------------------------------------------
# Incremental mel
# ----------------------------------------------------------------------


class StreamingMelFeaturizer:
    """Incremental log-mel extraction, bit-identical to whole-file mel.

    Frames ``[a, b)`` are computed by running the (dither-free) NeMo processor on a
    sample window that starts ``MARGIN_FRAMES`` frames before ``a`` and ends exactly at
    the last sample frame ``b-1`` needs, then slicing the margin off.  Mid-stream, a
    frame is "available" only once sample ``f*160 + 255`` exists; at flush time the true
    signal end supplies the same right reflection padding the whole-file computation
    sees.
    """

    def __init__(self, processor_kwargs: dict, device: str | torch.device):
        self.device = torch.device(device)
        self.processor = FastConformerMelProcessor(
            **{**processor_kwargs, "dither": 0.0, "pad_to": 0}
        )
        self.processor.to(self.device)
        self.buf = np.zeros(0, dtype=np.float32)

    def append(self, samples: np.ndarray) -> None:
        self.buf = np.concatenate(
            [self.buf, np.asarray(samples, dtype=np.float32).reshape(-1)]
        )

    def available_frames(self) -> int:
        """Frames emittable now without any right-edge padding."""
        if len(self.buf) < HALF_NFFT:
            return 0
        return (len(self.buf) - HALF_NFFT) // HOP + 1

    def total_frames_final(self) -> int:
        """Whole-file frame count once the stream has ended (len//160 + 1)."""
        return len(self.buf) // HOP + 1

    def get_frames(self, a: int, b: int, final: bool = False) -> torch.Tensor:
        """Return mel frames [a, b) as (1, n_mels, b - a)."""
        w0_frame = max(0, a - MARGIN_FRAMES)
        w0 = w0_frame * HOP
        end = len(self.buf) if final else min(len(self.buf), (b - 1) * HOP + HALF_NFFT)
        window = torch.from_numpy(self.buf[w0:end]).unsqueeze(0).to(self.device)
        length = torch.tensor([end - w0], device=self.device)
        mel, _ = self.processor(input_signal=window, length=length)
        return mel[:, :, a - w0_frame : b - w0_frame]

    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)


# ----------------------------------------------------------------------
# Cache-aware streaming driver
# ----------------------------------------------------------------------


class FastConformerMicStreamer:
    """Feeds arbitrary-sized 16 kHz sample chunks to the cache-aware model.

    Replicates the exact chunking of ``infer_fastconformer_streaming`` / NeMo's
    ``CacheAwareStreamingAudioBuffer``:

    - step 0 consumes mel frames ``[0, cs0)`` with ``drop_extra_pre_encoded=0``
    - step s>=1 consumes ``[cs0 + cs1*(s-1) - pc1, cs0 + cs1*s)`` (``pc1`` re-fed
      overlap frames) with the config's ``drop_extra_pre_encoded``
    - a step runs mid-stream only when *strictly more* frames are available than it
      consumes: a stream ending exactly on a chunk boundary must run that chunk with
      ``keep_all_outputs=True``, which only ``flush`` can know
    - at flush, the final partial chunk is zero-padded on the mel-frame axis to the
      expected count, with the true valid length; a leftover of fewer than
      ``sampling_frames[1]`` new frames is dropped entirely (NeMo parity)
    """

    def __init__(
        self,
        model: FastConformerCacheAwareMultilevelCTC,
        vocab: dict[str, dict[str, int]],
        device: str | torch.device,
    ):
        self.model = model.eval()
        self.device = torch.device(device)

        model.setup_streaming_params()
        cfg = model.encoder.streaming_cfg
        cs = cfg.chunk_size
        self.cs0, self.cs1 = (cs[0], cs[1]) if isinstance(cs, (list, tuple)) else (cs, cs)
        pc = cfg.pre_encode_cache_size
        self.pc1 = pc[1] if isinstance(pc, (list, tuple)) else pc
        self.drop = cfg.drop_extra_pre_encoded
        if hasattr(model.encoder, "pre_encode") and hasattr(
            model.encoder.pre_encode, "get_sampling_frames"
        ):
            self.sf1 = model.encoder.pre_encode.get_sampling_frames()[1]
        else:
            self.sf1 = 1

        self.vocab = vocab
        self.id_to_token = {
            level: {idx: tok for tok, idx in level_vocab.items()}
            for level, level_vocab in vocab.items()
        }
        self.featurizer = StreamingMelFeaturizer(
            model.config.processor_kwargs, self.device
        )
        self.reset()

    @property
    def phoneme_level(self) -> str:
        return "phonemes" if "phonemes" in self.level_ids else next(iter(self.level_ids))

    def reset(self) -> None:
        self.cache = self.model.get_initial_cache(batch_size=1)
        self.step = 0
        self.level_ids: dict[str, list[int]] = {
            level: [] for level in self.model.level_to_lm_head
        }
        self.chunk_log: list[dict] = []
        self.featurizer.reset()
        self.finished = False
        # CTC state for the per-chunk log, carried across chunk boundaries. Restarting
        # it each chunk re-emits a straddling phoneme, which here fabricates a shadda.
        self._log_prev = BLANK_ID

    def chunk_end(self, s: int) -> int:
        """Global mel-frame index one past step ``s``'s new frames."""
        return self.cs0 + self.cs1 * s

    @torch.no_grad()
    def _run_step(
        self, mel: torch.Tensor, length: int, keep_all: bool, drop: int
    ) -> None:
        t0 = time.perf_counter()
        out = self.model.streaming_step(
            processed_signal=mel,
            processed_length=torch.tensor([length], device=self.device),
            cache=self.cache,
            keep_all_outputs=keep_all,
            drop_extra_pre_encoded=drop,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.cache = out.cache

        new_ids: dict[str, list[int]] = {}
        for level, logits in out.logits.items():
            ids = logits[0].argmax(dim=-1).tolist()
            self.level_ids[level].extend(ids)
            new_ids[level] = ids

        ph_level = "phonemes" if "phonemes" in new_ids else next(iter(new_ids))
        tokens, self._log_prev = collapse_carry(new_ids[ph_level], self._log_prev)
        self.chunk_log.append(
            {
                "step": self.step,
                "frames_in": int(length),
                "frames_out": len(new_ids[ph_level]),
                "latency_ms": latency_ms,
                "phonemes": "".join(
                    self.id_to_token[ph_level].get(t, "?") for t in tokens
                ),
            }
        )
        self.step += 1

    def _run_regular_step(self, keep_all: bool, final: bool) -> None:
        s = self.step
        if s == 0:
            mel = self.featurizer.get_frames(0, self.cs0, final=final)
            self._run_step(mel, self.cs0, keep_all=keep_all, drop=0)
        else:
            a = self.chunk_end(s - 1) - self.pc1
            b = self.chunk_end(s)
            mel = self.featurizer.get_frames(a, b, final=final)
            self._run_step(mel, b - a, keep_all=keep_all, drop=self.drop)

    def feed(self, samples_16k: np.ndarray) -> None:
        """Append raw 16 kHz samples and run every step that is fully covered."""
        if self.finished:
            raise RuntimeError("Streamer is finished; call reset() first.")
        self.featurizer.append(samples_16k)
        while self.featurizer.available_frames() > self.chunk_end(self.step):
            self._run_regular_step(keep_all=False, final=False)

    def flush(self) -> None:
        """End of stream: run remaining chunks, final one with keep_all_outputs."""
        if self.finished:
            return
        self.finished = True
        # Too short for even one reflect-padded STFT window: nothing to emit.
        if len(self.featurizer.buf) <= HALF_NFFT:
            return

        T = self.featurizer.total_frames_final()
        while T > self.chunk_end(self.step):
            self._run_regular_step(keep_all=False, final=True)

        s = self.step
        if s == 0:
            mel = self.featurizer.get_frames(0, T, final=True)
            mel = torch.nn.functional.pad(mel, (0, self.cs0 - mel.size(-1)))
            self._run_step(mel, T, keep_all=True, drop=0)
        else:
            new = T - self.chunk_end(s - 1)
            if new >= self.sf1:
                a = self.chunk_end(s - 1) - self.pc1
                mel = self.featurizer.get_frames(a, T, final=True)
                expected = self.cs1 + self.pc1
                mel = torch.nn.functional.pad(mel, (0, expected - mel.size(-1)))
                self._run_step(mel, new + self.pc1, keep_all=True, drop=self.drop)
            # else 1 <= new < sf1: NeMo's iterator silently drops this tail.

    # ------------------------------------------------------------------
    # Decoding / rendering
    # ------------------------------------------------------------------

    def decode_ids(self, level: str) -> list[int]:
        return ctc_collapse(self.level_ids[level])

    def decode_tokens(self, level: str) -> list[str]:
        return [self.id_to_token[level].get(t, "?") for t in self.decode_ids(level)]

    def transcript(self) -> str:
        return "".join(self.decode_tokens(self.phoneme_level))

    def sifat_summary(self, last_n: int = 5) -> str:
        """Last few decoded tokens of every non-phoneme level, one line each."""
        lines = []
        for level in self.level_ids:
            if level == "phonemes":
                continue
            tokens = self.decode_tokens(level)[-last_n:]
            lines.append(f"{level}: {' '.join(tokens) if tokens else '-'}")
        return "\n".join(lines)

    def seconds_fed(self) -> float:
        return len(self.featurizer.buf) / SAMPLE_RATE


def load_model_and_vocab(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | torch.device = "cpu",
    token: str | None = None,
) -> tuple[FastConformerCacheAwareMultilevelCTC, dict[str, dict[str, int]]]:
    model = FastConformerCacheAwareMultilevelCTC.from_pretrained(model_id, token=token)
    model.to(device)
    model.eval()
    model.processor.to(device)  # processor is not an nn.Module; model.to() skips it
    model.setup_streaming_params()
    vocab_path = hf_hub_download(model_id, "vocab.json", token=token)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    return model, vocab


# ----------------------------------------------------------------------
# Shared measurement helpers
# ----------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = _REPO_ROOT / "assets" / "audio-sampels"
DEFAULT_REF_AUDIO = ASSET_DIR / "test_sample.mp3"
DEFAULT_GAP_AUDIO = ASSET_DIR / "test.wav"  # longest asset, and a WAV


def dbfs(x: np.ndarray) -> tuple[float, float]:
    """(rms, peak) in dBFS, floored so silence does not blow up the log."""
    x = np.asarray(x, dtype=np.float32)
    to_db = lambda v: 20.0 * np.log10(v) if v > 1e-12 else -np.inf
    return (
        to_db(float(np.sqrt(np.mean(x**2))) if x.size else 0.0),
        to_db(float(np.max(np.abs(x))) if x.size else 0.0),
    )


def nonblank(ids: list[int]) -> float:
    return sum(1 for i in ids if i != BLANK_ID) / max(len(ids), 1)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def probe_duration(path: str | Path) -> float | None:
    """Independent decode reference.

    soundfile and librosa both sit on libsndfile, so comparing those two cannot catch a
    short mp3 decode: for ``test_sample.mp3`` they agree on 1.64 s while ffmpeg decodes
    3.92 s from the same bytes.  ffprobe is the outside opinion.
    """
    try:
        return float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.strip()
        )
    except Exception:
        return None


def make_gapped(
    wav: np.ndarray, period_s: float, drop_ms: float
) -> tuple[np.ndarray, float]:
    """Excise ``drop_ms`` every ``period_s`` and concatenate.

    That is what a dropped browser chunk looks like to the model.  Asserts it actually
    removed something: an earlier version of this test used a period longer than the
    file and silently measured nothing while reporting success.
    """
    keep = np.ones(len(wav), dtype=bool)
    width = int(drop_ms / 1000 * SAMPLE_RATE)
    step = int(period_s * SAMPLE_RATE)
    for start in range(step, len(wav), step):
        keep[start : start + width] = False
    removed = 1.0 - float(keep.mean())
    assert removed > 0.0, (
        f"gap test removed nothing: {len(wav) / SAMPLE_RATE:.1f}s of audio with a "
        f"{period_s}s period — the buffer is shorter than one period"
    )
    return wav[keep], removed


class StreamingDiagnostics:
    """Probes that compare the streaming path against offline on the same audio."""

    def __init__(
        self,
        model: FastConformerCacheAwareMultilevelCTC,
        vocab: dict[str, dict[str, int]],
        device: str | torch.device,
    ):
        self.model = model.eval()
        self.vocab = vocab
        self.device = torch.device(device)
        self.ph = "phonemes" if "phonemes" in vocab else next(iter(vocab))
        self.id_to_token = {i: t for t, i in vocab[self.ph].items()}
        self.processor = FastConformerMelProcessor(
            **{**model.config.processor_kwargs, "dither": 0.0, "pad_to": 0}
        ).to(self.device)

    def tokens(self, ids: list[int]) -> str:
        return "".join(self.id_to_token.get(t, "?") for t in ctc_collapse(ids))

    @torch.no_grad()
    def offline_probe(self, wav: np.ndarray) -> tuple[str, float, float]:
        """Whole-file forward. Returns (transcript, non-blank ratio, mean log-mel)."""
        t = torch.tensor([wav], dtype=torch.float32, device=self.device)
        n = torch.tensor([len(wav)], dtype=torch.long, device=self.device)
        ids = self.model(raw_audio=t, audio_length=n).logits[self.ph][0]
        ids = ids.argmax(dim=-1).tolist()
        mel, _ = self.processor(input_signal=t, length=n)
        return self.tokens(ids), nonblank(ids), float(mel.mean())

    def replay(
        self, wav: np.ndarray, chunk_s: float = 0.5
    ) -> FastConformerMicStreamer:
        """Drive a fresh streamer over ``wav`` in fixed-size chunks.

        A new streamer every call: the encoder cache must not leak between probes.
        """
        st = FastConformerMicStreamer(self.model, self.vocab, self.device)
        step = int(chunk_s * SAMPLE_RATE)
        for pos in range(0, len(wav), step):
            st.feed(wav[pos : pos + step])
        st.flush()
        return st

    def stream_probe(
        self, wav: np.ndarray, chunk_s: float = 0.5
    ) -> tuple[str, float, list[int]]:
        st = self.replay(wav, chunk_s)
        ids = st.level_ids[self.ph]
        return self.tokens(ids), nonblank(ids), ids

    # ------------------------------------------------------------------

    def run_probes(
        self,
        ref_path: str | Path = DEFAULT_REF_AUDIO,
        gap_path: str | Path = DEFAULT_GAP_AUDIO,
    ) -> None:
        """Regression harness.

        Input level and resampling are settled — flat across a 48 dB sweep and 0 sample
        drift respectively — and are kept here as cheap guards.  The open question is
        what dropped audio costs, which the gap sweep answers.
        """
        print(f"{'asset':<20} {'librosa':>9} {'ffprobe':>9} {'rms dBFS':>9}")
        for path in (ref_path, gap_path):
            y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
            loaded, real = len(y) / SAMPLE_RATE, probe_duration(path)
            shown = "      n/a" if real is None else f"{real:8.2f}s"
            flag = ""
            if real is not None and abs(loaded - real) > 0.2:
                flag = f"  <-- librosa reads {loaded / real:.0%} of it"
            print(f"{Path(path).name:<20} {loaded:>8.2f}s {shown} {dbfs(y)[0]:>+8.1f}{flag}")
        print("self_test loads its audio through librosa.load, so it gates on whatever "
              "that returns.")

        ref_wav, _ = librosa.load(str(ref_path), sr=SAMPLE_RATE, mono=True)
        gap_wav, _ = librosa.load(str(gap_path), sr=SAMPLE_RATE, mono=True)

        print("\n--- gain spot check (settled: expect flat) ---")
        _, _, base_mel = self.offline_probe(ref_wav)
        for db in (12, 0, -24):
            text, ratio, mel_mean = self.offline_probe(ref_wav * (10.0 ** (db / 20.0)))
            print(f"{db:>+4}dB | non-blank {ratio:.3f} | log-mel {mel_mean:+6.2f} "
                  f"(expect {base_mel + 0.2303 * db:+.2f}) | {text[:40]}")

        print(f"\n--- dropped-chunk sweep on {Path(gap_path).name} "
              f"({len(gap_wav) / SAMPLE_RATE:.1f}s) ---")
        intact_text, intact_ratio, _ = self.stream_probe(gap_wav)
        print(f"{'removed':>8} {'non-blank':>10} {'vs intact':>10}  transcript")
        print(f"{0.0:>7.1%} {intact_ratio:>10.3f} {1.0:>10.1%}  {intact_text[:44]}")
        for period_s, drop_ms in ((2.0, 100), (1.0, 100), (0.5, 100), (1.0, 500)):
            gapped, removed = make_gapped(gap_wav, period_s, drop_ms)
            text, ratio, _ = self.stream_probe(gapped)
            print(f"{removed:>7.1%} {ratio:>10.3f} "
                  f"{similarity(intact_text, text):>10.1%}  {text[:44]}")
        print("Read this as the accuracy cost of a live session's wall-clock deficit.")

        print("\n--- 48 kHz round-trip through soxr.ResampleStream (settled) ---")
        up = soxr.resample(ref_wav, SAMPLE_RATE, 48000)
        rs = soxr.ResampleStream(48000, SAMPLE_RATE, 1, dtype="float32")
        step = int(0.5 * 48000)
        parts = [rs.resample_chunk(up[p : p + step]) for p in range(0, len(up), step)]
        parts.append(rs.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
        round_trip = np.concatenate([p for p in parts if len(p)]).astype(np.float32)
        rt_text, rt_ratio, _ = self.stream_probe(round_trip)
        ref_text, ref_ratio, _ = self.stream_probe(ref_wav)
        print(f"length drift {len(round_trip) - len(ref_wav):+d} samples | "
              f"non-blank {rt_ratio:.3f} vs {ref_ratio:.3f} | "
              f"transcript match {similarity(ref_text, rt_text):.1%}")

    # ------------------------------------------------------------------

    def post_mortem(self, out_dir: str | Path = DEFAULT_OUT_DIR) -> bool:
        """Analyse a saved session: where the audio went, and whether live == replay."""
        out_dir = Path(out_dir)
        session = json.loads((out_dir / "session.json").read_text(encoding="utf-8"))
        wav, _ = librosa.load(str(out_dir / "capture.wav"), sr=SAMPLE_RATE, mono=True)

        errors = session["errors"]
        print(f"=== errors: {len(errors)} ===")
        if errors:
            print(errors[-1])

        wall = float(session["wall_seconds"])
        captured = len(wav) / SAMPLE_RATE
        deficit = wall - captured
        rms, peak = dbfs(wav)
        print(f"\ncaptured : {captured:.1f}s over {wall:.1f}s wall clock "
              f"(deficit {deficit:+.1f}s)")
        print(f"level    : rms {rms:+.1f} dBFS, peak {peak:+.1f} dBFS")
        print(f"browser  : sr {session['native_sr']}, dtypes {session['dtypes']}")

        self._attribute_deficit(session, wall, captured, deficit)

        live_samples = session.get("n_samples")
        if live_samples is not None and live_samples != len(wav):
            drift = (len(wav) - live_samples) / SAMPLE_RATE
            print(f"\ncapture round-trip: wrote {live_samples} samples, read back "
                  f"{len(wav)} ({drift:+.3f}s) — the replay is not seeing the same audio")

        live_ids = list(session["level_ids"][self.ph])
        replay_streamer = self.replay(wav)
        replay_ids = replay_streamer.level_ids[self.ph]
        replay_text = self.tokens(replay_ids)
        replay_ratio = nonblank(replay_ids)
        live_text = self.tokens(live_ids)
        live_steps, replay_steps = session.get("steps"), replay_streamer.step
        if live_steps is not None and live_steps != replay_steps:
            print(f"\nstep count: live ran {live_steps}, replay ran {replay_steps} "
                  f"({live_steps - replay_steps:+d}) over "
                  f"{replay_streamer.featurizer.total_frames_final()} mel frames")

        n = min(len(live_ids), len(replay_ids))
        first_diff = next((i for i in range(n) if live_ids[i] != replay_ids[i]), None)
        exact = len(live_ids) == len(replay_ids) and first_diff is None
        agreement = sum(1 for a, b in zip(live_ids, replay_ids) if a == b) / max(n, 1)

        print(f"\nlive   : {len(live_ids):>5} frames | non-blank {nonblank(live_ids):.3f}"
              f"\n  {live_text[:90]}")
        print(f"replay : {len(replay_ids):>5} frames | non-blank {replay_ratio:.3f}"
              f"\n  {replay_text[:90]}")
        print(f"frame agreement {agreement:.4f}"
              + ("" if first_diff is None else f" | first difference at frame {first_diff}"))

        joined = "".join(e["phonemes"] for e in session["chunk_log"])
        log_ok = joined == live_text
        print(f"\nper-chunk log concatenates to transcript: {log_ok}")
        if not log_ok:
            print(f"  log        : {joined[:90]}")
            print(f"  transcript : {live_text[:90]}")

        offline_text, offline_ratio, _ = self.offline_probe(wav)
        print(f"\noffline on capture: non-blank {offline_ratio:.3f}\n  {offline_text[:90]}")

        print("\n================ VERDICT ================")
        if exact:
            print("LIVE MATCHES REPLAY — the streaming path is verified end to end on "
                  f"real microphone audio ({len(live_ids)} frames, exact).")
        elif agreement > 0.99:
            extra = len(live_ids) - len(replay_ids)
            print(f"NEAR MATCH — {agreement:.2%} of frames agree over the common prefix; "
                  f"lengths {len(live_ids)} vs {len(replay_ids)} ({extra:+d}).")
            print("  The shorter run is a strict prefix, so this is a tail-step "
                  "difference, not a decode error. Check the two lines above: a capture "
                  "round-trip drift or a step-count difference names the cause.")
        else:
            print(f"LIVE DIFFERS FROM REPLAY — only {agreement:.2%} of frames agree. "
                  "The same buffer replays correctly, so this is a live-only bug.")
        return exact and log_ok

    @staticmethod
    def _attribute_deficit(
        session: dict, wall: float, captured: float, deficit: float
    ) -> None:
        """Split the wall-clock deficit into warm-up, mid-stream drops and stop latency.

        Each arrival ``(t, n)`` means n samples landed at t, so the chunk covers roughly
        ``[t - n/sr, t]``.  Audio can therefore be lost before the first span, between
        spans, or after the last one — and separately, audio that *did* arrive can fail
        to reach the model at all, which is a bug rather than a latency cost and is
        reported on its own line.
        """
        arrivals = session.get("arrivals") or []
        print("\n--- deficit attribution ---")
        if not arrivals:
            print("  no arrivals recorded")
            return
        ts = np.array([t for t, _ in arrivals])
        ns = np.array([n for _, n in arrivals]) / SAMPLE_RATE
        warmup = max(0.0, ts[0] - ns[0])
        tail = max(0.0, wall - ts[-1])
        drops = max(0.0, (ts[-1] - (ts[0] - ns[0])) - ns.sum())
        stranded = float(ns.sum()) - captured
        print(f"  chunks received   : {len(arrivals)} ({ns.sum():.1f}s of audio)")
        print(f"  warm-up           : {warmup:5.1f}s  (start_recording -> first audio)")
        print(f"  drops mid-stream  : {drops:5.1f}s  (net gaps between chunks)")
        print(f"  stop latency      : {tail:5.1f}s  (last chunk -> stop handler)")
        if abs(stranded) > 0.05:
            print(f"  stranded          : {stranded:5.1f}s  (ARRIVED BUT NEVER FED — a bug)")
        print(f"  accounted         : {warmup + drops + tail + stranded:5.1f}s "
              f"of {deficit:.1f}s")
        if len(ts) > 1:
            # Individual gaps measure arrival jitter, not loss: a late chunk is followed
            # by one carrying the backlog, so only the net above is lost audio.
            inter = ts[1:] - ts[:-1] - ns[1:]
            worst = int(np.argmax(inter))
            print(f"  largest jitter    : {inter[worst]:5.1f}s at t={ts[worst + 1]:.1f}s "
                  "(jitter, not necessarily loss)")
        biggest = max((warmup, "warm-up"), (drops, "mid-stream drops"), (tail, "stop latency"))
        print(f"  -> dominated by {biggest[1]} ({biggest[0]:.1f}s)")
        if biggest[1] != "mid-stream drops":
            print("     Decoupling ingestion from inference does NOT recover these "
                  "seconds; the remedy is the operator's timing.")


# ----------------------------------------------------------------------
# Correctness gate: true streaming vs the trusted file-based simulation
# ----------------------------------------------------------------------


def self_test(
    audio_path: str | Path,
    model: FastConformerCacheAwareMultilevelCTC,
    device: str | torch.device,
    vocab: dict[str, dict[str, int]] | None = None,
    seed: int = 0,
) -> bool:
    """Compare the mic-streaming path against ``infer_fastconformer_streaming``.

    For base-v1 the streaming output is exactly identical to offline, so every per-level
    frame-argmax id must match exactly.  Argmax ids (not logits) are compared because
    GPU kernel selection can shift logits by ~1e-6.
    """
    device = torch.device(device)
    model = model.eval()
    rng = np.random.default_rng(seed)
    wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    print(f"audio: {audio_path} ({len(wav) / SAMPLE_RATE:.2f}s)")

    # --- 1) mel exactness: incremental windowed frames vs whole-file mel ---
    proc_kwargs = model.config.processor_kwargs
    feat = StreamingMelFeaturizer(proc_kwargs, device)
    feat.append(wav)
    whole, _ = feat.processor(
        input_signal=torch.from_numpy(wav).unsqueeze(0).to(device),
        length=torch.tensor([len(wav)], device=device),
    )
    n_avail = feat.available_frames()
    max_diff, a = 0.0, 0
    while a < n_avail:
        b = min(n_avail, a + int(rng.integers(1, 200)))
        got = feat.get_frames(a, b)
        max_diff = max(max_diff, (got - whole[:, :, a:b]).abs().max().item())
        a = b
    T = feat.total_frames_final()
    if T > n_avail:  # tail frames that need the right reflection padding
        got = feat.get_frames(n_avail, T, final=True)
        max_diff = max(max_diff, (got - whole[:, :, n_avail:T]).abs().max().item())
    mel_ok = max_diff < 1e-5
    print(f"mel exactness: max |diff| = {max_diff:.3e} -> {'OK' if mel_ok else 'FAIL'}")

    # --- 2) reference: file-based streaming simulation (dither = 0) ---
    ref_processor = FastConformerMelProcessor(
        **{**proc_kwargs, "dither": 0.0, "pad_to": 0}
    ).to(device)
    ref_logits = infer_fastconformer_streaming(
        [str(audio_path)], device, torch.float32, model, ref_processor
    )
    ref_ids = {lvl: t[0].argmax(dim=-1).tolist() for lvl, t in ref_logits.items()}

    # --- 3) true streaming: random-sized chunks + flush ---
    if vocab is None:
        vocab = {
            lvl: {str(i): i for i in range(head.out_features)}
            for lvl, head in model.level_to_lm_head.items()
        }
    streamer = FastConformerMicStreamer(model, vocab, device)
    pos = 0
    while pos < len(wav):
        n = int(rng.integers(160, 8000))
        streamer.feed(wav[pos : pos + n])
        pos += n
    streamer.flush()

    # --- 4) compare per level ---
    all_ok = mel_ok
    for level, ref in ref_ids.items():
        got = streamer.level_ids[level]
        if len(got) != len(ref):
            print(f"{level}: FAIL length mismatch (stream {len(got)} vs ref {len(ref)})")
            all_ok = False
            continue
        n_match = sum(g == r for g, r in zip(got, ref))
        ok = n_match == len(ref)
        all_ok = all_ok and ok
        print(f"{level}: {n_match}/{len(ref)} frames match "
              f"({n_match / max(len(ref), 1):.4f}) -> {'OK' if ok else 'FAIL'}")

    ph = streamer.phoneme_level
    id_to_tok = streamer.id_to_token[ph]
    ref_text = "".join(id_to_tok.get(t, "?") for t in ctc_collapse(ref_ids[ph]))
    print(f"reference  phonemes: {ref_text}")
    print(f"streaming  phonemes: {streamer.transcript()}")
    print("SELF TEST PASSED" if all_ok else "SELF TEST FAILED")
    return all_ok


# ----------------------------------------------------------------------
# Gradio microphone app
# ----------------------------------------------------------------------

# Browser WebRTC defaults (AGC / noise suppression / echo cancellation) are tuned for
# telephony and mangle recitation.  Gradio exposes no getUserMedia constraints, so the
# page patches the API itself.  Requires a hard reload of the link to take effect.
MIC_CONSTRAINTS_JS = """
<script>
(function () {
  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia || md.__muaalemPatched) return;
  const orig = md.getUserMedia.bind(md);
  md.getUserMedia = function (c) {
    if (c && c.audio) {
      const a = (typeof c.audio === "object") ? c.audio : {};
      c = Object.assign({}, c, {audio: Object.assign({}, a, {
        echoCancellation: false, noiseSuppression: false, autoGainControl: false})});
    }
    return orig(c);
  };
  md.__muaalemPatched = true;
})();
</script>
"""


def build_demo(
    model: FastConformerCacheAwareMultilevelCTC,
    vocab: dict[str, dict[str, int]],
    device: str | torch.device,
    out_dir: str | Path = DEFAULT_OUT_DIR,
):
    """Live microphone app.

    Ingestion and inference run on separate handlers.  Gradio drops stream events that
    arrive while their handler is still running, so the ``.stream()`` callback does
    nothing but resample and enqueue; a timer tick runs the model and redraws.  Keeping
    inference out of that path took mid-stream audio loss from ~9 s to 0.3 s over a
    30 s session.
    """
    import gradio as gr  # imported lazily: the `ui` extra is optional

    out_dir = Path(out_dir)
    ph = "phonemes" if "phonemes" in vocab else next(iter(vocab))
    state: dict = {}

    def _fresh(status: str) -> None:
        state.clear()
        state.update(
            streamer=FastConformerMicStreamer(model, vocab, device),
            resampler=None, native_sr=None, resampler_closed=False, delivered_sr=None,
            pending=[], arrivals=[], log=[], errors=[], dtypes=set(), stopping=False,
            t0=time.perf_counter(), status=status, capture=None,
        )

    _fresh("idle")

    def _to_float_mono(y: np.ndarray) -> np.ndarray:
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        else:
            y = np.asarray(y, dtype=np.float32)
        return y.mean(axis=1) if y.ndim > 1 else y

    def _resample(sr: int, y: np.ndarray, last: bool = False) -> np.ndarray:
        """Stateful 16 kHz resample.

        Refuses a finalised stream: soxr raises 'Input after last input', and because
        Gradio delivers queued stream events *after* stop_recording, that is what used
        to kill the app mid-session.
        """
        if state["resampler_closed"]:
            return np.zeros(0, dtype=np.float32)
        if sr == SAMPLE_RATE and not last:
            return y
        if state["resampler"] is None or state["native_sr"] != sr:
            state["resampler"] = soxr.ResampleStream(sr, SAMPLE_RATE, 1, dtype="float32")
            state["native_sr"] = sr
        out = state["resampler"].resample_chunk(y, last=last)
        if last:
            state["resampler_closed"] = True
        return out

    def _drain() -> None:
        """Run inference on everything that arrived since the last tick."""
        st = state["streamer"]
        if st.finished or not state["pending"]:
            return
        pending, state["pending"] = state["pending"], []
        samples = np.concatenate(pending)
        rms, peak = dbfs(samples)
        before = len(st.chunk_log)
        st.feed(samples)
        for entry in st.chunk_log[before:]:
            state["log"].append({**entry, "rms": rms, "peak": peak})

    def _render(capture: str | None = None):
        capture = capture or state.get("capture")
        st = state["streamer"]
        ids = st.level_ids[ph]
        blank = sum(1 for i in ids if i == BLANK_ID) / max(len(ids), 1)
        fed, wall = st.seconds_fed(), time.perf_counter() - state["t0"]
        lat = [e["latency_ms"] for e in st.chunk_log]
        arrivals = state["arrivals"]
        warmup = (arrivals[0][0] - arrivals[0][1] / SAMPLE_RATE) if arrivals else None

        summary = (
            f"{state['status']} | steps {st.step} | fed {fed:.1f}s / wall {wall:.1f}s "
            f"(deficit {wall - fed:+.1f}s"
            + (f", warm-up {warmup:.1f}s" if warmup is not None else "")
            + f") | blank {blank:.1%} | avg {np.mean(lat) if lat else 0:.0f} ms | "
            f"sr {state['delivered_sr']} | "
            f"dtypes {sorted(str(d) for d in state['dtypes']) or '-'}"
            + (" | FLUSHED" if st.finished else "")
        )

        last = state["log"][-1] if state["log"] else None
        level = (
            "waiting for audio — do not start reciting until this shows a level"
            if last is None
            else f"rms {last['rms']:+.1f} dBFS | peak {last['peak']:+.1f} dBFS | "
            f"suggested gain {TARGET_DBFS - last['rms']:+.0f} dB"
        )
        log = "\n".join(
            f"step {e['step']:03d} | in {e['frames_in']:3d} mel | out {e['frames_out']:3d} "
            f"| {e['latency_ms']:6.1f} ms | {e['rms']:+6.1f} dBFS | {e['phonemes']}"
            for e in state["log"][-30:]
        )
        return (st.transcript(), st.sifat_summary(), log, level, summary,
                "\n\n".join(state["errors"][-3:]), capture)

    def _guard(fn):
        """Render exceptions into the UI: Gradio buries handler tracebacks in the
        server log, which is invisible behind a share link."""
        def wrapper(*args):
            try:
                return fn(*args)
            except Exception:
                state.setdefault("errors", []).append(traceback.format_exc())
                state["status"] = "ERROR"
                return _render()
        return wrapper

    @_guard
    def on_start():
        _fresh("recording")
        return _render()

    def on_stream(audio_chunk, gain_db):
        """Ingestion only — no model call, no render.  See build_demo's docstring."""
        try:
            st = state["streamer"]
            # `stopping` closes the window between on_stop's drain and flush; without
            # it a chunk landing in that gap is never fed and vanishes from the capture.
            # `finished` is checked before the resampler is touched, because queued
            # events keep arriving after stop_recording.
            if state.get("stopping") or st.finished or audio_chunk is None:
                return
            sr, y = audio_chunk
            state["delivered_sr"] = sr
            state["dtypes"].add(y.dtype)
            y16k = _resample(sr, _to_float_mono(y))
            if not len(y16k):
                return
            y16k = y16k * (10.0 ** (gain_db / 20.0))
            state["pending"].append(y16k)
            # Every arrival, not just those completing a mel chunk, or the timeline
            # cannot account for the audio we are trying to find.
            state["arrivals"].append((time.perf_counter() - state["t0"], len(y16k)))
        except Exception:
            state.setdefault("errors", []).append(traceback.format_exc())

    @_guard
    def on_tick():
        _drain()
        return _render()

    @_guard
    def on_stop():
        st = state["streamer"]
        if st.finished:
            return _render()
        state["stopping"] = True  # no further appends; everything pending is now ours
        if state["resampler"] is not None and not state["resampler_closed"]:
            tail = _resample(state["native_sr"], np.zeros(0, dtype=np.float32), last=True)
            if len(tail):
                state["pending"].append(tail)
                state["arrivals"].append((time.perf_counter() - state["t0"], len(tail)))
        _drain()
        st.flush()
        state["status"] = "stopped"

        wav = np.array(st.featurizer.buf, copy=True)
        if len(wav):
            out_dir.mkdir(parents=True, exist_ok=True)
            sf.write(out_dir / "capture.wav", wav, SAMPLE_RATE, subtype="FLOAT")
            (out_dir / "session.json").write_text(
                json.dumps(
                    {
                        "chunk_log": state["log"],
                        "n_samples": int(len(wav)),
                        "steps": st.step,
                        "arrivals": state["arrivals"],
                        "wall_seconds": time.perf_counter() - state["t0"],
                        "native_sr": state["delivered_sr"],
                        "dtypes": sorted(str(d) for d in state["dtypes"]),
                        "errors": state["errors"],
                        "level_ids": {k: list(v) for k, v in st.level_ids.items()},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            state["capture"] = str(out_dir / "capture.wav")
            print(f"session written to {out_dir}/")
        return _render()

    @_guard
    def on_reset():
        _fresh("reset")
        return _render()

    with gr.Blocks(title="Muaalem FastConformer live streaming",
                   head=MIC_CONSTRAINTS_JS) as demo:
        gr.Markdown(
            "## Muaalem FastConformer — live cache-aware streaming\n"
            "Wait for **Input level** to show a reading before reciting: the browser "
            "takes a few seconds to start delivering audio and anything said before "
            "then is never captured. Press **stop** when done — that flushes the final "
            "chunk and writes the session to disk."
        )
        audio_input = gr.Audio(sources=["microphone"], streaming=True, type="numpy",
                               label="Microphone")
        gain = gr.Slider(-6, 36, value=0, step=1, label="Input gain (dB)")
        level_box = gr.Textbox(label="Input level", lines=1)
        summary_box = gr.Textbox(label="Status", lines=2)
        transcript_box = gr.Textbox(label="Phonetic transcript", lines=4, rtl=True)
        sifat_box = gr.Textbox(label="Sifat (latest)", lines=8, rtl=True)
        log_box = gr.Textbox(label="Per-chunk log (last 30)", lines=14)
        error_box = gr.Textbox(label="Errors", lines=8)
        capture_box = gr.Audio(label="Captured 16 kHz buffer", type="filepath",
                               interactive=False)
        reset_btn = gr.Button("Reset")

        outputs = [transcript_box, sifat_box, log_box, level_box, summary_box,
                   error_box, capture_box]
        audio_input.start_recording(fn=on_start, outputs=outputs)
        # outputs=[] keeps the ingestion path free of any UI work.
        audio_input.stream(fn=on_stream, inputs=[audio_input, gain], outputs=[],
                           stream_every=0.5)
        audio_input.stop_recording(fn=on_stop, outputs=outputs)
        reset_btn.click(fn=on_reset, outputs=outputs)
        gr.Timer(0.5).tick(fn=on_tick, outputs=outputs)

    return demo


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", nargs="?", default="app",
                        choices=["app", "self-test", "probes", "post-mortem", "all"])
    parser.add_argument("audio", nargs="?", default=None,
                        help="audio file for self-test (default: the bundled sample)")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-share", action="store_true")
    args = parser.parse_args(argv)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print(f"loading {args.model} on {device} ...")
    model, vocab = load_model_and_vocab(args.model, device, token=args.token)

    audio = args.audio or DEFAULT_REF_AUDIO
    if args.command == "self-test":
        return 0 if self_test(audio, model, device, vocab=vocab) else 1
    if args.command == "probes":
        StreamingDiagnostics(model, vocab, device).run_probes()
        return 0
    if args.command == "post-mortem":
        return 0 if StreamingDiagnostics(model, vocab, device).post_mortem(args.out_dir) else 1
    if args.command == "all":
        ok = self_test(audio, model, device, vocab=vocab)
        print()
        StreamingDiagnostics(model, vocab, device).run_probes()
        return 0 if ok else 1

    demo = build_demo(model, vocab, device, out_dir=args.out_dir)
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=not args.no_share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
