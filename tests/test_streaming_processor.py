try:
    import pytest
except ImportError:  # running as a plain script
    pytest = None
else:
    # Skip rather than error at collection: this module needs the GPU stack, even
    # though test_chunk_from_frames_invariants below is pure arithmetic. Moving
    # StreamingProcessorMath beside prepare_quran_dataset.ctc_decoding would free it.
    pytest.importorskip("torch")
    pytest.importorskip("nemo")

import torch

from prepare_quran_dataset.modeling_fastconformer_cache_aware.processor import (
    FastConformerMelProcessor,
)


class StreamingProcessorMath:
    def __init__(
        self,
        hop=160,
        n_fft=512,
        pad=512,
        window=400,
        front_pad=120,
    ):
        """Configure the Streaming Processor parameters once; every method reuses them.

        Args:
            hop: STFT hop size in samples.
            n_fft: STFT FFT size.
            pad: zero-padding per chunk (must equal n_fft).
            window: Hann window size (center=True) in samples.
            front_pad: leading pad applied before chunking.
        """
        self.hop = hop
        self.n_fft = n_fft
        self.pad = pad
        self.window = window
        self.front_pad = front_pad
        self.check_front_pad()
        self.overlap = self._calc_overlap()
        self.drop_start = self._calc_drop_start()
        self.drop_end = self._calc_drop_end()

    def calc_samples_to_frames(self, L: int) -> int:
        """Number of frames given number of samples.

        Note: the moving window is n_fft not window, and window corresponds to
        center=True hann window of the n_fft window.
        """
        return (L + self.pad - self.n_fft) // self.hop + 1

    def check_front_pad(self):
        """Check front pad for center=True torch.stft."""
        assert self.pad == self.n_fft
        assert self.pad % 2 == 0
        assert (self.n_fft - self.window) % 2 == 0
        assert (
            self.pad // 2 + self.front_pad - (self.n_fft - self.window) // 2
        ) % self.hop == 0, "Invalid front_pad calculation"

    def check_chunk_size(self, chunk: int):
        """Check correct chunk and shift size.

        Args:
            chunk: chunk size in samples. Defaults to self.chunk.
        """
        assert self.pad == self.n_fft
        assert self.pad % 2 == 0
        assert self.window % 2 == 0
        assert (chunk - self.window // 2 + self.front_pad) % self.hop == 0, (
            "invalid chunk_size"
        )

    def _calc_overlap(self) -> int:
        """The overlap with the previous chunk"""
        return self.window - self.hop

    def _calc_drop_start(self) -> int:
        """Number of frames to drop from the beginning of each processed chunk"""
        return (self.front_pad + self.window // 2) // self.hop

    def _calc_drop_end(self) -> int:
        """Number of frames to drop from the end of each processed chunk"""
        return self.window // (self.hop * 2)

    def calc_chunk_from_frames(self, frames: int) -> int:
        """Chunk length in samples that nets exactly `frames` usable mel frames.

        Two invariants pin the formula (both asserted by the notebook):

        1. `calc_samples_to_frames(front_pad + chunk) - drop_start - drop_end == frames`
        2. `chunk - overlap == hop * frames`, so consecutive chunks advance by
           exactly `frames` frames and the stream neither gaps nor overlaps.

        Only `drop_end` enters the expression: `drop_start` is already paid for
        by `front_pad`, which exists to make `front_pad + window // 2` a whole
        number of hops. Including it again overshoots by `drop_start` frames.
        """
        return self.hop * (frames + self.drop_end) - self.front_pad + self.window // 2


def run_offline_processor(
    sp: StreamingProcessorMath,
    wave: torch.FloatTensor,
    processor: FastConformerMelProcessor,
):
    L = wave.shape[1] + sp.front_pad
    wave = torch.nn.functional.pad(wave, (sp.front_pad, 0, 0, 0))
    assert wave.shape[1] == L
    features, lens = processor(wave, torch.tensor([L] * wave.shape[0]))
    assert features.shape[2] == lens[0] == sp.calc_samples_to_frames(L), (
        "Can not predict len of offline processor"
    )
    wave = torch.nn.functional.pad(wave, (sp.pad // 2, sp.pad // 2, 0, 0))
    start = 0
    sample_windows_l = []
    for _ in range(sp.calc_samples_to_frames(L)):
        end = start + sp.n_fft
        sample_windows_l.append(wave[:, start:end])
        # print(wave[:, start:end].shape)
        start += sp.hop
    return features, torch.stack(sample_windows_l, dim=2)


def run_streaming_processor(
    sp: StreamingProcessorMath,
    wave: torch.FloatTensor,
    processor: FastConformerMelProcessor,
    chunk: int,
):
    sp.check_front_pad()
    sp.check_chunk_size(chunk)
    num_iter = (wave.shape[1] - chunk) // (chunk - sp.overlap) + 2
    feat_l = []
    windows_l = []
    start = 0
    end = 0
    for idx in range(num_iter):
        end = start + chunk
        input_wave = torch.nn.functional.pad(
            wave[:, start:end], (sp.front_pad, 0, 0, 0)
        )
        feats, _ = processor(
            input_wave, torch.tensor([chunk + sp.front_pad] * wave.shape[0])
        )

        debug_input_wave = torch.nn.functional.pad(
            input_wave, (sp.pad // 2, sp.pad // 2, 0, 0)
        )
        chunk_windows_l = []
        d_start = 0
        for _ in range(sp.calc_samples_to_frames(input_wave.shape[1])):
            d_end = d_start + sp.n_fft
            chunk_windows_l.append(debug_input_wave[:, d_start:d_end])
            d_start += sp.hop

        if idx == 0:
            # start (this is not meant for streaming but for comparison of the orignal offline processor
            feat_l.append(feats[:, :, : -sp.drop_end])
            chunk_windows_l = chunk_windows_l[: -sp.drop_end]
        elif idx == num_iter - 1:
            # end (not meant for streaming but to compare with the original offline processor)
            feat_l.append(feats[:, :, sp.drop_start :])
            chunk_windows_l = chunk_windows_l[sp.drop_start :]

        else:
            feat_l.append(feats[:, :, sp.drop_start : -sp.drop_end])
            chunk_windows_l = chunk_windows_l[sp.drop_start : -sp.drop_end]

        start = end - sp.overlap
        windows_l += chunk_windows_l
    return torch.cat(feat_l, dim=2), torch.stack(windows_l, dim=2)


def make_processor() -> FastConformerMelProcessor:
    """Build the processor without dither so results are deterministic.

    NeMo adds random noise (dither) per call while the module is in train mode,
    which makes offline vs streaming outputs differ by ~1e-4 everywhere.
    """
    # NOTE: we set dither to 0.0 for sake of comparing the two processors but leave as the default for actual run
    return FastConformerMelProcessor(dither=0.0)


def test_chunk_from_frames_invariants():
    """The two properties `calc_chunk_from_frames` must satisfy.

    Kept out of `__main__` so pytest actually runs them: the original asserts lived
    only in the script body, where a wrong formula stayed invisible because the
    offline/streaming comparison is self-consistent either way.

    Pure arithmetic — no processor, no model — so this runs without nemo.
    """
    sp = StreamingProcessorMath()
    assert (sp.overlap, sp.drop_start, sp.drop_end) == (240, 2, 1)
    for frames in (1, 4, 13, 52):
        chunk = sp.calc_chunk_from_frames(frames)
        sp.check_chunk_size(chunk)
        net = (
            sp.calc_samples_to_frames(sp.front_pad + chunk)
            - sp.drop_start
            - sp.drop_end
        )
        assert net == frames, f"{frames} frames -> chunk {chunk} nets {net}"
        assert chunk - sp.overlap == sp.hop * frames, (
            f"stride {chunk - sp.overlap} != hop * {frames}"
        )


if __name__ == "__main__":
    test_chunk_from_frames_invariants()
    processor = make_processor()
    wave = torch.randn(1, 320000)
    sp = StreamingProcessorMath()
    print(f"overlap: {sp.overlap}, drops: {sp.drop_start, sp.drop_end}")

    frames_per_chunk = 4
    chunk = sp.calc_chunk_from_frames(frames_per_chunk)
    print(f"chunk: {chunk} samples ({chunk / 16:.1f} ms) -> {frames_per_chunk} frames")

    offline_feats, offline_windows = run_offline_processor(sp, wave, processor)
    streaming_feats, streaming_windows = run_streaming_processor(
        sp,
        wave,
        processor,
        chunk=chunk,
    )
    print(f"offline sample windows: {offline_windows.shape}")
    print(f"streaming sample windows: {streaming_windows.shape}")

    # Compare over the frames both paths produced; the tail chunk can differ.
    margin = min(offline_feats.shape[2], streaming_feats.shape[2])
    off, stream = offline_feats[:, :, :margin], streaming_feats[:, :, :margin]
    print(
        f"Both sample windows match: "
        f"{torch.allclose(offline_windows[:, 56:456, :margin], streaming_windows[:, 56:456, :margin])}"
    )
    print(f"Offline Len: {offline_feats.shape}")
    print(f"Streaming Len: {streaming_feats.shape}")
    print(f"Num of chunk frames: {sp.calc_samples_to_frames(sp.front_pad + chunk)}")
    print(f"max |offline - streaming|: {(off - stream).abs().max().item():.3e}")
    print(f"Both processors match (atol=1e-6): {torch.allclose(off, stream, atol=1e-6)}")
    print(f"Both processors match (default tol): {torch.allclose(off, stream)}")
