"""Standalone mel-spectrogram processor wrapping NeMo's AudioToMelSpectrogramPreprocessor.

This module provides :class:`FastConformerMelProcessor`, a lightweight wrapper around
NVIDIA NeMo's ``AudioToMelSpectrogramPreprocessor`` that converts raw PCM audio
waveforms to log-mel spectrograms.

The processor is designed to be used either standalone (for data collation or
standalone inference scripts) or as an internal component of the
:class:`~modeling_fastconformer_cache_aware_ctc.FastConformerCacheAwareMultilevelCTC`
model.  It supports serialisation via :meth:`save_pretrained` and
:meth:`from_pretrained`, enabling round-trip Hub compatibility.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor


class FastConformerMelProcessor:
    """Wrapper around NeMo's ``AudioToMelSpectrogramPreprocessor``.

    Converts batches of raw PCM audio waveforms to log-mel spectrograms using
    NeMo's feature extraction pipeline (STFT → mel filterbank → log compression).
    The processor handles variable-length sequences and supports batch processing.

    .. note::
       ``AudioToMelSpectrogramPreprocessor`` has no trainable parameters, but it
       may contain fixed buffers (e.g. window functions).  Use :meth:`to` to
       move these buffers to the appropriate device.

    Args:
        sample_rate:
            Audio sampling rate in Hz.  Defaults to ``16000``.
        normalize:
            Normalisation type.  ``"NA"`` (no normalisation, recommended for
            streaming models), ``"per_feature"``, or ``"all_feature"``.
            Defaults to ``"NA"``.
        window_size:
            STFT window size in seconds.  Defaults to ``0.025`` (25 ms).
        window_stride:
            STFT hop length in seconds.  Defaults to ``0.01`` (10 ms).
            This determines the raw mel-spectrogram frame rate: one frame
            every 10 ms.
        window:
            FFT window function type.  Defaults to ``"hann"``.
        features:
            Number of mel filterbank channels.  Defaults to ``80``.
        n_fft:
            Number of FFT bins.  Defaults to ``512``.
        frame_splicing:
            Number of consecutive mel frames to concatenate.  Defaults to ``1``
            (no splicing).
        dither:
            Magnitude of dither noise added before the FFT for numerical
            stability.  Defaults to ``1e-5``.
        pad_to:
            If non-zero, pad input signals to a multiple of this many samples
            before feature extraction.  ``0`` disables padding.  Defaults to
            ``0``.

    Example:
        >>> processor = FastConformerMelProcessor(features=80)
        >>> audio = torch.randn(2, 16000)  # (batch, samples)
        >>> lengths = torch.tensor([16000, 12000])
        >>> mel, mel_len = processor(audio, lengths)
        >>> mel.shape
        torch.Size([2, 80, 100])  # (batch, freq, time)
        >>> mel_len
        tensor([100, 75])
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        normalize: str = "NA",
        window_size: float = 0.025,
        window_stride: float = 0.01,
        window: str = "hann",
        features: int = 80,
        n_fft: int = 512,
        frame_splicing: int = 1,
        dither: float = 0.00001,
        pad_to: int = 0,
    ) -> None:
        """Initialise the mel-spectrogram processor.

        Args:
            sample_rate: Sampling rate.
            normalize: Normalisation type (``"NA"``, ``"per_feature"``,
                ``"all_feature"``).
            window_size: FFT window length in seconds.
            window_stride: FFT hop length in seconds (10 ms → 100 Hz frame rate).
            window: Window function (``"hann"``, ``"hamming"``, etc.).
            features: Number of mel channels (80 is standard).
            n_fft: Number of FFT bins.
            frame_splicing: Consecutive frame concatenation factor.
            dither: Noise magnitude for FFT stabilisation.
            pad_to: Pad signals to multiple of this many samples.
        """
        self.sample_rate = sample_rate
        self.window_stride = window_stride
        self.features = features

        self._init_kwargs = {
            "sample_rate": sample_rate,
            "normalize": normalize,
            "window_size": window_size,
            "window_stride": window_stride,
            "window": window,
            "features": features,
            "n_fft": n_fft,
            "frame_splicing": frame_splicing,
            "dither": dither,
            "pad_to": pad_to,
        }
        self._processor = AudioToMelSpectrogramPreprocessor(**self._init_kwargs)

    def __getstate__(self) -> dict[str, Any]:
        """State for pickling — drops the NeMo module, which is not picklable.

        NeMo's ``FilterbankFeatures.__init__`` assigns
        ``self.forward = torch.no_grad()(self.forward)`` (an instance-level
        wrapper), which pickle cannot serialize.  The preprocessor is stateless
        (config → module), so it is dropped here and rebuilt in
        :meth:`__setstate__` with the exact same construction kwargs.
        """
        state = self.__dict__.copy()
        state.pop("_processor", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled processor by rebuilding the NeMo module."""
        self.__dict__.update(state)
        self._processor = AudioToMelSpectrogramPreprocessor(**self._init_kwargs)

    def __call__(
        self,
        input_signal: torch.FloatTensor,
        length: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """Convert raw audio waveforms to log-mel spectrograms.

        Args:
            input_signal:
                Raw PCM audio waveforms of shape ``(batch_size, num_samples)``.
                All sequences in the batch must be padded to the same length;
                use ``length`` to indicate the number of valid samples per
                waveform.
            length:
                Number of valid samples per waveform, shape ``(batch_size,)``.

        Returns:
            processed_signal:
                Log-mel spectrograms.  Shape ``(batch_size, num_mel_bins,
                num_frames)``.
            processed_length:
                Number of valid mel frames per sequence, shape ``(batch_size,)``.
        """
        return self._processor(input_signal=input_signal, length=length)

    def to(self, device: torch.device | str) -> FastConformerMelProcessor:
        """Move the internal processor to the specified device.

        Args:
            device: Target device (e.g. ``"cuda:0"``, ``torch.device("cpu")``).

        Returns:
            Self, with the underlying NeMo processor moved to ``device``.
        """
        self._processor = self._processor.to(device)
        return self

    @property
    def device(self) -> torch.device:
        """The :class:`torch.device` where the processor currently resides."""
        return next(self._processor.parameters()).device

    def state_dict(self) -> dict[str, Any]:
        """Return a configuration dictionary suitable for serialisation.

        The ``AudioToMelSpectrogramPreprocessor`` has no trainable parameters,
        so only the configuration values used at construction time are returned.

        Returns:
            A flat dictionary of configuration parameters.
        """
        return {
            "sample_rate": self.sample_rate,
            "window_stride": self.window_stride,
            "features": self.features,
        }

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save processor configuration to a JSON file.

        The saved ``processor_config.json`` can be loaded back with
        :meth:`from_pretrained`.

        Args:
            save_directory:
                Path to an existing or new directory where
                ``processor_config.json`` will be written.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        config_path = save_directory / "processor_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2, ensure_ascii=False)

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
    ) -> str:
        """Upload ``processor_config.json`` to a HuggingFace Hub repository.

        Args:
            repo_id:
                Hub repository identifier (e.g. ``"username/model_name"``).
            token:
                HuggingFace authentication token.  Defaults to ``None``, in
                which case the token stored in the environment is used.
            private:
                Whether to create the repository as private if it does not
                already exist.

        Returns:
            The ``repo_id`` the processor was pushed to.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api = HfApi(token=token)
            api.create_repo(repo_id, exist_ok=True, private=private)
            api.upload_file(
                path_or_fileobj=str(Path(tmp_dir) / "processor_config.json"),
                path_in_repo="processor_config.json",
                repo_id=repo_id,
                repo_type="model",
            )
        return repo_id

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str | Path,
        **kwargs: Any,
    ) -> FastConformerMelProcessor:
        """Load a processor configuration from a previously saved directory.

        Args:
            save_directory:
                Path to a directory containing ``processor_config.json``.
                This can be a local path or a HuggingFace Hub model identifier
                (e.g. ``"username/model_name"``).
            **kwargs:
                Additional keyword arguments that override the saved
                configuration values (e.g. ``features=128``).

        Returns:
            A new :class:`FastConformerMelProcessor` initialised with the
            merged configuration.

        Raises:
            FileNotFoundError:
                If ``save_directory`` does not contain
                ``processor_config.json``.
        """
        config_path = Path(save_directory) / "processor_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Cannot find processor config at expected path: {config_path}. "
                f"Ensure the directory contains 'processor_config.json'."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config.update(kwargs)
        return cls(**config)
