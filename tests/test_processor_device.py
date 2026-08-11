"""The mel front-end must follow the model onto the GPU.

``FastConformerMelProcessor`` is deliberately not an ``nn.Module`` (registering
it would drag NeMo's ``window``/``fb`` buffers into ``state_dict()``), so
``model.to("cuda")`` never reaches it.  It syncs itself to the input's device
inside ``__call__`` instead; these tests pin that behaviour down.

Run with::

    uv run pytest tests/test_processor_device.py -v
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo.collections.asr")

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (  # noqa: E402
    FastConformerCacheAwareMultilevelCTC,
    FastConformerCacheAwareMultilevelCTCConfig,
    FastConformerMelProcessor,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

SAMPLES = 16000


def _build_model():
    config = FastConformerCacheAwareMultilevelCTCConfig(
        level_to_vocab_size={"phonemes": 8},
        att_context_size=[6, 2],
        n_layers=2,
        d_model=64,
        n_heads=2,
        subsampling_conv_channels=32,
    )
    return FastConformerCacheAwareMultilevelCTC(config).eval()


def test_device_property_does_not_raise_on_cpu():
    """The NeMo preprocessor has buffers but no parameters.

    Reading the device off ``parameters()`` raises ``StopIteration``; it has to
    come from ``buffers()``.
    """
    processor = FastConformerMelProcessor()
    assert processor.device == torch.device("cpu")


@requires_cuda
def test_processor_follows_model_to_cuda():
    model = _build_model().to("cuda")
    audio = torch.randn(1, SAMPLES, device="cuda")
    length = torch.tensor([SAMPLES], device="cuda")

    # Must not raise "Expected all tensors to be on the same device".
    out = model(raw_audio=audio, audio_length=length)

    assert out.logits["phonemes"].device.type == "cuda"
    assert model.processor.device.type == "cuda"


@requires_cuda
def test_processor_stays_float32_under_bf16_model():
    """The mel front-end must not be dragged into bfloat16 with the model."""
    model = _build_model().to("cuda")
    processor = model.processor

    processor.to("cuda")
    dtypes = {buf.dtype for buf in processor._processor.buffers()}
    assert dtypes == {torch.float32}
