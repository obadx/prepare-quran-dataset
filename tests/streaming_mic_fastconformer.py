"""Backwards-compatible shim.

The implementation moved to
``prepare_quran_dataset.modeling_fastconformer_cache_aware.streaming_mic`` so that the
mic app lives with the model rather than under ``tests/``, and the CTC collapse rules
live in the dependency-free ``prepare_quran_dataset.ctc_decoding`` where they can be
unit-tested without a GPU.

This module is kept only so the Colab notebook's
``from streaming_mic_fastconformer import ...`` keeps working unchanged.  New code
should import from the package directly.
"""

from prepare_quran_dataset.ctc_decoding import (  # noqa: F401
    BLANK_ID,
    EOS_ID,
    SPECIAL_IDS,
    collapse_carry,
    ctc_collapse,
)
from prepare_quran_dataset.modeling_fastconformer_cache_aware.streaming_mic import (  # noqa: F401
    DEFAULT_MODEL_ID,
    HALF_NFFT,
    HOP,
    MARGIN_FRAMES,
    SAMPLE_RATE,
    FastConformerMicStreamer,
    StreamingDiagnostics,
    StreamingMelFeaturizer,
    build_demo,
    load_model_and_vocab,
    main,
    self_test,
)

if __name__ == "__main__":
    raise SystemExit(main())
