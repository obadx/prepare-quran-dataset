"""Configuration for the FastConformer cache-aware multi-level CTC model.

This module defines :class:`FastConformerCacheAwareMultilevelCTCConfig`, a
subclass of :class:`~transformers.PretrainedConfig` that bundles together:

- Mel-spectrogram processor parameters (``processor_kwargs``)
- FastConformer encoder architecture parameters
- Multi-level CTC vocabulary sizes and loss weights
- CTC loss hyper-parameters

The configuration is serialisable via HuggingFace's ``save_pretrained`` /
``from_pretrained`` API, enabling Hub-based distribution.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class FastConformerCacheAwareMultilevelCTCConfig(PretrainedConfig):
    r"""Configuration for :class:`~modeling_fastconformer_cache_aware_ctc.FastConformerCacheAwareMultilevelCTC`.

    This is the configuration class to store the architecture and hyper-parameter
    settings of a FastConformer-based multi-level CTC model.  It is used to
    instantiate a :class:`~modeling_fastconformer_cache_aware_ctc.FastConformerCacheAwareMultilevelCTC`
    according to the specified arguments.

    Instantiating a configuration with the defaults will yield a configuration
    similar to the **FastConformer-CTC-BPE-Streaming** architecture used in
    the NeMo test suite (17-layer, d_model=512, subsampling_factor=4,
    att_context_size=[78, 12], chunked_limited attention style).

    Configuration objects inherit from :class:`~transformers.PretrainedConfig`
    and can be used to control the model outputs.  Read the documentation from
    :class:`~transformers.PretrainedConfig` for more information.

    .. rubric:: Configuration sections

    **Processor parameters** (``processor_kwargs``)
        These are forwarded to :class:`~processor.FastConformerMelProcessor`.
        The defaults match NeMo's ``AudioToMelSpectrogramPreprocessor`` with
        16 kHz sample rate, 25 ms window, 10 ms stride, 80 mel bands, and
        512-point FFT.  ``normalize="NA"`` is used because streaming models
        should not rely on input normalisation (the whole utterance is not
        available at the first chunk).

    **Encoder parameters**
        Nested under ``self``, these define the FastConformer / Conformer
        encoder architecture.  Key parameters include:

        *   ``feat_in`` — number of input mel channels (default 80).
        *   ``n_layers`` — number of Conformer blocks (default 17).
        *   ``d_model`` — hidden / channel dimension (default 512).
        *   ``subsampling_factor`` — temporal subsampling factor (default 4).
        *   ``att_context_size`` — left/right attention context ``[left, right]``
            (default ``[90, 17]``).
        *   ``att_context_style`` — attention masking strategy; set to
            ``"chunked_limited"`` for cache-aware streaming training.

    **Multi-level CTC parameters**
        *   ``level_to_vocab_size`` — dictionary mapping each output level
            (e.g. ``"phonemes"``, ``"shidda_or_rakhawa"``) to its vocabulary
            size.
        *   ``level_to_loss_weight`` — per-level loss weights.  Weights are
            **not** required to sum to 1; the remaining weight mass is
            distributed equally among unmentioned levels.  If a level is not
            present in this dictionary, its weight is automatically computed.

    **CTC loss parameters**
        *   ``ctc_loss_reduction`` — ``"mean"`` or ``"sum"``.
        *   ``ctc_zero_infinity`` — whether to zero infinite losses.

    Args:
        level_to_vocab_size:
            Dictionary mapping each output level name to its vocabulary size.
            At least one level must be provided.  Vocabulary sizes determine
            the output dimensionality of each CTC head (e.g.
            ``{"phonemes": 44, "shidda_or_rakhawa": 3}``).
        level_to_loss_weight:
            Dictionary mapping level names to their CTC loss weight multiplier.
            Unmentioned levels receive the remaining weight mass equally.
            Defaults to ``{"phonemes": 0.4}``.
        processor_kwargs:
            Keyword arguments forwarded to
            :class:`~processor.FastConformerMelProcessor`.  Defaults match
            the standard FastConformer front-end (16 kHz, 80 mel bands,
            10 ms hop, ``"NA"`` normalisation).

        # ---- Encoder architecture ----
        feat_in:
            Number of input mel-spectrogram channels.  Defaults to ``80``.
        feat_out:
            Output feature dimension.  ``-1`` means ``d_model`` is used.
            Defaults to ``-1``.
        n_layers:
            Number of Conformer blocks in the encoder.  Defaults to ``17``.
        d_model:
            Hidden / channel dimension of the encoder.  Defaults to ``512``.
        use_bias:
            Whether to use bias in linear layers of the encoder.
            Defaults to ``True``.
        subsampling:
            Subsampling method.  ``"dw_striding"`` for depth-wise striding,
            ``"striding"``, ``"vggnet"``, etc.  Defaults to ``"dw_striding"``.
        subsampling_factor:
            Total temporal subsampling factor (must be a power-of-2 for
            striding methods).  Defaults to ``4``.
        subsampling_conv_channels:
            Number of channels in the subsampling convolution layers.
            Defaults to ``256``.
        causal_downsampling:
            Whether to use causal (left-only) padding in subsampling
            convolutions.  Must be ``True`` for cache-aware streaming.
            Defaults to ``True``.
        ff_expansion_factor:
            Expansion factor for the feed-forward sub-layer inside each
            Conformer block.  Defaults to ``4``.
        self_attention_model:
            Self-attention variant.  ``"rel_pos"`` for relative positional
            encodings (Transformer-XL style).  Defaults to ``"rel_pos"``.
        n_heads:
            Number of attention heads.  Defaults to ``8``.
        att_context_size:
            Attention context in encoder frames.  Format
            ``[left_context, right_context]`` or, to enable a **constant
            lookahead delay**, ``[left_context, right_context,
            constant_lookahead_delay]``.  The optional third entry is the
            number of future encoder frames each position may attend (the
            chunk attends left and right, with the right context fixed to
            ``constant_lookahead_delay``).  Defaults to ``[78, 12]``
            (1040 ms worst-case latency with 40 ms per encoder frame).
        att_context_style:
            Attention context style.  Set to ``"chunked_limited"`` for
            cache-aware streaming training.  Defaults to ``"chunked_limited"``.
        xscaling:
            Whether to scale encoder inputs by ``sqrt(d_model)``.
            Defaults to ``True``.
        pos_emb_max_len:
            Maximum length for relative position embeddings.
            Defaults to ``5000``.
        conv_kernel_size:
            Kernel size for convolution sub-layers in Conformer blocks.
            Defaults to ``9``.
        conv_norm_type:
            Normalisation type in convolution sub-layers.  ``"layer_norm"``
            for streaming models; ``"batch_norm"`` for full-context models.
            Defaults to ``"layer_norm"``.
        conv_context_size:
            Convolution context size.  ``"causal"`` for streaming models;
            ``None`` for full-context.  Defaults to ``"causal"``.
        dropout:
            Overall dropout probability (applied in most sub-layers).
            Defaults to ``0.1``.
        dropout_pre_encoder:
            Dropout applied before the encoder.  Defaults to ``0.1``.
        dropout_emb:
            Dropout applied to embeddings.  Defaults to ``0.0``.
        dropout_att:
            Dropout applied in attention modules.  Defaults to ``0.1``.
        stochastic_depth_drop_prob:
            Probability for stochastic depth (layer dropping).  ``0.0``
            disables it.  Defaults to ``0.0``.
        stochastic_depth_mode:
            Mode for stochastic depth.  ``"linear"`` or ``"uniform"``.
            Defaults to ``"linear"``.
        stochastic_depth_start_layer:
            First layer index to apply stochastic depth from (1-indexed).
            Defaults to ``1``.

        # ---- CTC loss ----
        ctc_loss_reduction:
            Reduction applied to the CTC loss.  ``"mean"`` or ``"sum"``.
            Defaults to ``"mean"``.
        ctc_zero_infinity:
            If ``True``, zero infinite losses and their gradients.
            Defaults to ``False``.
        pad_token_id:
            Token ID used for padding and as the CTC blank label.
            Defaults to ``0``.

    Example:
        >>> config = FastConformerCacheAwareMultilevelCTCConfig(
        ...     level_to_vocab_size={"phonemes": 44, "shidda_or_rakhawa": 3},
        ...     level_to_loss_weight={"phonemes": 0.4},
        ...     n_layers=17, d_model=512,
        ... )
        >>> config.level_to_loss_weight  # doctest: +SKIP
        {'phonemes': 0.4, 'shidda_or_rakhawa': 0.3}
        >>> config.model_type
        'fastconformer-cache-aware'
    """

    model_type: str = "fastconformer-cache-aware"

    def __init__(
        self,
        # ---- Multi-level CTC ----
        level_to_vocab_size: dict[str, int] | None = None,
        level_to_loss_weight: dict[str, float] | None = None,
        # ---- Processor ----
        processor_kwargs: dict[str, Any] | None = None,
        # ---- Encoder architecture ----
        feat_in: int = 80,
        feat_out: int = -1,
        n_layers: int = 17,
        d_model: int = 512,
        use_bias: bool = True,
        subsampling: str = "dw_striding",
        subsampling_factor: int = 4,
        subsampling_conv_channels: int = 256,
        causal_downsampling: bool = True,
        ff_expansion_factor: int = 4,
        self_attention_model: str = "rel_pos",
        n_heads: int = 8,
        att_context_size: list[int] | None = [90, 17],
        att_context_style: str = "chunked_limited",
        xscaling: bool = True,
        pos_emb_max_len: int = 5000,
        conv_kernel_size: int = 9,
        conv_norm_type: str = "layer_norm",
        conv_context_size: str | None = "causal",
        dropout: float = 0.1,
        dropout_pre_encoder: float = 0.1,
        dropout_emb: float = 0.0,
        dropout_att: float = 0.1,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        # ---- CTC loss ----
        ctc_loss_reduction: str = "mean",
        ctc_zero_infinity: bool = False,
        pad_token_id: int = 0,
        # ---- HF boilerplate ----
        **kwargs: Any,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # ---- Processor ----
        self.processor_kwargs = processor_kwargs or {
            "sample_rate": 16000,
            "normalize": "NA",
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "features": feat_in,
            "n_fft": 512,
            "frame_splicing": 1,
            "dither": 0.00001,
            "pad_to": 0,
        }

        # ---- Multi-level CTC ----
        if level_to_vocab_size is None:
            level_to_vocab_size = {}
        if not level_to_vocab_size:
            raise ValueError(
                "At least one CTC level must be defined in "
                "`level_to_vocab_size`.  Received an empty dictionary."
            )
        self.level_to_vocab_size = level_to_vocab_size

        if level_to_loss_weight is None:
            level_to_loss_weight = {}
        loss_weights_sum = sum(level_to_loss_weight.values())
        if loss_weights_sum > 1.0:
            raise ValueError(
                f"The sum of explicit loss weights ({loss_weights_sum:.4f}) "
                f"exceeds 1.0.  Total must be ≤ 1.0.  "
                f"Got: {level_to_loss_weight}"
            )
        unmentioned_levels = [
            l for l in level_to_vocab_size if l not in level_to_loss_weight
        ]
        if unmentioned_levels:
            remaining_weight = 1.0 - loss_weights_sum
            per_level = remaining_weight / len(unmentioned_levels)
            for level in unmentioned_levels:
                level_to_loss_weight[level] = per_level
        self.level_to_loss_weight = level_to_loss_weight

        # ---- Encoder architecture ----
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.n_layers = n_layers
        self.d_model = d_model
        self.use_bias = use_bias
        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.causal_downsampling = causal_downsampling
        self.ff_expansion_factor = ff_expansion_factor
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        if isinstance(att_context_size, list):
            if len(att_context_size) not in (2, 3):
                raise ValueError(
                    "`att_context_size` must be `[left, right]` or "
                    "`[left, right, constant_lookahead_delay]`.  "
                    f"Got {len(att_context_size)} elements: {att_context_size}."
                )
            if len(att_context_size) == 3 and att_context_size[2] < 0:
                raise ValueError(
                    "`constant_lookahead_delay` (att_context_size[2]) must be "
                    f"non-negative.  Got {att_context_size[2]}."
                )
        self.att_context_size = att_context_size or -1
        self.att_context_style = att_context_style
        self.xscaling = xscaling
        self.pos_emb_max_len = pos_emb_max_len
        self.conv_kernel_size = conv_kernel_size
        self.conv_norm_type = conv_norm_type
        self.conv_context_size = conv_context_size
        self.dropout = dropout
        self.dropout_pre_encoder = dropout_pre_encoder
        self.dropout_emb = dropout_emb
        self.dropout_att = dropout_att
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob
        self.stochastic_depth_mode = stochastic_depth_mode
        self.stochastic_depth_start_layer = stochastic_depth_start_layer

        # ---- CTC loss ----
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  level_to_vocab_size={self.level_to_vocab_size},\n"
            f"  level_to_loss_weight={self.level_to_loss_weight},\n"
            f"  n_layers={self.n_layers}, d_model={self.d_model},\n"
            f"  subsampling_factor={self.subsampling_factor},\n"
            f"  att_context_size={self.att_context_size},\n"
            f"  att_context_style={self.att_context_style},\n"
            f")"
        )
