from transformers.configuration_utils import PretrainedConfig


class WhisperEncoderForMultilevelCTCConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`WhisperEncoderForMultilevelCTC`]. It is used to
    instantiate a Whisper Encoder model for Multilevel CTC according to the specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        level_to_vocab_size (`dict[str, int]`, *optional*):
            Every level has its own vocabulary: {'phonemes': 44, 'hams_or_jahr': 3, ....}
        level_to_loss_weight (`dict[str, float]`, *optional*):
            Every level has its own loss weight such that the sum of all levels adds to 1.
            If you supply only one level: the rest of the levels will have loss weight of (1-given_loss_weight) / number of rest of levels

        d_model (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the encoder.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Whether to scale the embeddings by sqrt(d_model).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features.
        num_mel_bins (`int`, *optional*, defaults to 80):
            The number of mel frequency bins.
        apply_spec_augment (`bool`, *optional*, defaults to `False`):
            Whether to apply SpecAugment data augmentation.
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage of feature vectors along the time axis to mask.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2):
            Minimum number of masks along the time axis.
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage of feature vectors along the feature axis to mask.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0):
            Minimum number of masks along the feature axis.
        ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`.
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`.
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights.
        classifier_proj_size (`int`, *optional*, defaults to 256):
            Dimensionality of the projection before token mean-pooling for classification.
    """

    model_type = "multi_level_ctc_whisper"

    def __init__(
        self,
        level_to_vocab_size: dict[str, int] = {},
        level_to_loss_weight: dict[str, float] = {"phonemes": 0.4},
        d_model=768,
        encoder_layers=12,
        encoder_attention_heads=12,
        encoder_ffn_dim=3072,
        encoder_layerdrop=0.0,
        activation_function="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=False,
        max_source_positions=1500,
        num_mel_bins=80,
        apply_spec_augment=False,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=50257,
        eos_token_id=50257,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layerdrop = encoder_layerdrop
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.num_mel_bins = num_mel_bins
        self.level_to_vocab_size = level_to_vocab_size
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.use_cache = use_cache

        loss_weights_sum = sum(level_to_loss_weight.values())
        if loss_weights_sum > 1:
            raise ValueError(
                f"The sum of loss weight per level has to be less than one! got: `{level_to_loss_weight}`"
            )
        unmentioned_loss_levels_count = len(
            [l for l in self.level_to_vocab_size if l not in level_to_loss_weight]
        )
        for level in self.level_to_vocab_size:
            if level not in level_to_loss_weight:
                level_to_loss_weight[level] = (
                    1 - loss_weights_sum
                ) / unmentioned_loss_levels_count
        self.level_to_loss_weight = level_to_loss_weight

        # fine-tuning config parameters for SpecAugment
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        # ctc loss
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # SequenceClassification-specific parameter
        self.classifier_proj_size = classifier_proj_size

    @property
    def inputs_to_logits_ratio(self):
        return 2


__all__ = ["WhisperEncoderForMultilevelCTCConfig"]
