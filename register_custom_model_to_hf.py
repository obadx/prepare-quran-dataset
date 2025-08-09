from multi_level_ctc_model.configuration_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTCConfig,
)
from multi_level_ctc_model.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from transformers import AutoConfig, AutoModel, AutoModelForCTC


AutoConfig.register("multi_level_ctc", Wav2Vec2BertForMultilevelCTCConfig)
AutoModel.register(Wav2Vec2BertForMultilevelCTCConfig, Wav2Vec2BertForMultilevelCTC)
AutoModelForCTC.register(
    Wav2Vec2BertForMultilevelCTCConfig, Wav2Vec2BertForMultilevelCTC
)
