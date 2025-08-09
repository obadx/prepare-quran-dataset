import json
from multi_level_ctc_model.configuration_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTCConfig,
)
from multi_level_ctc_model.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from transformers import AutoConfig, AutoModel, AutoModelForCTC

if __name__ == "__main__":
    # Register the model
    AutoConfig.register("multi_level_ctc", Wav2Vec2BertForMultilevelCTCConfig)
    AutoModel.register(Wav2Vec2BertForMultilevelCTCConfig, Wav2Vec2BertForMultilevelCTC)
    AutoModelForCTC.register(
        Wav2Vec2BertForMultilevelCTCConfig, Wav2Vec2BertForMultilevelCTC
    )

    # pushin ghte model to the hub
    Wav2Vec2BertForMultilevelCTCConfig.register_for_auto_class()
    Wav2Vec2BertForMultilevelCTC.register_for_auto_class("AutoModel")
    Wav2Vec2BertForMultilevelCTC.register_for_auto_class("AutoModelForCTC")

    with open("./vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}

    config = Wav2Vec2BertForMultilevelCTCConfig(level_to_vocab_size=level_to_vocab_size)
    model = Wav2Vec2BertForMultilevelCTC(config)

    # TODO: add weights
    # model.model.load_state_dict(pretrained_model.state_dict())

    print("Pushing the model to the hub")
    model.push_to_hub("obadx/test-model-tok")
