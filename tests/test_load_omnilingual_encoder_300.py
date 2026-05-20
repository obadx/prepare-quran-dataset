from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    AutoFeatureExtractor,
)
import torch


if __name__ == "__main__":
    config = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=False,
        adapter_stride=1,
    )
    processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    model = Wav2Vec2ForPreTraining.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        # "facebook/wav2vec2-large-xlsr-53",
        config=config,
        dtype="float32",
    )
    print(model)
    print(processor)

    batch_size = 2
    inputs = processor(
        [[0] * 16000] * batch_size, sampling_rate=16000, return_tensors="pt"
    )
    model_output = model(**{k: v.float() for k, v in inputs.items()})
    print({k: v.shape for k, v in model_output.items()})
