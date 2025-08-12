import json
from transformers import AutoFeatureExtractor
from prepare_quran_dataset.modeling.modeling_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTC,
)
from librosa.core import load
import torch

from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer

if __name__ == "__main__":
    repo_id = "obadx/Muaalem-model-dev"
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(repo_id)
    multi_level_tokenizer = MultiLevelTokenizer(repo_id)
    processor = AutoFeatureExtractor.from_pretrained(repo_id)

    device = torch.device("cpu")
    dtype = torch.bfloat16
    model.to(device, dtype=dtype)

    # wave, _ = load("/home/abdullah/Downloads/test_sample_2.mp3", sr=16000)
    wave, _ = load("/home/abdullah/Downloads/test_sample_3.mp3", sr=16000)
    # wave, _ = load("/home/abdullah/Downloads/test_sample.mp3", sr=16000)
    # wave, _ = load("/home/abdullah/Downloads/test.wav", sr=16000)
    features = processor(wave, sampling_rate=16000, return_tensors="pt")
    features = {k: v.to(device, dtype=dtype) for k, v in features.items()}
    outs = model(**features, return_dict=False)[0]

    level_to_pred_ids = {k: torch.argmax(v, dim=-1) for k, v in outs.items()}
    decoded_outs = multi_level_tokenizer.decode(
        level_to_pred_ids,
        place_zeros_in_between=False,
    )
    print(json.dumps(decoded_outs, indent=1, ensure_ascii=False))
