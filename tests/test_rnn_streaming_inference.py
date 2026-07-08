from prepare_quran_dataset.modeling_streaming_rnn.inference import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCInference,
)
from prepare_quran_dataset.modeling_streaming_rnn.configuration_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCConfig,
)


if __name__ == "__main__":
    model_name_or_path = "./results-streaming-rnn-v2/checkpoint-35480"
    rnn_inf = Wav2Vec2BertForRNNStreamingMultilevelCTCInference(model_name_or_path)
