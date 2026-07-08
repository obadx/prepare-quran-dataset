from dataclasses import dataclass


import numpy as np
from numpy import ndarray
import torch
from transformers import AutoFeatureExtractor


from .configuration_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCConfig,
)
from .modeling_rnn_streaming_multi_level_ctc import (
    Wav2Vec2BertForRNNStreamingMultilevelCTC,
)


def calc_frames(L, W=400, H=160, S=2) -> int:
    """Calulate the number of wav2vecBert processor num of frames given the input wav length
    This can be achives by:
    from transformers import AutoFeatureExtractor
    processor = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
    processor(np.zeros(15500), return_tensors='np', sampling_rate=16000)['attention_mask'][0].sum()
    args:
        L: wav length
        W: window length
        H: hop length
        S: stride
    """
    return max(0, int(1 + np.floor((L - W) / H)) // S)


def cal_samples_from_frames(F, W=400, H=160, S=2) -> int:
    """Calculate **full** samples from input grames"""
    return (S * F - 1) * H + W


@dataclass
class StreamingRNNInferenceOutput:
    phonemes_ids: list[int]


class Wav2Vec2BertForRNNStreamingMultilevelCTCInference(object):
    def __init__(
        self,
        model_name_or_path: str,
        sampling_rate=16000,
        device: str = "cpu",
        dtype=torch.bfloat16,
    ):
        self.sr = sampling_rate
        self.device = device
        self.dtype = dtype

        config = Wav2Vec2BertForRNNStreamingMultilevelCTCConfig.from_pretrained(
            model_name_or_path
        )
        self.config = config
        self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

        self.chunk_samples = cal_samples_from_frames(config.chunk_frames)
        self.input_samples = cal_samples_from_frames(
            config.chunk_frames + config.lookback_frames + config.lookahead_frames
        )
        self.lookback_samples = (
            (self.input_samples - self.chunk_samples)
            * config.lookback_frames
            // (config.lookback_frames + config.lookahead_frames)
        )
        self.lookahead_samples = (
            (self.input_samples - self.chunk_samples)
            * config.lookahead_frames
            // (config.lookback_frames + config.lookahead_frames)
        )
        assert (
            self.chunk_samples + self.lookahead_samples + self.lookahead_samples
            == self.input_samples
        )

        self.model = Wav2Vec2BertForRNNStreamingMultilevelCTC.from_pretrained(
            model_name_or_path
        ).to(device, dtype=dtype)
        self.model.eval()

        # reset the model and buffer
        self.reset()

    def get_first_input_samples(self) -> int:
        return self.input_samples

    def get_chunk_samples(self) -> int:
        return self.chunk_samples

    def reset(self):
        """reset the LSTM state and buffer"""
        self.buffer = np.zeros(self.lookahead_samples + self.lookback_samples)

        # zeroout lstm inputs

    def __call__(
        self, wav: ndarray, is_first=False
    ) -> Iterator[StreamingRNNInferenceOutput]:
        if is_first:
            assert len(wav) == self.input_samples
            input_samples = wav
            self.buffer = wav[-(self.lookback_samples + self.lookahead_samples) :]
        else:
            assert len(wav) == self.chunk_samples
            input_samples = np.pad(self.buffer, wav)
        self.buffer = wav[-(self.lookback_samples + self.lookahead_samples) :]

        inputs = self.processor(
            input_samples, sampling_rate=self.sr, return_tensors="pt"
        )
