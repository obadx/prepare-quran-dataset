from typing import Iterator
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
            model_name_or_path,
            max_chunk_batch=1,
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
            self.chunk_samples + self.lookahead_samples + self.lookback_samples
            == self.input_samples
        )

        self.model = Wav2Vec2BertForRNNStreamingMultilevelCTC.from_pretrained(
            model_name_or_path, config=config
        ).to(device, dtype=dtype)
        self.model.eval()

        # reset the model and buffer
        self.reset()

    def get_chunk_samples(self) -> int:
        return self.chunk_samples

    def reset(self):
        """reset the LSTM state and buffer"""
        self.buffer = np.zeros(self.lookahead_samples + self.lookback_samples)
        self.rnn_history = None

    @torch.inference_mode()
    def __call__(
        self,
        wav: ndarray,
    ) -> Iterator[StreamingRNNInferenceOutput]:
        assert len(wav) == self.chunk_samples
        input_samples = np.concatenate([self.buffer, wav])
        self.buffer = wav[-(self.lookback_samples + self.lookahead_samples) :]

        inputs = self.processor(
            input_samples,
            sampling_rate=self.sr,
            return_tensors="pt",
            do_normalize_per_mel_bins=False,
        )
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}
        outputs = self.model(
            **inputs,
            stream_inference=True,
            rnn_history=self.rnn_history,
            return_dict=True,
        )
        phoneme_ids = outputs.logits["phonemes"].argmax(dim=-1)[0].tolist()
        self.rnn_history = outputs.rnn_history
        yield StreamingRNNInferenceOutput(phonemes_ids=phoneme_ids)
