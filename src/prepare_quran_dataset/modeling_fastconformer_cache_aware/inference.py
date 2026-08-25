from collections.abc import Sequence

import torch
from librosa.core import load
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)
from nemo.utils import logging

from ..modeling_streaming_rnn.multi_level_tokenizer import MultiLevelTokenizer
from .modeling_fastconformer_cache_aware_ctc import (
    FastConformerCache,
    FastConformerCacheAwareMultilevelCTC,
)
from .processor import FastConformerMelProcessor

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
reference script: https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
"""


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


class MuaalemCacheAwareStreamingAudioBuffer(CacheAwareStreamingAudioBuffer):
    def __init__(
        self,
        model: FastConformerCacheAwareMultilevelCTC,
        preprocessor: FastConformerMelProcessor,
        online_normalization=None,
        pad_and_drop_preencoded=False,
    ):
        """
        Args:
            model: An ASR model.
            online_normalization (bool): whether to perform online normalization per chunk or
            normalize the whole audio before chunking
            pad_and_drop_preencoded (bool): if true pad first audio chunk and always drop preencoded
        """
        self.model = model
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None
        self.step = 0
        self.pad_and_drop_preencoded = pad_and_drop_preencoded
        self.preprocessor = preprocessor

        self.online_normalization = online_normalization
        model.encoder.setup_streaming_params()
        self.streaming_cfg = model.encoder.streaming_cfg

        self.input_features = model.encoder._feat_in

        self.preprocessor = self.extract_preprocessor()

        if hasattr(model.encoder, "pre_encode") and hasattr(
            model.encoder.pre_encode, "get_sampling_frames"
        ):
            self.sampling_frames = model.encoder.pre_encode.get_sampling_frames()
        else:
            self.sampling_frames = None

    def extract_preprocessor(self):
        # TODO:
        return self.preprocessor


@torch.no_grad()
def infer_fastconformer_streaming(
    audio_sources: Sequence[str | bytes],
    device: str | torch.device,
    dtype: torch.dtype,
    model: FastConformerCacheAwareMultilevelCTC,
    processor: FastConformerMelProcessor,
    multi_level_tokenizer: MultiLevelTokenizer,
    sampling_rate=16000,
) -> dict[str, list[list[int]]]:
    streaming_buffer = MuaalemCacheAwareStreamingAudioBuffer(
        model=model,
        preprocessor=processor,
    )
    # stream audio files in a manifest file in batched mode
    batch_size = len(audio_sources)

    for sample_idx, source in enumerate(audio_sources):
        _ = streaming_buffer.append_audio_file(source, stream_id=-1)
    final_offline_tran = None

    cache = model.get_initial_cache_state(batch_size=batch_size)

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
        # otherwise the last outputs would get dropped
        chunk_audio = chunk_audio.to(dtype)
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal=chunk_audio,
            processed_signal_length=chunk_lengths,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=streaming_buffer.is_buffer_empty(),
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                asr_model, step_num, pad_and_drop_preencoded
            ),
            return_transcription=True,
        )
    streaming_buffer.reset_buffer()


__all__ = ["infer_fastconformer_streaming"]
