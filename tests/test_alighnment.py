import json
from transformers import AutoFeatureExtractor
from prepare_quran_dataset.modeling.modeling_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTC,
)
from librosa.core import load
import torch
from torchaudio.models.decoder import ctc_decoder
import matplotlib.pyplot as plt

from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer
from prepare_quran_dataset.modeling.vocab import PAD_TOKEN


def plot_alignments(waveform, emission, tokens, timesteps, sample_rate):
    t = torch.arange(waveform.size(0)) / sample_rate
    ratio = waveform.size(0) / emission.size(1) / sample_rate

    chars = []
    words = []
    word_start = None
    for token, timestep in zip(tokens, timesteps * ratio):
        if token == "|":
            if word_start is not None:
                words.append((word_start, timestep))
            word_start = None
        else:
            chars.append((token, timestep))
            if word_start is None:
                word_start = timestep

    fig, axes = plt.subplots(3, 1)

    def _plot(ax, xlim):
        ax.plot(t, waveform)
        for token, timestep in chars:
            ax.annotate(token.upper(), (timestep, 0.5))
        for word_start, word_end in words:
            ax.axvspan(word_start, word_end, alpha=0.1, color="red")
        ax.set_ylim(-0.6, 0.7)
        ax.set_yticks([0])
        ax.grid(True, axis="y")
        ax.set_xlim(xlim)

    _plot(axes[0], (0.3, 2.5))
    _plot(axes[1], (2.5, 4.7))
    _plot(axes[2], (4.7, 6.9))
    axes[2].set_xlabel("time (sec)")
    fig.tight_layout()
    plt.savefig("timeings.png")


if __name__ == "__main__":
    repo_id = "obadx/muaalem-model-v2_1"
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(repo_id)
    multi_level_tokenizer = MultiLevelTokenizer(repo_id)
    processor = AutoFeatureExtractor.from_pretrained(repo_id)

    device = torch.device("cpu")
    dtype = torch.bfloat16
    model.to(device, dtype=dtype)

    # wave, _ = load("/home/abdullah/Downloads/test_sample_2.mp3", sr=16000)
    # wave, _ = load("/home/abdullah/Downloads/test_sample_3.mp3", sr=16000)
    # wave, _ = load("/home/abdullah/Downloads/test_sample.mp3", sr=16000)
    wave, _ = load("/home/abdullah/Downloads/no_qalqala_2.wav", sr=16000)
    # wave, _ = load("/home/abdullah/Downloads/test.wav", sr=16000)
    features = processor(wave, sampling_rate=16000, return_tensors="pt")
    features = {k: v.to(device, dtype=dtype) for k, v in features.items()}
    outs = model(**features, return_dict=False)[0]

    with open(
        "vocab.json",
        "r",
        encoding="utf-8",
    ) as f:
        vocab = json.load(f)

    beam_search_decoder = ctc_decoder(
        lexicon=None,
        tokens=list(vocab["phonemes"].keys()) + ["|"],
        nbest=1,
        beam_size=1500,
        blank_token=PAD_TOKEN,
    )
    for level in outs:
        outs[level] = outs[level].to(torch.float32)
    level_to_pred_ids = {}
    for level in outs:
        level_to_pred_ids[level] = beam_search_decoder(outs[level])

    for idx in range(len(level_to_pred_ids["phonemes"][0])):
        print(level_to_pred_ids["phonemes"][0][0].tokens)
        print(level_to_pred_ids["phonemes"][0][0].timesteps)
        phonemes_id_to_token = list(vocab["phonemes"].keys()) + ["|"]

        phonemes = [
            phonemes_id_to_token[int(t)]
            for t in level_to_pred_ids["phonemes"][0][0].tokens
        ]
        print("".join(phonemes))

    # plot_alignments(
    #     torch.tensor(wave).unsqueeze(0),
    #     outs["phonemes"],
    #     [
    #         phonemes_id_to_token[int(t)]
    #         for t in level_to_pred_ids["phonemes"][0][0].tokens
    #     ],
    #     level_to_pred_ids["phonemes"][0][0].timesteps,
    #     16000,
    # )

    # decoded_outs = multi_level_tokenizer.decode(
    #     level_to_pred_ids,
    #     place_zeros_in_between=False,
    # )
    # print(json.dumps(decoded_outs, indent=1, ensure_ascii=False))
