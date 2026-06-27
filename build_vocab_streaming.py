from prepare_quran_dataset.modeling_streaming_rnn.vocab import (
    build_quran_phoneme_script_vocab,
)

if __name__ == "__main__":
    build_quran_phoneme_script_vocab("./vocab_streaming/vocab.json")
