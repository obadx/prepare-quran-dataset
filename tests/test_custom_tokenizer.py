import json

from transformers import Wav2Vec2CTCTokenizer, AutoTokenizer
from transformers.tokenization_utils_base import AddedToken, BatchEncoding

from prepare_quran_dataset.modeling.vocab import PAD_TOKEN


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}


class Wav2Vec2CTCTMultilevelokenizer(Wav2Vec2CTCTokenizer):
    """
    Constructs a Wav2Vec2CTC tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            The token used for defining the end of a word.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.
        target_lang (`str`, *optional*):
            A target language the tokenizer should set by default. `target_lang` has to be defined for multi-lingual,
            nested vocabulary such as [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all).

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
        target_lang=None,
        **kwargs,
    ):
        self._word_delimiter_token = word_delimiter_token

        self.do_lower_case = do_lower_case
        self.replace_word_delimiter_char = replace_word_delimiter_char
        self.target_lang = target_lang

        ## custom part
        self.hamo = "hamo"

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)

        # if target lang is defined vocab must be a nested dict
        # with each target lang being one vocabulary
        if target_lang is not None:
            self.encoder = self.vocab[target_lang]
        else:
            self.encoder = self.vocab

        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            replace_word_delimiter_char=replace_word_delimiter_char,
            target_lang=target_lang,
            **kwargs,
        )

        # make sure that tokens made of several
        # characters are not split at tokenization
        for token in self.encoder:
            if len(token) > 1:
                self.add_tokens(
                    AddedToken(token, rstrip=True, lstrip=True, normalized=False)
                )


# Example usage and testing
if __name__ == "__main__":
    tokenizer = Wav2Vec2CTCTMultilevelokenizer.from_pretrained(
        "./", pad_token=PAD_TOKEN, target_lang="phonemes"
    )
    print(tokenizer)
    print(tokenizer.config)

    # AutoTokenizer.register(slow_tokenizer_class=Wav2Vec2CTCTMultilevelokenizer)

    # Push to hub (uncomment when ready)
    # tokenizer.push_to_hub("your-username/multilevel-phonetic-tokenizer")    # Push to hub (uncomment when ready)
    # tokenizer.push_to_hub("obadx/test-model-tok")
