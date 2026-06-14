from typing import get_origin, Literal, Any

from quran_transcript import SifaOutput, quran_phonetizer
from transformers import Wav2Vec2CTCTokenizer

from .vocab import PAD_TOKEN, PAD_TOKEN_IDX, SIFAT_ATTR_TO_ARABIC, SIFAT_ATTR_TO_ENGLISH


def add_zero_between(L, x=PAD_TOKEN_IDX):
    out = []
    for i, item in enumerate(L):
        out.append(item)
        if i < len(L) - 1:  # Don't add zero after the last element
            out.append(0)
    return out


class MultiLevelTokenizer:
    def __init__(self, model_name_or_path: str):
        self.levels = ["phonemes"]
        for fieldname, fieldinfo in SifaOutput.model_fields.items():
            if get_origin(fieldinfo.annotation) == Literal:
                self.levels.append(fieldname)

        self.level_to_tokenizer = {}
        for level in self.levels:
            self.level_to_tokenizer[level] = Wav2Vec2CTCTokenizer.from_pretrained(
                model_name_or_path, pad_token=PAD_TOKEN, target_lang=level
            )

        self.level_to_id_vocab = self.get_level_to_id_to_voab()
        self.sifat_level_to_id_to_en_vocab = self.get_sifat_levels_to_en_name()

    def get_tokenizer(self):
        return self.level_to_tokenizer["phonemes"]

    @property
    def vocab(self):
        return self.get_tokenizer().vocab

    @property
    def id_to_vocab(self):
        return self.level_to_id_vocab

    @property
    def sifat_to_en_vocab(self):
        return self.sifat_level_to_id_to_en_vocab

    def tokenize(
        self,
        phonetic_script: list[str] | str,
        sifat: list[list[SifaOutput | dict]] | list[SifaOutput | dict],
        to_dict=False,
        **kwargs,
    ) -> dict:
        if isinstance(phonetic_script, str):
            phonetic_script = [phonetic_script]
        if not isinstance(sifat[0], list):
            sifat = [sifat]

        if isinstance(sifat[0][0], dict):
            sifat = [[SifaOutput(**s) for s in inner_list] for inner_list in sifat]

        level_to_text_list = {}
        for level in self.levels:
            if level == "phonemes":
                text_list = phonetic_script
            else:
                text_list = [
                    "".join(
                        [SIFAT_ATTR_TO_ARABIC[getattr(s, level)] for s in inner_list]
                    )
                    for inner_list in sifat
                ]
            level_to_text_list[level] = text_list

        level_to_tokenized = {}
        for level in self.levels:
            level_to_tokenized[level] = self.level_to_tokenizer[level](
                level_to_text_list[level], **kwargs
            )

        if to_dict:
            out_dict = {"input_ids": {}, "attention_mask": {}}
            for level in level_to_tokenized:
                for k in out_dict:
                    out_dict[k][level] = level_to_tokenized[level][k]
            return out_dict
        return level_to_tokenized

    def decode(
        self, level_to_input_ids: dict[str, Any], place_zeros_in_between=False
    ) -> dict[str, list[str] | str]:
        level_to_decoded_outs = {}
        for level in level_to_input_ids:
            input_ids = level_to_input_ids[level]
            if place_zeros_in_between:
                input_ids = [add_zero_between(ids) for ids in input_ids]
            level_to_decoded_outs[level] = self.level_to_tokenizer[level].batch_decode(
                input_ids,
            )
        return level_to_decoded_outs

    def get_level_to_id_to_voab(self):
        vocab = self.get_tokenizer().vocab
        level_to_ids_to_vocab = {}
        for level in vocab:
            level_to_ids_to_vocab[level] = {v: k for k, v in vocab[level].items()}
        return level_to_ids_to_vocab

    def get_sifat_levels_to_en_name(self):
        level_to_id_to_vocab = self.get_level_to_id_to_voab()
        level_to_id_to_en_vocab = {}
        for level in level_to_id_to_vocab:
            if level == "phonemes":
                continue
            level_to_id_to_en_vocab[level] = {
                k: SIFAT_ATTR_TO_ENGLISH[v] if k != PAD_TOKEN_IDX else PAD_TOKEN
                for k, v in level_to_id_to_vocab[level].items()
            }
        return level_to_id_to_en_vocab

