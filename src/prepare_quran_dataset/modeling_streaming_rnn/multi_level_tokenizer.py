from typing import get_origin, Literal, Any

from quran_transcript import SifaOutput, quran_phonetizer
from transformers import Wav2Vec2CTCTokenizer

from .vocab import (
    PAD_TOKEN,
    PAD_TOKEN_IDX,
    EOS_TOKEN,
    EOS_TOKEN_IDX,
    SIFAT_ATTR_TO_ARABIC,
    SIFAT_ATTR_TO_ENGLISH,
)


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
                model_name_or_path,
                pad_token=PAD_TOKEN,
                eos_token=EOS_TOKEN,
                target_lang=level,
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
        add_eos=False,
        **kwargs,
    ) -> dict:
        """Tokenize phonetic script and sifat attributes across all levels.

        Each level (phonemes, hams_or_jahr, shidda_or_rakhawa, etc.) has its
        own Wav2Vec2CTCTokenizer instance. All levels are tokenized
        independently with the same batch size, then collated by padding.

        Empty-phoneme sequences (``""``) produce empty text at all levels.
        The HF tokenizer converts empty text to empty ``input_ids`` and
        zero-length ``attention_mask``. When ``padding="longest"`` is passed
        via ``**kwargs``, these become all-PAD-id sequences
        (``PAD_TOKEN_IDX=0``) with ``attention_mask=0``, making them proper
        padding entries without requiring manual mask manipulation.

        Args:
            phonetic_script: One phonetic transcription string per sample.
                Pass ``""`` to create a fully-padded batch entry (all
                ``[PAD]`` ids, ``attention_mask=0`` across every level).
            sifat: Sifat attribute outputs, one ``list[SifaOutput]``
                per sample. Pass ``[]`` when the corresponding
                ``phonetic_script`` entry is ``""``. Each inner list should
                have the same length as the corresponding phoneme string.
            to_dict: If ``True``, return
                ``{"input_ids": {...}, "attention_mask": {...}}`` with
                level names as sub-keys. Otherwise return the raw HF
                tokenizer output dict.
            add_eos: If ``True``, append ``[EOS]`` token text to every
                non-empty sequence before tokenization. Empty (padding)
                sequences are left unchanged so they remain all-PAD.
            **kwargs: Forwarded to HF
                ``Wav2Vec2CTCTokenizer.__call__``. Common options include
                ``padding="longest"``, ``truncation=True``,
                ``return_tensors="pt"``.

        Returns:
            Dict mapping each level to its tokenizer output. If ``to_dict``
            is ``True``, the top-level keys are ``"input_ids"`` and
            ``"attention_mask"``, each containing a
            ``{level: ids_or_mask}`` sub-dict.
        """
        if isinstance(phonetic_script, str):
            phonetic_script = [phonetic_script]
        if not isinstance(sifat[0], list):
            sifat = [sifat]

        # Convert inner dicts to SifaOutput objects on a per-sample basis,
        # since some samples may have empty sifat lists (e.g. when phonemes is "").
        for i, inner_list in enumerate(sifat):
            if inner_list and isinstance(inner_list[0], dict):
                sifat[i] = [SifaOutput(**s) for s in inner_list]

        level_to_text_list = {}
        for level in self.levels:
            if level == "phonemes":
                text_list = phonetic_script
            else:
                # When phonemes for a sample is empty, force all sifat levels to
                # empty string too, so HF tokenizer produces all-PAD ids with
                # attention_mask=0 for that sample.
                text_list = []
                for i, ph in enumerate(phonetic_script):
                    if not ph:
                        text_list.append("")
                    else:
                        s_list = sifat[i] if i < len(sifat) and sifat[i] else []
                        text_list.append(
                            "".join(
                                [
                                    SIFAT_ATTR_TO_ARABIC[getattr(s, level)]
                                    for s in s_list
                                ]
                            )
                        )
            level_to_text_list[level] = text_list

        # Append EOS token at text level so HF tokenizer assigns the correct
        # EOS token ID natively. Skip empty sequences (padding placeholders).
        if add_eos:
            for level in self.levels:
                level_to_text_list[level] = [
                    t + EOS_TOKEN if t else t for t in level_to_text_list[level]
                ]

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
            level_to_id_to_en_vocab[level] = {}
            for k, v in level_to_id_to_vocab[level].items():
                if k == PAD_TOKEN_IDX:
                    level_to_id_to_en_vocab[level][k] = PAD_TOKEN
                elif k == EOS_TOKEN_IDX:
                    level_to_id_to_en_vocab[level][k] = EOS_TOKEN
                else:
                    level_to_id_to_en_vocab[level][k] = SIFAT_ATTR_TO_ENGLISH[v]
        return level_to_id_to_en_vocab
