from pathlib import Path
import json

from datasets import load_dataset
from quran_transcript import quran_phonetizer, MoshafAttributes

from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer


if __name__ == "__main__":
    ds_path = Path("/cluster/users/shams035u1/data/mualem-recitations-annotated")
    moshaf_ds = load_dataset(str(ds_path), name="moshaf_metadata", split="train")
    multi_level_tokenizer = MultiLevelTokenizer("./")

    moshaf_to_dup = {}
    for moshaf in moshaf_ds:
        ds = load_dataset(
            str(ds_path), name=f"moshaf_{moshaf['id']}", split="train", num_proc=32
        )
        moshaf_attr = MoshafAttributes(**moshaf)
        uth_txts = ds["uthmani"]
        seg_ids = ds["segment_index"]
        batch = 16
        for idx in range(0, len(seg_ids), batch):
            try:
                texts = uth_txts[idx:batch]
                photenized_outs = [
                    quran_phonetizer(
                        texts[idx],
                        moshaf_attr,
                        remove_spaces=True,
                    )
                    for idx in range(len(texts))
                ]

                labels = multi_level_tokenizer.tokenize(
                    [p.phonemes for p in photenized_outs],
                    [p.sifat for p in photenized_outs],
                    to_dict=True,
                    return_tensors="pt",
                    padding="longest",
                )
            except Exception as e:
                print(f"Moshaf: {moshaf['id']}")
                print(f"Erro with ids: {seg_ids[idx:batch]}")
                for ph_out, txt, seg_idx in zip(
                    photenized_outs, texts, seg_ids[idx:batch]
                ):
                    print(
                        f"{seg_idx} -> {txt}\nPhonemes: {ph_out.phonemes}\nSifat:\n{json.dumps(ph_out.sifat, ensure_ascii=False, indent=2)}"
                    )
                    print("-" * 40)
                raise e
