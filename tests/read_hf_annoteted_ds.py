from datasets import load_dataset
import soundfile as sf

if __name__ == "__main__":
    ds_path = "../out-quran-ds"

    ds = load_dataset(ds_path, name="reciters_metadata")["train"]
    print(ds)
    print(ds.features)
    print("\n" * 4)
    print(ds.features._to_yaml_list())
    # print(ds[0])

    print("\n" * 10)
    ds = load_dataset(ds_path, name="moshaf_metadata")["train"]
    for key, val in ds[3].items():
        print(f"item[{key}] = {val}")
    print(ds)
    print(ds.features)
    print("\n" * 4)
    print(ds.features._to_yaml_list())

    ds = load_dataset(ds_path, name="moshaf_tracks", split="moshaf_0.0")
    print(ds.features)
    item = ds[50]
    for key, val in item.items():
        print(f"item[{key}] = {val}")
    sf.write(f"{item['segment_index']}.wav", item["audio"]["array"], 16000)
