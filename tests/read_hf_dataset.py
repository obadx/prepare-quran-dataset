from datasets import load_dataset

if __name__ == '__main__':
    ds_path = '../quran-dataset'

    ds = load_dataset(ds_path, name='reciters_metadata')['train']
    print(ds)
    print(ds.features)
    print('\n' * 4)
    print(ds.features._to_yaml_list())
    # print(ds[0])

    print('\n' * 10)
    ds = load_dataset(ds_path, name='moshaf_metadata')['train']
    for key, val in ds[3].items():
        print(f'item[{key}] = {val}')
    print(ds)
    print(ds.features)
    print('\n' * 4)
    print(ds.features._to_yaml_list())

    print('\n' * 10)
    ds = load_dataset(
        'audiofolder', data_dir='../quran-dataset/dataset')['train']
    for key, val in ds[0].items():
        print(f'item[{key}] = {val}')
    print(ds)
    print(ds.features)
    print('\n' * 4)
    print(ds.features._to_yaml_list())
