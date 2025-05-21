from prepare_quran_dataset.construct.data_classes import Moshaf, Reciter


if __name__ == '__main__':
    execluded_fields = set([
        'downloaded_sources',
        'recitation_files',
    ])

    features, metadata = Reciter.extract_huggingface_features()
    print(features)
    print(metadata)

    print('-' * 50)

    features, metadata = Moshaf.extract_huggingface_features(execluded_fields)
    print(features)
    print(metadata)
