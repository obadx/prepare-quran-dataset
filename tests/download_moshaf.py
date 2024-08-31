from pathlib import Path

from prepare_quran_dataset.construct.database import (
    get_files,
    download_moshaf_from_urls,
    download_media_and_fill_metadata,
    Reciter,
    Moshaf,
)


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # Test get_files
    # ----------------------------------------------------------------------
    # pathes = [Path('data/'), Path('src')]
    # files = get_files(pathes)
    # for file in files:
    #     print(file)
    # print('Len of files', len(files))

    # ----------------------------------------------------------------------
    # Test download_moshaf
    # ----------------------------------------------------------------------
    # urls = [
    #     'https://storage.googleapis.com/drive-bulk-export-anonymous/20240803T123812.801Z/4133399871716478688/647c412c-3306-49d5-bc8a-b8ab504391d6/1/975a2587-26d0-4c35-87c7-450e6b001b14?authuser',
    #
    #     # 'https://download.quran.islamway.net/quran3/696/001.mp3',
    #     'https://download.quran.islamway.net/quran3/696/110.mp3',
    #     'https://download.quran.islamway.net/quran3/696/111.mp3',
    #     'https://download.quran.islamway.net/quran3/696/112.mp3',
    # ]
    # download_moshaf_from_urls(
    #     urls=urls,
    #     moshaf_path='data/mohaf_123',
    #     moshaf_name='moshaf_123',
    #     download_path='data/downloads',
    #     remove_after_download=True,
    # )

    # ----------------------------------------------------------------------
    # Test download_moshaf_and_fill_metadata
    # ----------------------------------------------------------------------
    urls = [
        'https://storage.googleapis.com/drive-bulk-export-anonymous/20240831T161147.511Z/4133399871716478688/0fca15a3-ad5c-471f-b26c-f6c98d4326c9/1/c6fcdd31-606a-40a2-9329-1bcef883c496?authuser',

        # 'https://download.quran.islamway.net/quran3/696/001.mp3',
        'https://download.quran.islamway.net/quran3/696/090.mp3',
        'https://download.quran.islamway.net/quran3/696/091.mp3',
        'https://download.quran.islamway.net/quran3/696/093.mp3',
    ]

    specific = {
        '114': 'https://download.quran.islamway.net/quran3/696/001.mp3',
        '113': 'https://download.quran.islamway.net/quran3/696/095.mp3'
    }

    reciter = Reciter(
        id=0,
        arabic_name='الحصري',
        english_name='Ahossary',
        country_code='EG',
    )

    moshaf = Moshaf(
        id='0.0',
        name='المصحف المرتل',
        reciter_id=0,
        reciter_arabic_name=reciter.arabic_name,
        reciter_english_name=reciter.english_name,
        sources=urls,
        specific_sources=specific,
        rewaya='hafs',
        madd_monfasel_len=2,
        madd_mottasel_len=4,
        madd_alayn_lazem_len=6,
        madd_aared_len=2,
        madd_mottasel_mahmooz_aared_len=4,
        tasheel_or_madd='tasheel',
        daaf_harka='fath',
        idghaam_nakhlqkm='kamel',
        noon_tamanna='ishmam',
    )

    updated_moshaf = download_media_and_fill_metadata(
        moshaf, database_path='data/dataset', download_path='data/downloads')
    print(updated_moshaf.total_duraion_minutes)
    moshaf.model_validate(moshaf)
    for key, value in updated_moshaf.dict().items():
        print(f'{key}: {value}')

    print('Json Validation')
    # TODO: validate json
    # moshaf.model_validate_json(moshaf)
