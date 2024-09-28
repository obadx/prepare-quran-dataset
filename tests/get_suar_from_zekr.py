import json

from prepare_quran_dataset.construct.utils import extract_sura_from_zekr


if __name__ == '__main__':
    # Example usage
    url = "https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=5403"
    # url = "https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=813"
    result = extract_sura_from_zekr(url)
    print(json.dumps(result, indent=4, ensure_ascii=False))
