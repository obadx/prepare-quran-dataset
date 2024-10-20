import json

from prepare_quran_dataset.construct.utils import extract_suar_from_mp3quran


if __name__ == '__main__':
    # Example usage
    url = "https://mp3quran.net/ar/Aamer"
    # url = "https://zekr.online/ar/author/61/mhmod-khll-lhsr/quran?mushaf_id=813"
    result = extract_suar_from_mp3quran(url)
    print(json.dumps(result, indent=4, ensure_ascii=False))
