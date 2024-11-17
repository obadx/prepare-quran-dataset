import json
from prepare_quran_dataset.construct.utils import extract_suar_from_archive


if __name__ == '__main__':
    url = 'https://archive.org/details/Musshaf-Mujawwed-Kamel-Yousef-Al-Buthimi-High-Quality/'
    # url = 'https://archive.org/details/Al-Husary'

    out = extract_suar_from_archive(url)
    print(json.dumps(out, indent=4))
