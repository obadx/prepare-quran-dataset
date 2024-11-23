import tempfile
from pathlib import Path

from prepare_quran_dataset.construct.utils import (
    download_file_fast,
    download_multi_file_fast,
    timer,
)


@timer
def download_sequential(sura_to_url: dict[str, str], out_path: Path, segments=20):
    out_path = Path(out_path)
    print(f'Segments: {segments}')
    for url in sura_to_url.values():
        download_file_fast(
            url,
            out_path,
            num_download_segments=segments,
            redownload=True,
            show_progress=False,
        )


@timer
def download_concurrent(sura_to_url: dict[str, str], out_path: Path, segments=20):
    out_path = Path(out_path)
    print(f'Segments: {segments}')
    download_multi_file_fast(
        list(sura_to_url.values()),
        [out_path] * len(sura_to_url),
        max_dl_workers=10,
        num_download_segments=segments,
        redownload=True,
        show_progress=False,
    )


if __name__ == '__main__':
    sura_to_url = {
        "001": "https://cdns1.zekr.online/quran/10341/1/128.mp3",
        "002": "https://cdns1.zekr.online/quran/10341/2/128.mp3",
        "003": "https://cdns1.zekr.online/quran/10341/3/128.mp3",
        "004": "https://cdns1.zekr.online/quran/10341/4/128.mp3",
        "005": "https://cdns1.zekr.online/quran/10341/5/128.mp3",
        "006": "https://cdns1.zekr.online/quran/10341/6/128.mp3",
        "007": "https://cdns1.zekr.online/quran/10341/7/128.mp3",
        "008": "https://cdns1.zekr.online/quran/10341/8/128.mp3",
        "009": "https://cdns1.zekr.online/quran/10341/9/128.mp3",
        "010": "https://cdns1.zekr.online/quran/10341/10/128.mp3",
        "011": "https://cdns1.zekr.online/quran/10341/11/128.mp3",
        "012": "https://cdns1.zekr.online/quran/10341/12/128.mp3",
        "013": "https://cdns1.zekr.online/quran/10341/13/128.mp3",
        "014": "https://cdns1.zekr.online/quran/10341/14/128.mp3",
        "015": "https://cdns1.zekr.online/quran/10341/15/128.mp3",
        "016": "https://cdns1.zekr.online/quran/10341/16/128.mp3",
        "017": "https://cdns1.zekr.online/quran/10341/17/128.mp3",
        "018": "https://cdns1.zekr.online/quran/10341/18/128.mp3",
        "019": "https://cdns1.zekr.online/quran/10341/19/128.mp3",
        "020": "https://cdns1.zekr.online/quran/10341/20/128.mp3",
        "021": "https://cdns1.zekr.online/quran/10341/21/128.mp3",
        "022": "https://cdns1.zekr.online/quran/10341/22/128.mp3",
        "023": "https://cdns1.zekr.online/quran/10341/23/128.mp3",
        "024": "https://cdns1.zekr.online/quran/10341/24/128.mp3",
        "025": "https://cdns1.zekr.online/quran/10341/25/128.mp3",
        "026": "https://cdns1.zekr.online/quran/10341/26/128.mp3",
        "027": "https://cdns1.zekr.online/quran/10341/27/128.mp3",
        "028": "https://cdns1.zekr.online/quran/10341/28/128.mp3",
        "029": "https://cdns1.zekr.online/quran/10341/29/128.mp3",
        "030": "https://cdns1.zekr.online/quran/10341/30/128.mp3",
        "031": "https://cdns1.zekr.online/quran/10341/31/128.mp3",
        "032": "https://cdns1.zekr.online/quran/10341/32/128.mp3",
        "033": "https://cdns1.zekr.online/quran/10341/33/128.mp3",
        "034": "https://cdns1.zekr.online/quran/10341/34/128.mp3",
        "035": "https://cdns1.zekr.online/quran/10341/35/128.mp3",
        "036": "https://cdns1.zekr.online/quran/10341/36/128.mp3",
        "037": "https://cdns1.zekr.online/quran/10341/37/128.mp3",
        "038": "https://cdns1.zekr.online/quran/10341/38/128.mp3",
        "039": "https://cdns1.zekr.online/quran/10341/39/128.mp3",
        "040": "https://cdns1.zekr.online/quran/10341/40/128.mp3",
        "041": "https://cdns1.zekr.online/quran/10341/41/128.mp3",
        "042": "https://cdns1.zekr.online/quran/10341/42/128.mp3",
        "043": "https://cdns1.zekr.online/quran/10341/43/128.mp3",
        "044": "https://cdns1.zekr.online/quran/10341/44/128.mp3",
        "045": "https://cdns1.zekr.online/quran/10341/45/128.mp3",
        "046": "https://cdns1.zekr.online/quran/10341/46/128.mp3",
        "047": "https://cdns1.zekr.online/quran/10341/47/128.mp3",
        "048": "https://cdns1.zekr.online/quran/10341/48/128.mp3",
        "049": "https://cdns1.zekr.online/quran/10341/49/128.mp3",
        "050": "https://cdns1.zekr.online/quran/10341/50/128.mp3",
        "051": "https://cdns1.zekr.online/quran/10341/51/128.mp3",
        "052": "https://cdns1.zekr.online/quran/10341/52/128.mp3",
        "053": "https://cdns1.zekr.online/quran/10341/53/128.mp3",
        "054": "https://cdns1.zekr.online/quran/10341/54/128.mp3",
        "055": "https://cdns1.zekr.online/quran/10341/55/128.mp3",
        "056": "https://cdns1.zekr.online/quran/10341/56/128.mp3",
        "057": "https://cdns1.zekr.online/quran/10341/57/128.mp3",
        "058": "https://cdns1.zekr.online/quran/10341/58/128.mp3",
        "059": "https://cdns1.zekr.online/quran/10341/59/128.mp3",
        "060": "https://cdns1.zekr.online/quran/10341/60/128.mp3",
        "061": "https://cdns1.zekr.online/quran/10341/61/128.mp3",
        "062": "https://cdns1.zekr.online/quran/10341/62/128.mp3",
        "063": "https://cdns1.zekr.online/quran/10341/63/128.mp3",
        "064": "https://cdns1.zekr.online/quran/10341/64/128.mp3",
        "065": "https://cdns1.zekr.online/quran/10341/65/128.mp3",
        "066": "https://cdns1.zekr.online/quran/10341/66/128.mp3",
        "067": "https://cdns1.zekr.online/quran/10341/67/128.mp3",
        "068": "https://cdns1.zekr.online/quran/10341/68/128.mp3",
        "069": "https://cdns1.zekr.online/quran/10341/69/128.mp3",
        "070": "https://cdns1.zekr.online/quran/10341/70/128.mp3",
        "071": "https://cdns1.zekr.online/quran/10341/71/127.mp3",
        "072": "https://cdns1.zekr.online/quran/10341/72/128.mp3",
        "073": "https://cdns1.zekr.online/quran/10341/73/128.mp3",
        "074": "https://cdns1.zekr.online/quran/10341/74/128.mp3",
        "075": "https://cdns1.zekr.online/quran/10341/75/127.mp3",
        "076": "https://cdns1.zekr.online/quran/10341/76/128.mp3",
        "077": "https://cdns1.zekr.online/quran/10341/77/128.mp3",
        "078": "https://cdns1.zekr.online/quran/10341/78/128.mp3",
        "079": "https://cdns1.zekr.online/quran/10341/79/128.mp3",
        "080": "https://cdns1.zekr.online/quran/10341/80/128.mp3",
        "081": "https://cdns1.zekr.online/quran/10341/81/127.mp3",
        "082": "https://cdns1.zekr.online/quran/10341/82/128.mp3",
        "083": "https://cdns1.zekr.online/quran/10341/83/128.mp3",
        "084": "https://cdns1.zekr.online/quran/10341/84/128.mp3",
        "085": "https://cdns1.zekr.online/quran/10341/85/128.mp3",
        "086": "https://cdns1.zekr.online/quran/10341/86/128.mp3",
        "087": "https://cdns1.zekr.online/quran/10341/87/128.mp3",
        "088": "https://cdns1.zekr.online/quran/10341/88/128.mp3",
        "089": "https://cdns1.zekr.online/quran/10341/89/128.mp3",
        "090": "https://cdns1.zekr.online/quran/10341/90/128.mp3",
        "091": "https://cdns1.zekr.online/quran/10341/91/128.mp3",
        "092": "https://cdns1.zekr.online/quran/10341/92/128.mp3",
        "093": "https://cdns1.zekr.online/quran/10341/93/128.mp3",
        "094": "https://cdns1.zekr.online/quran/10341/94/127.mp3",
        "095": "https://cdns1.zekr.online/quran/10341/95/128.mp3",
        "096": "https://cdns1.zekr.online/quran/10341/96/128.mp3",
        "097": "https://cdns1.zekr.online/quran/10341/97/128.mp3",
        "098": "https://cdns1.zekr.online/quran/10341/98/128.mp3",
        "099": "https://cdns1.zekr.online/quran/10341/99/128.mp3",
        "100": "https://cdns1.zekr.online/quran/10341/100/127.mp3",
        "101": "https://cdns1.zekr.online/quran/10341/101/128.mp3",
        "102": "https://cdns1.zekr.online/quran/10341/102/127.mp3",
        "103": "https://cdns1.zekr.online/quran/10341/103/128.mp3",
        "104": "https://cdns1.zekr.online/quran/10341/104/128.mp3",
        "105": "https://cdns1.zekr.online/quran/10341/105/128.mp3",
        "106": "https://cdns1.zekr.online/quran/10341/106/128.mp3",
        "107": "https://cdns1.zekr.online/quran/10341/107/128.mp3",
        "108": "https://cdns1.zekr.online/quran/10341/108/128.mp3",
        "109": "https://cdns1.zekr.online/quran/10341/109/128.mp3",
        "110": "https://cdns1.zekr.online/quran/10341/110/128.mp3",
        "111": "https://cdns1.zekr.online/quran/10341/111/128.mp3",
        "112": "https://cdns1.zekr.online/quran/10341/112/128.mp3",
        "113": "https://cdns1.zekr.online/quran/10341/113/128.mp3",
        "114": "https://cdns1.zekr.online/quran/10341/114/128.mp3"
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        download_sequential(sura_to_url, temp_dir, segments=10)
        download_sequential(sura_to_url, temp_dir, segments=20)
        download_concurrent(sura_to_url, temp_dir, segments=10)
        download_concurrent(sura_to_url, temp_dir, segments=20)
