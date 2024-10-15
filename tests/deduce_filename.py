from prepare_quran_dataset.construct.utils import deduce_filename

if __name__ == '__main__':
    urls = []
    urls.append('https://cdns1.zekr.online/quran/5210/111/64.mp3')
    urls.append('https://zekr.online/ar/single/Download/Sura/263038?title=%D8%A7%D9%84%D9%86%D8%A7%D8%B3%20-%20%D8%A3%D8%AD%D9%85%D8%AF%20%D9%85%D8%AD%D9%85%D8%AF%20%D8%B9%D8%A7%D9%85%D8%B1')
    urls.append(
        'https://codeload.github.com/Abdullahaml1/prepare-quran-dataset/zip/refs/heads/main')
    urls.append('https://download.quran.islamway.net/archives/212-hi.zip')

    for url in urls:
        print('url', url)
        print(deduce_filename(url))
        print('\n' * 3)
