import shutil

from prepare_quran_dataset.construct.utils import extract_zipfile

if __name__ == '__main__':
    # big zipfile
    extract_path = 'data/hafs'
    extract_zipfile(zipfile_path='data/212-hi.zip',
                    extract_path=extract_path, num_workers=12)

    # # empy zipfil2
    # extract_path = 'data/out_test'
    # extract_zipfile(zipfile_path='data/test.zip',
    #                 extract_path=extract_path, num_workers=12)

    # shutil.rmtree(extract_path)
