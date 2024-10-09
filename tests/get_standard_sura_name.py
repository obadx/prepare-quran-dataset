from prepare_quran_dataset.construct.utils import get_suar_list
from prepare_quran_dataset.construct.database import get_sura_standard_name


if __name__ == '__main__':

    print(get_sura_standard_name(' الفاتحة001.mp3'))
    print(get_sura_standard_name('الفاتحة' + '.mp3'))
    print(get_sura_standard_name('الأخلاص' + '.mp3'))
    print(get_sura_standard_name('المائدة  أحمد' + '.mp3'))
    print(get_sura_standard_name('المائدة  004' + '.mp3'))
    print(get_sura_standard_name('المائدة  4' + '.mp3'))
