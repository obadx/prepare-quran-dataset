from prepare_quran_dataset.construct.database import ReciterPool
from prepare_quran_dataset.construct.data_classes import Reciter


if __name__ == '__main__':
    reciter_pool = ReciterPool(path='data/reciters.jsonl')
    print(reciter_pool)
    print('Len Of Reciter Pool:', len(reciter_pool))

    print('\n\nAfter Insertion')
    reciter = Reciter(
        arabic_name='محمود خليل الحصري',
        english_name='Mahmoud Khalil Alhossary',
        country_code='EG',
    )
    reciter_pool.insert(reciter)
    print('Len Of Reciter Pool:', len(reciter_pool))
    print(reciter_pool)
    print(reciter_pool[0])
    hf_ds = reciter_pool.get_huggingface_dataset()
    print(hf_ds)
    reciter = reciter_pool[0]
    reciter_pool.update(reciter)

    print('\n\nAdding another reciter')
    reciter = Reciter(
        arabic_name='محمد صديق المنشاوي',
        english_name='Mohamed Sedek Almeshawy',
        country_code='EG',
    )
    reciter_pool.insert(reciter)
    print(reciter_pool)

    print('\n\nSaveing')
    reciter_pool.save()

    # print('\n\nError: editing a rectier with reciter data exists in the database')
    # reciter = reciter_pool[1]
    # reciter.arabic_name = 'محمود خليل الحصري'
    # reciter_pool.update(reciter)

    print('\n\nIterating')
    for reciter in reciter_pool:
        print(reciter)
