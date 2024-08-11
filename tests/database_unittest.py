import unittest
import tempfile
import shutil
from pathlib import Path

from prepare_quran_dataset.construct.database import ReciterPool, MoshafPool
from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf
from prepare_quran_dataset.construct.base import ItemExistsInPoolError


def same_reciter(
        test_obj: unittest.TestCase,
        reciter_x: Reciter,
        reciter_y: Reciter
) -> None:
    """Helper Method"""
    attributes = ['arabic_name', 'english_name', 'country_code']
    for attr in attributes:
        test_obj.assertEqual(
            getattr(reciter_x, attr),
            getattr(reciter_y, attr)
        )


class TestReciterPool(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = Path(self.temp_dir) / "reciter_pool.jsonl"
        self.temp_file.touch()
        self.reciter_pool = ReciterPool(path=str(self.temp_file))

        self.reciters = [Reciter(
            arabic_name="محمد صديق المنشاوي",
            english_name="Mohammad Siddiq Al-Minshawi",
            country_code="EG"),
            Reciter(
            arabic_name="محمود خليل الحصري",
            english_name="Mahmoud Khalil Alhossary",
            country_code="EG")
        ]

    def tearDown(self):
        """This function is executed at the end of the code"""
        # delete the temdir
        shutil.rmtree(self.temp_dir)

    def test_insert_reciter(self):
        for idx, reciter in enumerate(self.reciters):
            self.reciter_pool.insert(reciter)
            self.assertEqual(len(self.reciter_pool), idx + 1)

    def test_get_reciter(self):

        self.test_insert_reciter()
        for id in range(len(self.reciters)):
            same_reciter(self, self.reciter_pool[id], self.reciters[id])

    def test_iterating(self):

        self.test_insert_reciter()
        for reciter in self.reciter_pool:
            same_reciter(self, reciter, self.reciters[reciter.id])

    def test_saveing_and_loading(self):
        self.test_insert_reciter()
        self.reciter_pool.save()

        # loading from the saved file
        reciter_pool = ReciterPool(self.temp_file)
        self.assertEqual(len(reciter_pool), len(self.reciter_pool))
        self.assertEqual(len(reciter_pool), len(self.reciters))

        for reciter in reciter_pool:
            same_reciter(self, reciter, self.reciters[reciter.id])

    def test_update(self):
        self.test_insert_reciter()
        reciter = self.reciter_pool[0]
        reciter.country_code = 'SA'
        self.reciter_pool.update(reciter)

        # loading from the database
        self.reciter_pool.save()
        loaded_reciter_pool = ReciterPool(self.temp_file)
        self.assertEqual(
            reciter.country_code,
            loaded_reciter_pool[0].country_code)

    def test_update_reciter_attribute_that_affects_hash(self,):
        self.test_insert_reciter()
        reciter = self.reciter_pool[0]
        reciter.arabic_name = 'حماده'
        self.reciter_pool.update(reciter)

        same_reciter(self, reciter, self.reciter_pool[0])

    def test_update_reciter_with_name_of_existing_reciter(self):
        self.test_insert_reciter()
        reciter = self.reciter_pool[0]
        reciter.arabic_name = self.reciter_pool[1].arabic_name

        with self.assertRaises(ItemExistsInPoolError):
            self.reciter_pool.update(reciter)

    def test_delete_reciter(self):
        self.test_insert_reciter()
        self.reciter_pool.delete(0)
        same_reciter(self, self.reciter_pool[1], self.reciters[1])
        self.assertEqual(len(self.reciter_pool), 1)

        with self.assertRaises(KeyError):
            self.reciter_pool[0]

        self.reciter_pool.insert(self.reciters[0])
        same_reciter(self, self.reciter_pool[2], self.reciters[0])
        self.assertEqual(len(self.reciter_pool), 2)


def same_moshaf(
    test_obj: unittest.TestCase,
    moshaf_x: Moshaf,
    moshaf_y: Moshaf
) -> None:
    for (name_x, field_x), (name_y, field_y) in zip(moshaf_x.model_fields.items(), moshaf_y.model_fields.items()):
        if name_x == name_y == 'id':
            continue
        test_obj.assertEqual(field_x, field_y)


class TestMoshafPool(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.reciter_pool_file = Path(self.temp_dir) / "reciter_pool.jsonl"
        self.moshaf_pool_file = Path(self.temp_dir) / "moshaf_pool.jsonl"

        self.reciter_pool_file.touch()
        self.moshaf_pool_file.touch()

        self.reciter_pool = ReciterPool(path=self.reciter_pool_file)
        self.moshaf_pool = MoshafPool(
            reciter_pool=self.reciter_pool,
            metadata_path=self.moshaf_pool_file)

        reciters = [Reciter(
            arabic_name="محمد صديق المنشاوي",
            english_name="Mohammad Siddiq Al-Minshawi",
            country_code="EG"),
            Reciter(
            arabic_name="محمود خليل الحصري",
            english_name="Mahmoud Khalil Alhossary",
            country_code="EG"),
            Reciter(
            arabic_name="عبد الباسط عبد الصمد",
            english_name="Abdelbasset Abdelsamad",
            country_code="EG"),

        ]
        for reciter in reciters:
            self.reciter_pool.insert(reciter)

        self.moshaf_dict = {
            '0.0':
            Moshaf(
                name='المصحف المرتل',
                reciter_id=0,
                reciter_arabic_name=self.reciter_pool[0].arabic_name,
                reciter_english_name=self.reciter_pool[0].english_name,
                sources=[
                    'https://download.quran.islamway.net/archives/212-hi.zip'],
                rewaya='hafs',
                madd_monfasel_len=2,
                madd_mottasel_len=4,
                madd_alayn_lazem_len=6,
                madd_aared_len=2,
                madd_mottasel_mahmooz_aared_len=4,
                tasheel_or_madd='tasheel',
                daaf_harka='fath',
                idghaam_nkhlqkm='kamel',
                noon_tamnna='ishmam',
            ),
                '0.1':
            Moshaf(
                name='المصحف المعلم',
                reciter_id=0,
                reciter_arabic_name=self.reciter_pool[0].arabic_name,
                reciter_english_name=self.reciter_pool[0].english_name,
                sources=[
                    'https://download.quran.islamway.net/archives/211-hi.zip'],
                rewaya='hafs',
                madd_monfasel_len=2,
                madd_mottasel_len=4,
                madd_alayn_lazem_len=6,
                madd_aared_len=2,
                madd_mottasel_mahmooz_aared_len=4,
                tasheel_or_madd='tasheel',
                daaf_harka='fath',
                idghaam_nkhlqkm='kamel',
                noon_tamnna='ishmam',
            ),
            '1.0':
            Moshaf(
                name='المصحف المرتل',
                reciter_id=1,
                reciter_arabic_name=self.reciter_pool[1].arabic_name,
                reciter_english_name=self.reciter_pool[1].english_name,
                sources=[
                    'https://download.quran.islamway.net/archives/14642-hi.zip'],
                rewaya='hafs',
                madd_monfasel_len=2,
                madd_mottasel_len=4,
                madd_alayn_lazem_len=6,
                madd_aared_len=2,
                madd_mottasel_mahmooz_aared_len=4,
                tasheel_or_madd='tasheel',
                daaf_harka='fath',
                idghaam_nkhlqkm='kamel',
                noon_tamnna='ishmam',
            ),
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_insert_moshaf(self):
        for idx, moshaf in enumerate(self.moshaf_dict.values()):
            self.moshaf_pool.insert(moshaf)
            self.assertEqual(len(self.moshaf_pool), idx + 1)

        self.assertEqual(len(self.reciter_pool[0].moshaf_set_ids), 2)
        self.assertEqual(len(self.reciter_pool[1].moshaf_set_ids), 1)
        self.assertEqual(len(self.reciter_pool[2].moshaf_set_ids), 0)
    #

    def test_iteration(self):
        self.test_insert_moshaf()
        moshaf_ids = [moshaf.id for moshaf in self.moshaf_pool]
        self.assertEqual(set(moshaf_ids), set(self.moshaf_dict.keys()))

    def test_get_moshaf(self):
        self.test_insert_moshaf()
        for id, moshaf in self.moshaf_dict.items():
            same_moshaf(self, moshaf, self.moshaf_pool[id])

    def test_get_moshaf_from_reciter(self):
        self.test_insert_moshaf()

        for reciter in self.reciter_pool:
            for moshaf_id in reciter.moshaf_set_ids:
                moshaf = self.moshaf_pool[moshaf_id]
                self.assertEqual(reciter.id, moshaf.reciter_id)
                self.assertEqual(reciter.arabic_name,
                                 moshaf.reciter_arabic_name)

    def test_editing_output_object_does_not_affect_database_object(self):
        self.test_insert_moshaf()
        moshaf = self.moshaf_pool['1.0']
        moshaf.name = 'tajweed'
        pool_moshaf = self.moshaf_pool['1.0']
        self.assertNotEqual(pool_moshaf.name, moshaf.name)

    def test_update_non_hash_attributes(self):
        self.test_insert_moshaf()
        moshaf = self.moshaf_pool['1.0']
        moshaf.name = 'tajweed'
        self.moshaf_pool.update(moshaf)
        self.assertEqual(moshaf.name, self.moshaf_pool['1.0'].name)
        self.moshaf_pool.save()

        loaded_moshaf_pool = MoshafPool(
            reciter_pool=self.reciter_pool,
            metadata_path=self.moshaf_pool_file)
        self.assertEqual(moshaf.name, loaded_moshaf_pool['1.0'].name)

    def test_update_with_changing_moshaf_reciter(self):
        self.test_insert_moshaf()
        moshaf = self.moshaf_pool['1.0']
        moshaf.reciter_id = 0
        self.moshaf_pool.update(moshaf)
        self.assertEqual(len(self.reciter_pool[0].moshaf_set_ids), 3)
        self.assertEqual(len(self.reciter_pool[1].moshaf_set_ids), 0)
        self.assertEqual(len(self.reciter_pool[2].moshaf_set_ids), 0)

        moshaf = self.moshaf_pool['0.2']
        self.assertEqual(
            self.reciter_pool[0].arabic_name, moshaf.reciter_arabic_name)
        self.assertEqual(
            self.reciter_pool[0].english_name, moshaf.reciter_english_name)

    def test_delete_moshaf(self):
        """deleting item and inserting it for reciter "1" """
        self.test_insert_moshaf()

        deleted_item = self.moshaf_pool.delete('0.0')
        self.assertEqual(len(self.moshaf_pool), 2)
        self.assertEqual(len(self.reciter_pool[0].moshaf_set_ids), 1)
        self.assertNotIn('0.0', self.reciter_pool[0].moshaf_set_ids)
        with self.assertRaises(KeyError):
            self.moshaf_pool['0.0']

        deleted_item.reciter_id = 1
        self.moshaf_pool.insert(deleted_item)
        self.assertEqual(len(self.moshaf_pool), 3)
        self.assertEqual(len(self.reciter_pool[1].moshaf_set_ids), 2)
        self.assertEqual(
            self.reciter_pool[1].arabic_name,
            self.moshaf_pool['1.1'].reciter_arabic_name)
        self.assertEqual(
            self.reciter_pool[1].english_name,
            self.moshaf_pool['1.1'].reciter_english_name)


if __name__ == '__main__':
    unittest.main()
