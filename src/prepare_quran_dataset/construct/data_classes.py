from typing import Literal, Optional
from pydantic import BaseModel, Field


# class FooBarModel(BaseModel):
#     # do not modify attributes once object is created
#     model_config = ConfigDict(frozen=True)
#
#     id: int

class Reciter(BaseModel):
    id: int = Field(
        default=-1, description='The ID of the reciter starting of 0')
    arabic_name: str
    english_name: str
    country_code: str = Field(
        min_length=2, max_length=2,
        description='The two number ISO country code of the reciter')
    mohaf_ids: list[str] = Field(
        default_factory=lambda: [],
        description='Every Moshaf ID is a string has the following structure "reciter_id"."mohaf_id"')


class AudioFile(BaseModel):
    name: str = Field(description='Name of the file')
    path: str = ""
    sample_rate: int
    duration_minutes: float


class Moshaf(BaseModel):
    """
    Attributes:
        id (str): Every Moshaf ID is a string has the following structure "reciter_id"."mohaf_id"
        name (str): 'The arabic name of the moshaf i.e "محف محمود خليلي الحصري"'
        path (str): Absolute path to the moshaf
        reciter_id (int): The ID of the reciter starting of 0
        reciter_arabic_name (str):
        reciter_english_name (str):
        sources (list[str]): List of urls to download recitations
        num_recitations (int): Number of recitations inside the Moshaf
        total_duration_minutes (float): The total duration of Moshaf in Minutes
        is_complete (bool): If the Moshaf has all the recitations (114)
        recitation_files (list[AudioFile]): List of AudioFile objects
        publisher (Optional[str]): Publisher that records the recitations
        comments (Optional[str]): optional comments
        total_szie_mb (float): The total size in MegaBytes

    Quran Specific Attributes
        rewaya (Literal['hafs']): The rewya "روابة" of the moshaf
        madd_monfasel_len (Literal[2, 4, 5]): The length of Mad Al Monfasel "مد النفصل"
        madd_motasel_len (Literal[4, 5]): The length of Mad Al Motasel "مد المتصل"
        madd_aared_len (Literal[2, 4, 6]): The length of Mad Al Aared "مد العارض للسكون"
        madd_motasel_mahmooz_aared_len (Literal[4, 5, 6]): The length of Madd Almotasel Al Mahmooz during waqf "مد المتصل المهموز عند الوقف". Example "السماء"
        madd_ayn_lazem_len (Literal[4, 6]): The length of Lzem Harfy Madd "المد الحرفي اللازم لحرف العين" in surar: Maryam "مريم", AlShura "الشورى" either 4 or 6
        tasheel_or_madd (bool): Tasheel of Madd "وجع التسهيل أو المد" for 6 words in The Holy Quran: "ءالذكرين", "ءالله", "ءائن"
        daaf_harka (Literal['fath', 'dam']): The Haraka of "ضعف" in surah AlRoom "الروم" aya(54)
        idghaam_nkhlqkm (Literal['kamel', 'nakes']): The Idghaof of word "نخلقكم" in suran Almurslat "المرسلات" Aya (20) Either Idgham nakes "إدغام نافص" or Idghtam kamel "إدغام كامل"
        noon_tammna (Literal['ishmam', 'ikhtlas']): Warys to recite word tammna "تأمنا" in surah Yusuf Aya(11) "سورة يوسف". Eeither Ishmam "إشمام" or Ikhtlas "اختلاس"

    """
    id: str = Field(
        default="",
        description='Every Moshaf ID is a string has the following structure "reciter_id"."mohaf_id"')
    name: str = Field(
        description='The arabic name of the moshaf i.e "محف محمود خليلي الحصري"')
    path: str = Field(description='Absolute path to the moshaf')
    reciter_id: int = Field(description='The ID of the reciter starting of 0')
    reciter_arabic_name: str
    reciter_english_name: str
    sources: list[str] = Field(
        description=f'List of urls to download recitations')
    num_recitations: int = Field(
        description='Number of recitations inside the Moshaf')
    total_duraion_minutes: float = Field(
        description='The total duration of Moshaf in Minutes')
    is_complete: bool = Field(
        description='If the Moshaf has all the recitations (114)')
    recitation_files: list[AudioFile] = Field(
        description='List of AudioFile objects')
    publisher: Optional[str] = Field(
        default='',
        description='Publisher that records the recitations')
    comments: Optional[str] = Field(
        default='',
        description='optional comments')
    total_size_mb: float = Field(
        description='The total size in MegaBytes')

    # Quran Specific Attributes
    rewaya: Literal['hafs']
    madd_monfasel_len: Literal[2, 4, 5] = Field(
        description='The length of Mad Al Monfasel "مد النفصل"')
    madd_motasel_len: Literal[4, 5] = Field(
        description='The length of Mad Al Motasel "مد المتصل"')
    madd_aared_len: Literal[2, 4, 6] = Field(
        description='The length of Mad Al Aared "مد العارض للسكون"')
    madd_motasel_mahmooz_aared_len: Literal[4, 5, 6] = Field(
        description='The length of Madd Almotasel Al Mahmooz during waqf "مد المتصل المهموز عند الوقف". Example "السماء"')
    madd_alayn_lazem_len: Literal[4, 6] = Field(
        description='The length of Lzem Harfy Madd "المد الحرفي اللازم لحرف العين" in surar: Maryam "مريم", AlShura "الشورى" either 4 or 6')
    tasheel_or_madd: bool = Field(
        description='Tasheel of Madd "وجع التسهيل أو المد" for 6 words in The Holy Quran: "ءالذكرين", "ءالله", "ءائن"')
    daaf_harka: Literal['fath', 'dam'] = Field(
        description='The Haraka of "ضعف" in surah AlRoom "الروم" aya(54)')
    idghaam_nkhlqkm: Literal['kamel', 'nakes'] = Field(
        description='The Idghaof of word "نخلقكم" in suran Almurslat "المرسلات" Aya (20) Either Idgham nakes "إدغام نافص" or Idghtam kamel "إدغام كامل"')
    noon_tammna: Literal['ishmam', 'ikhtlas'] = Field(
        description='Warys to recite word tammna "تأمنا" in surah Yusuf Aya(11) "سورة يوسف". Eeither Ishmam "إشمام" or Ikhtlas "اختلاس"')

    def model_post_init(self, __context):
        pass
