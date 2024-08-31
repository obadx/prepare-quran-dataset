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
    arabic_name: str = Field(description='ArabicName(الاسم)')
    english_name: str = Field(
        description='ArabicName(الاسم باللغة الإنجليزية)')
    country_code: str = Field(
        min_length=2, max_length=2,
        description='ArabicName(رمز الدولة بالإنجليزية) The two number ISO country code of the reciter')
    moshaf_set_ids: set[str] = Field(
        default_factory=lambda: set(),
        description='Every Moshaf ID is a string has the following structure "reciter_id"."mohaf_id"')


class AudioFile(BaseModel):
    name: str = Field(description='Name of the file')
    path: str = ""
    sample_rate: int
    duration_minutes: float


# TODO: Add: arabice translation for the attributes
class Moshaf(BaseModel):
    id: str = Field(
        default="",
        description='Every Moshaf ID is a string has the following structure "reciter_id"."mohaf_id"')
    name: str = Field(
        description='The arabic name of the moshaf i.e "محف محمود خليلي الحصري"')
    path: str = Field(default="", description='Absolute path to the moshaf')
    reciter_id: int = Field(description='The ID of the reciter starting of 0')
    reciter_arabic_name: str = ""
    reciter_english_name: str = ""
    sources: list[str] = Field(
        description='List of urls to download recitations')
    specific_sources: dict[str, str] = Field(
        default={},
        description='Overwriting a specific group of files.'
        'Ex: {"002": "url_for_002"} will overwrite the recitation "002"'
        'downloaded by the `sources` attributes'
    )
    is_sura_parted: bool = Field(
        default=True,
        description='If every recitation file is a sperate sura or not')

    # Metadata Fields
    num_recitations: int = Field(
        default=0,
        description='Number of recitations inside the Moshaf')
    total_duraion_minutes: float = Field(
        default=0.0,
        description='The total duration of Moshaf in Minutes')
    is_complete: bool = Field(
        default=False,
        description='If the Moshaf has all the recitations (114)')
    recitation_files: list[AudioFile] = Field(
        default_factory=lambda: [],
        description='List of AudioFile objects')
    publisher: Optional[str] = Field(
        default='',
        description='Publisher that records the recitations')
    comments: Optional[str] = Field(
        default='',
        description='optional comments')
    total_size_mb: float = Field(
        default=0.0,
        description='The total size in MegaBytes')
    is_downloaded: bool = Field(
        default=False,
        description='moshaf is downloades using the sources or not')

    # Quran Specific Attributes
    rewaya: Literal['hafs']
    madd_monfasel_len: Literal[2, 4, 5] = Field(
        description='ArabicName(مقدرا المد المنفصل) The length of Mad Al Monfasel "مد النفصل"')
    madd_mottasel_len: Literal[4, 5] = Field(
        description='ArabicName(مقدار المد المتصل) The length of Mad Al Motasel "مد المتصل"')
    madd_aared_len: Literal[2, 4, 6] = Field(
        description='ArabicName(مقدار المد العارض) The length of Mad Al Aared "مد العارض للسكون"')
    madd_mottasel_mahmooz_aared_len: Literal[4, 5, 6] = Field(
        description='ArabicName(مقدار المد المتصل المتطرف المهموز عند الوقف)'
        ' The length of Madd Almotasel Al Mahmooz during waqf'
        ' "مد المتصل المهموز عند الوقف". Example "السماء"')
    madd_alayn_lazem_len: Literal[4, 6] = Field(
        description='ArabicName(مقدار   المد اللازم الحرفي للعين)'
        ' The length of Lzem Harfy Madd "المد الحرفي اللازم لحرف العين" in'
        ' surar: Maryam "مريم", AlShura "الشورى" either 4 or 6')
    tasheel_or_madd: Literal['tasheel', 'madd'] = Field(
        description='ArabicName(تسهيل أم مد) Tasheel of Madd'
        ' "وجع التسهيل أو المد" for 6 words in The Holy Quran:'
        ' "ءالذكرين", "ءالله", "ءائن"')
    daaf_harka: Literal['fath', 'dam'] = Field(
        description='ArabicName(وجه كلمة ضعف) The Haraka of "ضعف" in'
        ' surah AlRoom "الروم" aya(54)')
    idghaam_nakhlqkm: Literal['kamel', 'nakes'] = Field(
        description='ArabicName(نوع الإدغام في كلمة نخلقكم) The Idghaof of'
        ' word "نخلقكم" in suran Almurslat "المرسلات" Aya (20)'
        ' Either Idgham nakes "إدغام نافص" or Idghtam kamel "إدغام كامل"')
    noon_tamanna: Literal['ishmam', 'ikhtlas'] = Field(
        description='ArabicName(وجع تأمننا) Warys to recite word'
        ' tammna "تأمنا" ''in surah Yusuf Aya(11) "سورة يوسف".'
        ' Eeither Ishmam "إشمام" or Ikhtlas "اختلاس"')
