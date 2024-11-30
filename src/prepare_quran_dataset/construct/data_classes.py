from typing import Literal, Optional, Self
from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic.fields import FieldInfo, PydanticUndefined
from pathlib import Path


from .utils import get_audiofile_info
from .docs_utils import get_moshaf_field_docs
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
        arabic_name='الاسم باللغة الإنجليزية')
    country_code: str = Field(
        min_length=2, max_length=2,
        field_arabic_name='رمز الدولة بالإنجليزية',
        description='The two number ISO country code of the reciter',
    )
    moshaf_set_ids: set[str] = Field(
        default_factory=lambda: set(),
        description='Every Moshaf ID is a string has the following structure "reciter_id"."mohaf_id"')


class AudioFile(BaseModel):
    name: str = Field(description='Name of the file')
    path: str = ""
    sample_rate: int
    duration_minutes: float


class Moshaf(BaseModel):
    id: str = Field(
        default="",
        description='Every Moshaf ID is a string has the following structure "reciter_id"."mohaf_id"')
    name: str = Field(
        description='The arabic name of the moshaf i.e "المصحف المرتل"')
    path: str = Field(default="", description='Absolute path to the moshaf')
    reciter_id: int = Field(description='The ID of the reciter starting of 0')
    reciter_arabic_name: str = ""
    reciter_english_name: str = ""
    sources: list[str] = Field(
        description='List of urls to download recitations')
    specific_sources: dict[int, str] = Field(
        default={},
        description='Overwriting a specific group of files.'
        'As: {"sura_integer_index between 1 and 114": "sura_url"}'
        'Ex: {"002": "url_for_002"} will overwrite the recitation "002"'
        'downloaded by the `sources` attributes'
    )
    downloaded_sources: list[str] = Field(
        default=[],
        description='List of downloaded urls (sources) either from'
        ' sources or from specific sources')
    is_sura_parted: bool = Field(
        default=True,
        description='If every recitation file is a sperate sura or not')
    missing_recitations: set = Field(
        default=set(),
        description="The missing recitations from the Downloaded Moshaf"
        "It will filled if only `is_sura_parted==True` "
    )

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
    # Core Attributes (كليات)
    rewaya: Literal['hafs'] = Field(
        field_arabic_name='الرواية',
        field_arabic_attrs_map={
            "hafs": "حفص"
        },
        description='The type of the quran Rewaya.'

    )
    takbeer: Literal['no_takbeer', 'beginning_of_sharh', 'end_of_doha', 'general_takbeer'] = Field(
        default='no_takbeer',

        field_arabic_name="التكبير",
        field_arabic_attrs_map={
            "no_takbeer": "لا تكبير",
            "beginning_of_sharh": "التكبير من أول الشرح لأول الناس",
            "end_of_doha": "التكبير من آخر الضحى لآخر الناس",
            "general_takbeer": "التكبير أول كل سورة إلا التوبة"
        },
        description='The ways to add takbeer (الله أكبر) after Istiaatha'
        ' (استعاذة) and between end of the surah and beginning of the surah.'
        ' `no_takbeer`: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation)'
        ' `beginning_of_sharh`: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas'
        ' `end_of_dohaf`: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas'
        ' `general_takbeer`: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah'
    )

    madd_monfasel_len: Literal[2, 3, 4, 5] = Field(
        field_arabic_name="مد المنفصل",
        description=' The length of Mad Al Monfasel "مد النفصل" for Hafs Rewaya.'
    )
    madd_mottasel_len: Literal[4, 5, 6] = Field(
        field_arabic_name="مقدار المد المتصل",
        description=' The length of Mad Al Motasel "مد المتصل" for Hafs.'
    )
    madd_mottasel_waqf: Literal[4, 5, 6] = Field(
        field_arabic_name="مقدار المد المتصل وقفا",
        description=' The length of Madd Almotasel at pause for Hafs.'
        '. Example "السماء".')
    madd_aared_len: Literal[2, 4, 6] = Field(
        field_arabic_name="مقدار المد العارض",
        description=' The length of Mad Al Aared "مد العارض للسكون".')
    madd_alleen_len: Literal[2, 4, 6] = Field(
        default=None,
        field_arabic_name="مقدار مد اللين",
        description='The length of the Madd al-Leen when stopping at the end of a word'
        ' (for a sakin waw or ya preceded by a letter with a fatha) should be'
        " less than or equal to the length of Madd al-'Arid (the temporary stretch due to stopping)."
        ' **Default Value is equal to `madd_aared_len`**.'
        ' مقدرا مع اللين عن القوف (للواو الساكنة والياء الساكنة وقبلها حرف مفتوح) ويجب أن يكون مقدار مد اللين أقل من أو يساوي مع العارض'

    )
    ghonna_lam_and_raa: Literal['ghonna', 'no_ghonna'] = Field(
        default='no_ghonna',
        field_arabic_name="غنة اللام و الراء",
        field_arabic_attrs_map={
            "ghonna": "غنة",
            "no_ghonna": "لا غنة"
        },
        description='The ghonna for merging (Idghaam) noon with Lam and Raa for Hafs.'
    )

    # (الجزئيات)
    meem_aal_imran: Literal['waqf', 'wasl_2', 'wasl_6'] = Field(
        default='waqf',
        field_arabic_name="ميم آل عمران في قوله تعالى: {الم الله} وصلا",
        field_arabic_attrs_map={
            "waqf": "وقف",
            "wasl_2": "فتح الميم ومدها حركتين",
            "wasl_6": "فتح الميم ومدها ستة حركات"
        },
        description='The ways to recite the word meem Aal Imran (الم الله)'
        ' at connected recitation.'
        ' `waqf`: Pause with a prolonged madd (elongation) of 6 harakat (beats).'
        ' `wasl_2` Pronounce "meem" with fathah (a short "a" sound) and stretch it for 2 harakat.'
        ' `wasl_6` Pronounce "meem" with fathah and stretch it for 6 harakat.'
    )
    madd_yaa_alayn_alharfy: Literal[4, 6] = Field(
        default=6,
        field_arabic_name="مقدار   المد اللازم الحرفي للعين",
        description=' The length of Lzem Harfy of Yaa in letter Al-Ayen Madd'
        ' "المد الحرفي اللازم لحرف العين" in'
        ' surar: Maryam "مريم", AlShura "الشورى".')
    saken_before_hamz: Literal['tahqeek', 'general_sakt', 'local_sakt'] = Field(
        default='tahqeek',
        field_arabic_name="الساكن قبل الهمز",
        field_arabic_attrs_map={
            "tahqeek": "تحقيق",
            "general_sakt": "سكت عام",
            "local_sakt": "سكت خاص"
        },
        description='The ways of Hafs for saken before hamz. '
        '"The letter with sukoon before the hamzah (ء)".'
        "And it has three forms: full articulation (`tahqeeq`),"
        " general pause (`general_sakt`), and specific pause (`local_skat`)."

    )
    sakt_iwaja: Literal['sakt', 'waqf', 'idraj'] = Field(
        default='waqf',
        field_arabic_name="السكت عند عوجا في الكهف",
        field_arabic_attrs_map={
            "sakt": "سكت",
            "waqf": "وقف",
            "idraj": "إدراج"
        },
        description='The ways to recite the word "عوجا" (Iwaja).'
        ' `sakt` means slight pause.'
        ' `idraj` means not `sakt`.'
        ' `waqf`:  means full pause, so we can not determine weither'
        ' the reciter uses `sakt` or `idraj` (no sakt).'
    )
    sakt_marqdena: Literal['sakt', 'waqf', 'idraj'] = Field(
        default='waqf',
        field_arabic_name="السكت عند مرقدنا  في يس",
        field_arabic_attrs_map={
            "sakt": "سكت",
            "waqf": "وقف",
            "idraj": "إدراج"
        },
        description='The ways to recite the word "مرقدنا" (Marqadena) in Surat Yassen.'
        ' `sakt` means slight pause.'
        ' `idraj` means not `sakt`.'
        ' `waqf`:  means full pause, so we can not determine weither'
        ' the reciter uses `sakt` or `idraj` (no sakt).'

    )
    sakt_man_raq: Literal['sakt', 'waqf', 'idraj'] = Field(
        default='sakt',
        field_arabic_name="السكت عند  من راق في القيامة",
        field_arabic_attrs_map={
            "sakt": "سكت",
            "waqf": "وقف",
            "idraj": "إدراج"
        },
        description='The ways to recite the word "من راق" (Man Raq) in Surat Al Qiyama.'
        ' `sakt` means slight pause.'
        ' `idraj` means not `sakt`.'
        ' `waqf`:  means full pause, so we can not determine weither'
        ' the reciter uses `sakt` or `idraj` (no sakt).'

    )
    sakt_bal_ran: Literal['sakt', 'waqf', 'idraj'] = Field(
        default='sakt',
        field_arabic_name="السكت عند  بل ران في  المطففين",
        field_arabic_attrs_map={
            "sakt": "سكت",
            "waqf": "وقف",
            "idraj": "إدراج"
        },
        description='The ways to recite the word "بل ران" (Bal Ran) in Surat Al Motaffin.'
        ' `sakt` means slight pause.'
        ' `idraj` means not `sakt`.'
        ' `waqf`:  means full pause, so we can not determine weither'
        ' the reciter uses `sakt` or `idraj` (no sakt).'

    )
    sakt_maleeyah: Literal['sakt', 'waqf', 'idgham'] = Field(
        default='waqf',
        field_arabic_name="وجه  قوله تعالى {ماليه هلك} بالأحقاف",
        field_arabic_attrs_map={
            "sakt": "سكت",
            "waqf": "وقف",
            "idgham": "إدغام"
        },
        description='The ways to recite the word {ماليه هلك} in Surah Al-Ahqaf.'
        ' `sakt` means slight pause.'
        " `idgham` Assimilation of the letter 'Ha' (ه) into the letter 'Ha' (ه) with complete assimilation."
        '`waqf`:  means full pause, so we can not determine weither'
        ' the reciter uses `sakt` or `idgham`.'

    )
    between_anfal_and_tawba: Literal['waqf', 'sakt', 'wasl'] = Field(
        default='waqf',
        field_arabic_name="وجه بين الأنفال والتوبة",
        field_arabic_attrs_map={
            "waqf": "وقف",
            "sakt": "سكت",
            "wasl": "وصل"
        },
        description='The ways to recite end of Surah Al-Anfal and beginning of Surah At-Tawbah.'
    )
    noon_and_yaseen: Literal['izhar', 'idgham'] = Field(
        field_arabic_name="الإدغام والإظهار في النون عند الواو من قوله تعالى: {يس والقرآن}و {ن والقلم}",
        field_arabic_attrs_map={
            "izhar": "إظهار",
            "idgham": "إدغام"
        },
        default='izhar',
        description='Weither to merge noon of both: {يس} and {ن} with (و) "`idgham`" or not "`izhar`".'
    )
    yaa_ataan: Literal['wasl', 'hadhf', 'ithbat'] = Field(
        default='wasl',
        field_arabic_name=" إثبات الياء وحذفها وقفا في قوله تعالى {آتان} بالنمل",
        field_arabic_attrs_map={
            "wasl": "وصل",
            "hadhf": "حذف",
            "ithbat": "إثبات"
        },
        description="The affirmation and omission of the letter 'Yaa' in the pause of the verse {آتاني} in Surah An-Naml."
        '`wasl`: means connected recitation without pasuding as (آتانيَ).'
        '`hadhf`: means deletion of letter (ي) at puase so recited as (آتان).'
        '`ithbat`: means confirmation reciting letter (ي) at puase as (آتاني).'
    )
    start_with_ism: Literal['wasl', 'lism', 'alism'] = Field(
        default='wasl',
        field_arabic_name="وجه البدأ بكلمة {الاسم} في سورة الحجرات",
        field_arabic_attrs_map={
            "wasl": "وصل",
            "lism": "لسم",
            "alism": "ألسم"
        },
        description="The ruling on starting with the word {الاسم} in Surah Al-Hujurat."
        '`lism` Recited as (لسم) at the beginning. '
        '`alism` Recited as (ألسم). ath the beginning'
        '`wasl`: means completing recitaion without paussing as normal, '
        'So Reciting is as (بئس لسم).'
    )
    yabsut: Literal['seen', 'saad'] = Field(
        field_arabic_name="السين والصاد في قوله تعالى: {والله يقبض ويبسط} بالبقرة",
        field_arabic_attrs_map={
            "seen": "سين",
            "saad": "صاد"
        },
        default='seen',
        description="The ruling on pronouncing `seen` (س) or `saad` (ص) in the verse {والله يقبض ويبسط} in Surah Al-Baqarah."
    )
    bastah: Literal['seen', 'saad'] = Field(
        default='seen',
        field_arabic_name="السين والصاد في قوله تعالى:  {وزادكم في الخلق بسطة} بالأعراف",
        field_arabic_attrs_map={
            "seen": "سين",
            "saad": "صاد"
        },
        description="The ruling on pronouncing `seen` (س) or `saad` (ص ) in the verse {وزادكم في الخلق بسطة} in Surah Al-A'raf."
    )
    almusaytirun: Literal['seen', 'saad'] = Field(
        default='saad',
        field_arabic_name="السين والصاد في قوله تعالى {أم هم المصيطرون} بالطور",
        field_arabic_attrs_map={
            "seen": "سين",
            "saad": "صاد"
        },
        description="The pronunciation of `seen` (س) or `saad` (ص ) in the verse {أم هم المصيطرون} in Surah At-Tur."
    )
    bimusaytir: Literal['seen', 'saad'] = Field(
        default='saad',
        field_arabic_name="السين والصاد في قوله تعالى:  {لست عليهم بمصيطر} بالغاشية",
        field_arabic_attrs_map={
            "seen": "سين",
            "saad": "صاد"
        },
        description="The pronunciation of `seen` (س) or `saad` (ص ) in the verse {لست عليهم بمصيطر} in Surah Al-Ghashiyah."
    )
    tasheel_or_madd: Literal['tasheel', 'madd'] = Field(
        default='madd',
        field_arabic_name="همزة الوصل في قوله تعالى: {آلذكرين} بموضعي الأنعام و{آلآن} موضعي يونس و{آلله} بيونس والنحل",
        field_arabic_attrs_map={
            "tasheel": "تسهيل",
            "madd": "مد"
        },
        description=' Tasheel of Madd'
        ' "وجع التسهيل أو المد" for 6 words in The Holy Quran:'
        ' "ءالذكرين", "ءالله", "ءائن".')
    yalhath_dhalik: Literal['izhar', 'idgham', 'waqf'] = Field(
        default='idgham',
        field_arabic_name="الإدغام وعدمه في قوله تعالى: {يلهث ذلك} بالأعراف",
        field_arabic_attrs_map={
            "izhar": "إظهار",
            "idgham": "إدغام",
            "waqf": "وقف"
        },
        description="The assimilation (`idgham`) and non-assimilation (`izhar`) in the verse {يلهث ذلك} in Surah Al-A'raf."
        " `waqf`: means the rectier has paused on (يلهث)"
    )
    irkab_maana: Literal['izhar', 'idgham', 'waqf'] = Field(
        default='idgham',
        field_arabic_name="الإدغام والإظهار في قوله تعالى: {اركب معنا} بهود",
        field_arabic_attrs_map={
            "izhar": "إظهار",
            "idgham": "إدغام",
            "waqf": "وقف"
        },
        description="The assimilation and clear pronunciation in the verse {اركب معنا} in Surah Hud."
        'This refers to the recitation rules concerning whether the letter'
        ' "Noon" (ن) is assimilated into the following letter or pronounced'
        ' clearly when reciting this specific verse.'
        " `waqf`: means the rectier has paused on (اركب)"
    )
    noon_tamnna: Literal['ishmam', 'rawm'] = Field(
        default='ishmam',
        field_arabic_name=" الإشمام والروم (الاختلاس) في قوله تعالى {لا تأمنا على يوسف}",
        field_arabic_attrs_map={
            "ishmam": "إشمام",
            "rawm": "روم"
        },
        description="The nasalization (`ishmam`) or the slight drawing (`rawm`) in the verse {لا تأمنا على يوسف}"
    )
    harakat_daaf: Literal['fath', 'dam'] = Field(
        default='fath',
        field_arabic_name="حركة الضاد (فتح أو ضم) في قوله تعالى {ضعف} بالروم",
        field_arabic_attrs_map={
            "fath": "فتح",
            "dam": "ضم"
        },
        description="The vowel movement of the letter 'Dhad' (ض) (whether with `fath` or `dam`) in the word {ضعف} in Surah Ar-Rum."
    )
    alif_salasila: Literal['hadhf', 'ithbat', 'wasl'] = Field(
        default='wasl',
        field_arabic_name="إثبات الألف وحذفها وقفا في قوله تعالى: {سلاسلا} بسورة الإنسان",
        field_arabic_attrs_map={
            "hadhf": "حذف",
            "ithbat": "إثبات",
            "wasl": "وصل"
        },
        description="Affirmation and omission of the 'Alif' when pausing in the verse {سلاسلا} in Surah Al-Insan."
        'This refers to the recitation rule regarding whether the final'
        ' "Alif" in the word "سلاسلا" is pronounced (affirmed) or omitted'
        ' when pausing (waqf) at this word during recitation in the specific'
        ' verse from Surah Al-Insan.'
        ' `hadhf`: means to remove alif (ا) during puase as (سلاسل)'
        ' `ithbat`: means to recite alif (ا) during puase as (سلاسلا)'
        ' `wasl` means completing the recitation as normal without pausing'
        ', so recite it as (سلاسلَ وأغلالا)'
    )
    idgham_nakhluqkum: Literal['idgham_naqis', 'idgham_kamil'] = Field(
        default='idgham_kamil',
        field_arabic_name="إدغام القاف في الكاف إدغاما ناقصا أو كاملا {نخلقكم} بالمرسلات",
        field_arabic_attrs_map={
            "idgham_kamil": "إدغام كامل",
            "idgham_naqis": "إدغام ناقص"
        },
        description="Assimilation of the letter 'Qaf' into the letter 'Kaf,' whether incomplete (`idgham_naqis`) or complete (`idgham_kamil`), in the verse {نخلقكم} in Surah Al-Mursalat."
    )
    raa_firq: Literal['waqf', 'tafkheem', 'tarqeeq'] = Field(
        default='tafkheem',
        field_arabic_name="التفخيم والترقيق في راء {فرق} في الشعراء وصلا",
        field_arabic_attrs_map={
            "waqf": "وقف",
            "tafkheem": "تفخيم",
            "tarqeeq": "ترقيق"
        },
        description="Emphasis and softening of the letter 'Ra' in the word {فرق} in Surah Ash-Shu'ara' when connected (wasl)."
        'This refers to the recitation rules concerning whether the'
        ' letter "Ra" (ر) in the word "فرق"  is pronounced with'
        ' emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the'
        " specific verse from Surah Ash-Shu'ara' in connected speech."
        ' `waqf`: means pasuing so we only have one way (tafkheem of Raa)'
    )
    raa_alqitr: Literal['wasl', 'tafkheem', 'tarqeeq'] = Field(
        default='wasl',
        field_arabic_name="التفخيم والترقيق في راء {القطر} في سبأ وقفا",
        field_arabic_attrs_map={
            "wasl": "وصل",
            "tafkheem": "تفخيم",
            "tarqeeq": "ترقيق"
        },
        description="Emphasis and softening of the letter 'Ra' in the word {القطر} in Surah Saba' when pausing (waqf)."
        'This refers to the recitation rules regarding whether the letter "Ra"'
        ' (ر) in the word "القطر" is pronounced with emphasis'
        " (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Saba'."
        ' `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)'
    )
    raa_misr: Literal['wasl', 'tafkheem', 'tarqeeq'] = Field(
        default='wasl',
        field_arabic_name="التفخيم والترقيق في راء {مصر} في يونس وموضعي يوسف والزخرف  وقفا",
        field_arabic_attrs_map={
            "wasl": "وصل",
            "tafkheem": "تفخيم",
            "tarqeeq": "ترقيق"
        },
        description="Emphasis and softening of the letter 'Ra' in the word {مصر} in Surah Yunus, and in the locations of Surah Yusuf and Surah Az-Zukhruf when pausing (waqf)."
        'This refers to the recitation rules regarding whether the letter "Ra"'
        ' (ر) in the word "مصر" is pronounced with emphasis (`tafkheem`)'
        ' or softening (`tarqeeq`) at the specific pauses in these Surahs.'
        ' `wasl`: means not pasuing so we only have one way (tafkheem of Raa)'
    )
    raa_nudhur: Literal['wasl', 'tafkheem', 'tarqeeq'] = Field(
        default='tafkheem',
        field_arabic_name="التفخيم والترقيق  في راء {نذر} بالقمر وقفا",
        field_arabic_attrs_map={
            "wasl": "وصل",
            "tafkheem": "تفخيم",
            "tarqeeq": "ترقيق"
        },
        description="Emphasis and softening of the letter 'Ra' in the word {نذر} in Surah Al-Qamar when pausing (waqf)."
        'This refers to the recitation rules regarding whether the letter "Ra"'
        ' (ر) in the word "نذر" is pronounced with emphasis (`tafkheem`)'
        ' or softening (`tarqeeq`) when pausing at this word in Surah Al-Qamar.'
        ' `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)'
    )
    raa_yasr: Literal['wasl', 'tafkheem', 'tarqeeq'] = Field(
        default='tafkheem',
        field_arabic_name="التفخيم والترقيق في راء {يسر} في الفجر  وقفا",
        field_arabic_attrs_map={
            "wasl": "وصل",
            "tafkheem": "تفخيم",
            "tarqeeq": "ترقيق"
        },
        description="Emphasis and softening of the letter 'Ra' in the word {يسر} in Surah Al-Fajr when pausing (waqf)."
        'This refers to the recitation rules regarding whether the letter "Ra"'
        ' (ر) in the word "يسر" is pronounced with emphasis (`tafkheem`)'
        ' or softening (`tarqeeq`) when pausing at this word in Surah Al-Fajr.'
        ' `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)'
    )

    @model_validator(mode='after')
    def check_madd_alleen(self) -> Self:
        if self.madd_alleen_len > self.madd_aared_len:
            raise ValueError(
                f'مد  اللين يجب أن يكون أقل من أو يساوي مد العارض للسكون. مد العارض ({self.madd_aared_len})و مد اللين ({self.madd_alleen_len}).')
        return self

    @field_validator('specific_sources', mode='after')
    @classmethod
    def valid_sura_index(cls, specific_sources: dict[int, str]) -> str:
        for sura_index in specific_sources:

            if sura_index < 1 or sura_index > 114:
                raise ValueError(
                    f'Sura Index must be integer between 1 and 114, we got "{sura_index}"')
        return specific_sources

    def model_post_init(self, *args, **kwargs):
        self.is_downloaded = set(self.downloaded_sources) == (
            set(self.sources) | set(self.specific_sources.values()))

        if self.madd_alleen_len is None:
            self.madd_alleen_len = self.madd_aared_len

    def fill_metadata_after_download(self, moshaf_path: Path):

        total_duration_minutes = 0.0
        total_size_megabytes = 0.0
        recitation_files: list[AudioFile] = []
        for filepath in moshaf_path.iterdir():
            audio_file_info = get_audiofile_info(filepath)
            if audio_file_info:
                audio_file = AudioFile(
                    name=filepath.name,
                    path=str(filepath.absolute()),
                    sample_rate=audio_file_info.sample_rate,
                    duration_minutes=audio_file_info.duration_seconds / 60.0
                )
                recitation_files.append(audio_file)
                total_duration_minutes += audio_file_info.duration_seconds / 60.0
                total_size_megabytes += filepath.stat().st_size / (1024.0 * 1024.0)

        self.recitation_files = recitation_files
        self.path = str(moshaf_path.absolute())
        self.num_recitations = len(recitation_files)
        self.total_duraion_minutes = total_duration_minutes
        self.total_size_mb = total_size_megabytes
        self.downloaded_sources = list(
            set(self.sources) | set(self.specific_sources.values()))
        self.is_downloaded = set(self.downloaded_sources) == (
            set(self.sources) | set(self.specific_sources.values()))

        if self.is_sura_parted:
            self.is_complete = (self.get_all_sura_names() ==
                                self.get_downloaded_suar_names())
        else:
            self.is_complete = self.is_downloaded

        if self.is_sura_parted:
            self.missing_recitations = (
                self.get_all_sura_names() - self.get_downloaded_suar_names())

    def get_downloaded_suar_names(self) -> set[str]:
        """Gets the sura names of the dwonloaded recitations"""
        suar_names_set = set()
        for audiofile in self.recitation_files:
            sura_name = audiofile.name.split('.')[0]
            suar_names_set.add(sura_name)
        return suar_names_set

    def get_all_sura_names(self) -> set[str]:
        """Retruns the suar names as set of '001', '002', ...'114' """
        suar_names = set()
        for idx in range(1, 115, 1):
            suar_names.add(f'{idx:0{3}}')
        return suar_names

    @classmethod
    def generate_docs(cls) -> str:
        """Generates documentations for the Qura'anic Fields
        """
        md_table = '|Attribute Name|Arabic Name|Values|Default Value|More Info|'
        md_table += '\n' + '|-' * 5 + '|' + '\n'

        for fieldname, fieldinfo in cls.model_fields.items():
            docs = get_moshaf_field_docs(fieldname, fieldinfo)
            if not docs:
                continue
            md_table += '|'
            md_table += docs.english_name + '|'
            md_table += docs.arabic_name + '|'

            # Values
            for en, ar in docs.english2arabic_map.items():
                if en == ar:
                    md_table += f'- `{en}`<br>'
                else:
                    md_table += f'- `{en}` (`{ar}`)<br>'
            md_table += '|'

            # Default Value
            if fieldinfo.default == PydanticUndefined:
                md_table += '|'
            else:
                if fieldinfo.default is None:
                    md_table += '`None`|'
                else:
                    ar_val = docs.english2arabic_map[fieldinfo.default]
                    if ar_val == fieldinfo.default:
                        md_table += f'`{ar_val}`|'
                    else:
                        md_table += f'`{fieldinfo.default}` (`{ar_val}`)|'

            md_table += docs.more_info + '|'

            md_table += '\n'

        return md_table
