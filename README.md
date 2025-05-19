# prepare-quran-dataset

Preparing and chunking quranic recitation dataset for Automatic Pronunciation Error Correction and Detection for the Holy Quran

## Table of Content

* [Installation](#installation)
    * [Main Package](#main-package)
    * [Frontend](#frontend)
    * [Demo](#demo)
* [Hafs Ways](#أوجه-حفص)
* [Moshaf Attributes Docs](#moshaf-attributes-docs)

## Description

* This repos is a part of a project to create an AI quranic recitation teacher. in this repo we are building and downloading recitations form the web and organizing them as a training dataset for the project.
* The core database for gathering and annotating the recitations are in `src` dir as `prepare-quran-dataset` pip package along with a frontend in`frontend` directory.
* Our elementary element for building our Quranic dataset is `Moshaf` found in `src/prepare_quran_dataset/construct/data_classes.py` as a pydantic dataclass.
* The `Moshaf` Quranic attributes are listed [here](#moshaf-attributes-docs).

## Installation

We have two components to install:

* Main Package: which have the main dataclasses and the core logic
* Frontend: Which is a streamlit app to interact easily with the main package

### Main Package

1. Install python 3.12 or 3.13 using anaconda or any other option

2. clone the repo

```bash
git clone https://github.com/obadx/prepare-quran-dataset.git
```

3. Install the package. We are installing the package as a regular pip package

```bash
cd prepare-quran-dataset
python -m pip install -e ./
```

`-e` means editable installation

4. Run the unittests as verification

```bash
python -m unittest tests/database_unittest.py 
```

### Frontend

1. Install requirements:

```bash
cd frontend
python -m pip install -r requirements.txt
```

2. Run the app:

```bash
python -m streamlit run streamlit_app.py
```

### Demo

Basics:

https://github.com/user-attachments/assets/a9742b75-985c-4808-9fcb-cc8c0b6ff549





## أوجه حفص

### الكليات

#### 1. التكبير

ثلاث مذاهب:

* التكبير من أول (ألم نشرح) وما بعدها إلي أول (الناس)
* التكبير من آخر (الضحى) وما بعدها إلى آخر (الناس)
* التكبير أول كل سورة سوى براءة (التوبة)

#### 2. المد المنفصل والمتصل

* **المد المنفصل:** له 2و 3و 4و 5 حركات على التفصيل لكل طريق انظر في [المصادر](#المصادر)
* **المد المتصل:** له 4و 5و 6 حركات على التفصيل لك طرق أنظر في المصادر.

#### 3. الساكن قبل الهمز

له ثلاثة أوجه في السكت على (الساكن) و (الواو والياء بعد فتح) قبل الهمز

* عدم السكت
* السكت الخاص: السكت على (أل) و(شيء) و(الساكن المفصول)
* السكت العام: السكت الخاص ويضاف إليه السكت على الساكن الموصول

#### 4. النون الساكنة التنويون عند اللام والراء

له وجهان في إدغام النون والتوين مع اللام والراء إدغاما:

* بغير غنة
* بغنة

-----------------------

### الجزئيات

#### 5. السين والصاد في قوله تعالى: {والله يقبض ويبسط} بالبقرة و {وزادكم في الخلق بسطة} بالأعراف

في كلا الموضعين يوجد طرق بنطق كل واحد منها بال:

* السين
* الصاد  
للتفصيل انظر المصادر

#### 6. السين والصاد في قوله تعالى {أم هم المصيطرون}و {لست عليهم بمصيطر}

في كلا الموضعين يوجد طرق بنطق كل واحد منها بال:

* السين
* الصاد  
للتفصيل انظر المصادر

#### 7. همزة الوصل في قوله تعالى: {آلذكرين} بموضعي الأنعام و{آلآن} موضعي يونس و{آلله} بيونس والنحل

فيها وجهان

* إبدالهما مدا مشبعا (6 حركات)
* تسهيلها بين الهمزة والألف

#### 8. الإدغام وعدمه في قوله تعالى: {يلهث ذلك} بالأعراف

فيها وجهان

* إدغام الثاء في الذال إدغامل كاملا
* إظهار الثاء

#### 9. الإدغام والإظهار في قوله تعالى: {اركب معنا} بهود

فيها وجهان

* إدغام الباء في الميم
* إظهار الباء

#### 10. الإدغام والإظهار في  النون عند الواو من قوله تعالى: {يس والقرآن}و {ن والقلم}

فيها وجهان

* إظهار النون
* إدغام النون في الواو

#### 11. الإشمام والروم (الاختلاس) في قوله تعالى {لا تأمنا على يوسف}

بها وجهان

* إدغام النون مع الإشمام
* تحريك النوننين {تأمَنُنَا} مع الاختلاس أو الروم

#### 12. السكت والإدراج (عدم السكت) في قوله تعالي {عوجا} أول الكهف و{مرقدنا هذا} في يس و{من راق} في القيامة و{بل ران} في المطففين

بكل واحد منها وجهان مع اختلاف الطرق ويرجع [للمصادر](#المصادر): 

* السكت
* الإدراج (عدم السكت)

#### 13. مد ياء العين في قوله تعالى {كهيعص} أول مريم و{حم عسق} أول الشورى

بها ثلاثة أوجه:

* حركتان
* أربع حركات
* ستة حركات (مد مشبع)

#### 14. التفخيم والترقيق في راء {فرق} في الشعراء وصلا

وجهان وصلا:

* تفخيم الراء
* ترقيق الراء

#### 15. إثبات الياء وحذفها وقفا في قوله تعالى {آتان} بالنمل

وجهان وقفا:

* حذف الياء {آتان}
* إثبات الياء {آتاني}

#### 16. حركة الضاد (فتح أو ضم) في قوله تعالى {الله الذي خلقكم من ضعف ثم جعمل من ضعف قوة ثم جعل من بعد قوة ضعفا وشيبة} بالروم

فيها وجهان في الثلاث كلمات

* الفتح
* الضم

#### 17. إثبات الألف وحذفها وقفا في قوله تعالى: {سلاسلا} بسورة الإنسان

فيها وجهان وقفا:

* إثبات الألف {سلاسلا}
* حذف الألف {سلاسل}

#### 18. إدغام القاف في الكاف إدغاما ناقصا أو كاملا {نخلقكم} بالمرسلات

بها وجهان:

* إدغام القا في الكاف إدغاما *كاملا*
* إدغام القاف في الكاف إدغام *ناقصا*


### أوجه لكل طرق حفص

#### 19. ميم آل عمران في قوله تعالى: {الم الله} وصلا

وجهان وصلا ووجه واحد وقفا

* فتح الميم ومدها مدا طبيعيا (حركتان) وصلا
* فتح الميم ومدها مدا مشبعا (6 حركات) وصلا
* الوقوف على الميم ومدها مدا مشبعا (6 حركات)

#### 20. السكت والإدغام في قوله تعالى: {ماليه هلك} بالأحقاف

وجهان:

* إدغام الهاء في الهاء
* السكت على الهاء الأولى

#### 21. الأوجه بين الأنفال والتوبة

ثلاثة أوجه 

* قطع الجميع (الوقف)
* السكت
* الوصل

#### 22. التفخيم والترقيق في الراء في قوله تعالى: {القطر} بسبأ و{مصر} و{نذر} بالقمر و{يسر} و{أن أسر} و {فأسر} وقفا

بكل منها وجهان وقفا:

* التفخيم
* الترقيف  

ويترجح في كل كلمة وجه معين:

* الأرجح في {القطر}: الترقيق
* الأرجح في {مصر}: التفخيم
* الأرجح في {نذر}: التفخيم
* الأرجح في {بسر} بالفجر و{أن أسر} باطه والشعراء و{فأسر} بهود والحجر والدخان وقفا: الترقيق

#### 23. الابتداء بقوله تعالى: {الاسم} في سورة الحجرات

وجهان:

* إثبات ألف الوصل في (أل) :(ءَلِسم)
* حذف ألف الوصل في (أل) : (لِسم)

## تنبيه

معنى وجود أكثر من وجه لا يعنى أن لك الطرق تلك الأوجه كلها ولكن يرجع [للمصادر](#المصادر) لمعرفة الأوجه المحددة كل طريق

## المصادر

* [صريح النص في الأوجه المختلف فيها عند حفص للشيخ الضباع رحمه الله](https://dlib.nyu.edu/files/books/columbia_aco000774/columbia_aco000774_hi.pdf)

----------------------------

## Moshaf Attributes Docs

|-|-|-|-|-|
|rewaya|الرواية|- `hafs` (`حفص`)<br>||The type of the quran Rewaya.|
|recitation_speech|سرعة التلاوة|- `mujawad` (`مجود`)<br>- `above_murattal` (`فويق المرتل`)<br>- `murattal` (`مرتل`)<br>- `hadr` (`حدر`)<br>||The recitation speech sorted from slowes to fastest سرعة التلاوة مرتبة من الأبطأ إلي الأسرع:
            * `mujawad`: مجود
            * `above_murattal` فويق المرتل
            * `murattal`: مرتل
            * `hadr`: حدر
        |
|takbeer|التكبير|- `no_takbeer` (`لا تكبير`)<br>- `beginning_of_sharh` (`التكبير من أول الشرح لأول الناس`)<br>- `end_of_doha` (`التكبير من آخر الضحى لآخر الناس`)<br>- `general_takbeer` (`التكبير أول كل سورة إلا التوبة`)<br>|`no_takbeer` (`لا تكبير`)|The ways to add takbeer (الله أكبر) after Istiaatha (استعاذة) and between end of the surah and beginning of the surah. `no_takbeer`: "لا تكبير" — No Takbeer (No proclamation of greatness, i.e., there is no Takbeer recitation) `beginning_of_sharh`: "التكبير من أول الشرح لأول الناس" — Takbeer from the beginning of Surah Ash-Sharh to the beginning of Surah An-Nas `end_of_dohaf`: "التكبير من آخر الضحى لآخر الناس" — Takbeer from the end of Surah Ad-Duha to the end of Surah An-Nas `general_takbeer`: "التكبير أول كل سورة إلا التوبة" — Takbeer at the beginning of every Surah except Surah At-Tawbah|
|madd_monfasel_len|مد المنفصل|- `2`<br>- `3`<br>- `4`<br>- `5`<br>|| The length of Mad Al Monfasel "مد النفصل" for Hafs Rewaya.|
|madd_mottasel_len|مقدار المد المتصل|- `4`<br>- `5`<br>- `6`<br>|| The length of Mad Al Motasel "مد المتصل" for Hafs.|
|madd_mottasel_waqf|مقدار المد المتصل وقفا|- `4`<br>- `5`<br>- `6`<br>|| The length of Madd Almotasel at pause for Hafs.. Example "السماء".|
|madd_aared_len|مقدار المد العارض|- `2`<br>- `4`<br>- `6`<br>|| The length of Mad Al Aared "مد العارض للسكون".|
|madd_alleen_len|مقدار مد اللين|- `2`<br>- `4`<br>- `6`<br>|`None`|The length of the Madd al-Leen when stopping at the end of a word (for a sakin waw or ya preceded by a letter with a fatha) should be less than or equal to the length of Madd al-'Arid (the temporary stretch due to stopping). **Default Value is equal to `madd_aared_len`**. مقدرا مع اللين عن القوف (للواو الساكنة والياء الساكنة وقبلها حرف مفتوح) ويجب أن يكون مقدار مد اللين أقل من أو يساوي مع العارض|
|ghonna_lam_and_raa|غنة اللام و الراء|- `ghonna` (`غنة`)<br>- `no_ghonna` (`لا غنة`)<br>|`no_ghonna` (`لا غنة`)|The ghonna for merging (Idghaam) noon with Lam and Raa for Hafs.|
|meem_aal_imran|ميم آل عمران في قوله تعالى: {الم الله} وصلا|- `waqf` (`وقف`)<br>- `wasl_2` (`فتح الميم ومدها حركتين`)<br>- `wasl_6` (`فتح الميم ومدها ستة حركات`)<br>|`waqf` (`وقف`)|The ways to recite the word meem Aal Imran (الم الله) at connected recitation. `waqf`: Pause with a prolonged madd (elongation) of 6 harakat (beats). `wasl_2` Pronounce "meem" with fathah (a short "a" sound) and stretch it for 2 harakat. `wasl_6` Pronounce "meem" with fathah and stretch it for 6 harakat.|
|madd_yaa_alayn_alharfy|مقدار   المد اللازم الحرفي للعين|- `2`<br>- `4`<br>- `6`<br>|`6`| The length of Lzem Harfy of Yaa in letter Al-Ayen Madd "المد الحرفي اللازم لحرف العين" in surar: Maryam "مريم", AlShura "الشورى".|
|saken_before_hamz|الساكن قبل الهمز|- `tahqeek` (`تحقيق`)<br>- `general_sakt` (`سكت عام`)<br>- `local_sakt` (`سكت خاص`)<br>|`tahqeek` (`تحقيق`)|The ways of Hafs for saken before hamz. "The letter with sukoon before the hamzah (ء)".And it has three forms: full articulation (`tahqeeq`), general pause (`general_sakt`), and specific pause (`local_skat`).|
|sakt_iwaja|السكت عند عوجا في الكهف|- `sakt` (`سكت`)<br>- `waqf` (`وقف`)<br>- `idraj` (`إدراج`)<br>|`waqf` (`وقف`)|The ways to recite the word "عوجا" (Iwaja). `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).|
|sakt_marqdena|السكت عند مرقدنا  في يس|- `sakt` (`سكت`)<br>- `waqf` (`وقف`)<br>- `idraj` (`إدراج`)<br>|`waqf` (`وقف`)|The ways to recite the word "مرقدنا" (Marqadena) in Surat Yassen. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).|
|sakt_man_raq|السكت عند  من راق في القيامة|- `sakt` (`سكت`)<br>- `waqf` (`وقف`)<br>- `idraj` (`إدراج`)<br>|`sakt` (`سكت`)|The ways to recite the word "من راق" (Man Raq) in Surat Al Qiyama. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).|
|sakt_bal_ran|السكت عند  بل ران في  المطففين|- `sakt` (`سكت`)<br>- `waqf` (`وقف`)<br>- `idraj` (`إدراج`)<br>|`sakt` (`سكت`)|The ways to recite the word "بل ران" (Bal Ran) in Surat Al Motaffin. `sakt` means slight pause. `idraj` means not `sakt`. `waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idraj` (no sakt).|
|sakt_maleeyah|وجه  قوله تعالى {ماليه هلك} بالحاقة|- `sakt` (`سكت`)<br>- `waqf` (`وقف`)<br>- `idgham` (`إدغام`)<br>|`waqf` (`وقف`)|The ways to recite the word {ماليه هلك} in Surah Al-Ahqaf. `sakt` means slight pause. `idgham` Assimilation of the letter 'Ha' (ه) into the letter 'Ha' (ه) with complete assimilation.`waqf`:  means full pause, so we can not determine weither the reciter uses `sakt` or `idgham`.|
|between_anfal_and_tawba|وجه بين الأنفال والتوبة|- `waqf` (`وقف`)<br>- `sakt` (`سكت`)<br>- `wasl` (`وصل`)<br>|`waqf` (`وقف`)|The ways to recite end of Surah Al-Anfal and beginning of Surah At-Tawbah.|
|noon_and_yaseen|الإدغام والإظهار في النون عند الواو من قوله تعالى: {يس والقرآن}و {ن والقلم}|- `izhar` (`إظهار`)<br>- `idgham` (`إدغام`)<br>|`izhar` (`إظهار`)|Weither to merge noon of both: {يس} and {ن} with (و) "`idgham`" or not "`izhar`".|
|yaa_ataan| إثبات الياء وحذفها وقفا في قوله تعالى {آتان} بالنمل|- `wasl` (`وصل`)<br>- `hadhf` (`حذف`)<br>- `ithbat` (`إثبات`)<br>|`wasl` (`وصل`)|The affirmation and omission of the letter 'Yaa' in the pause of the verse {آتاني} in Surah An-Naml.`wasl`: means connected recitation without pasuding as (آتانيَ).`hadhf`: means deletion of letter (ي) at puase so recited as (آتان).`ithbat`: means confirmation reciting letter (ي) at puase as (آتاني).|
|start_with_ism|وجه البدأ بكلمة {الاسم} في سورة الحجرات|- `wasl` (`وصل`)<br>- `lism` (`لسم`)<br>- `alism` (`ألسم`)<br>|`wasl` (`وصل`)|The ruling on starting with the word {الاسم} in Surah Al-Hujurat.`lism` Recited as (لسم) at the beginning. `alism` Recited as (ألسم). ath the beginning`wasl`: means completing recitaion without paussing as normal, So Reciting is as (بئس لسم).|
|yabsut|السين والصاد في قوله تعالى: {والله يقبض ويبسط} بالبقرة|- `seen` (`سين`)<br>- `saad` (`صاد`)<br>|`seen` (`سين`)|The ruling on pronouncing `seen` (س) or `saad` (ص) in the verse {والله يقبض ويبسط} in Surah Al-Baqarah.|
|bastah|السين والصاد في قوله تعالى:  {وزادكم في الخلق بسطة} بالأعراف|- `seen` (`سين`)<br>- `saad` (`صاد`)<br>|`seen` (`سين`)|The ruling on pronouncing `seen` (س) or `saad` (ص ) in the verse {وزادكم في الخلق بسطة} in Surah Al-A'raf.|
|almusaytirun|السين والصاد في قوله تعالى {أم هم المصيطرون} بالطور|- `seen` (`سين`)<br>- `saad` (`صاد`)<br>|`saad` (`صاد`)|The pronunciation of `seen` (س) or `saad` (ص ) in the verse {أم هم المصيطرون} in Surah At-Tur.|
|bimusaytir|السين والصاد في قوله تعالى:  {لست عليهم بمصيطر} بالغاشية|- `seen` (`سين`)<br>- `saad` (`صاد`)<br>|`saad` (`صاد`)|The pronunciation of `seen` (س) or `saad` (ص ) in the verse {لست عليهم بمصيطر} in Surah Al-Ghashiyah.|
|tasheel_or_madd|همزة الوصل في قوله تعالى: {آلذكرين} بموضعي الأنعام و{آلآن} موضعي يونس و{آلله} بيونس والنمل|- `tasheel` (`تسهيل`)<br>- `madd` (`مد`)<br>|`madd` (`مد`)| Tasheel of Madd "وجع التسهيل أو المد" for 6 words in The Holy Quran: "ءالذكرين", "ءالله", "ءائن".|
|yalhath_dhalik|الإدغام وعدمه في قوله تعالى: {يلهث ذلك} بالأعراف|- `izhar` (`إظهار`)<br>- `idgham` (`إدغام`)<br>- `waqf` (`وقف`)<br>|`idgham` (`إدغام`)|The assimilation (`idgham`) and non-assimilation (`izhar`) in the verse {يلهث ذلك} in Surah Al-A'raf. `waqf`: means the rectier has paused on (يلهث)|
|irkab_maana|الإدغام والإظهار في قوله تعالى: {اركب معنا} بهود|- `izhar` (`إظهار`)<br>- `idgham` (`إدغام`)<br>- `waqf` (`وقف`)<br>|`idgham` (`إدغام`)|The assimilation and clear pronunciation in the verse {اركب معنا} in Surah Hud.This refers to the recitation rules concerning whether the letter "Noon" (ن) is assimilated into the following letter or pronounced clearly when reciting this specific verse. `waqf`: means the rectier has paused on (اركب)|
|noon_tamnna| الإشمام والروم (الاختلاس) في قوله تعالى {لا تأمنا على يوسف}|- `ishmam` (`إشمام`)<br>- `rawm` (`روم`)<br>|`ishmam` (`إشمام`)|The nasalization (`ishmam`) or the slight drawing (`rawm`) in the verse {لا تأمنا على يوسف}|
|harakat_daaf|حركة الضاد (فتح أو ضم) في قوله تعالى {ضعف} بالروم|- `fath` (`فتح`)<br>- `dam` (`ضم`)<br>|`fath` (`فتح`)|The vowel movement of the letter 'Dhad' (ض) (whether with `fath` or `dam`) in the word {ضعف} in Surah Ar-Rum.|
|alif_salasila|إثبات الألف وحذفها وقفا في قوله تعالى: {سلاسلا} بسورة الإنسان|- `hadhf` (`حذف`)<br>- `ithbat` (`إثبات`)<br>- `wasl` (`وصل`)<br>|`wasl` (`وصل`)|Affirmation and omission of the 'Alif' when pausing in the verse {سلاسلا} in Surah Al-Insan.This refers to the recitation rule regarding whether the final "Alif" in the word "سلاسلا" is pronounced (affirmed) or omitted when pausing (waqf) at this word during recitation in the specific verse from Surah Al-Insan. `hadhf`: means to remove alif (ا) during puase as (سلاسل) `ithbat`: means to recite alif (ا) during puase as (سلاسلا) `wasl` means completing the recitation as normal without pausing, so recite it as (سلاسلَ وأغلالا)|
|idgham_nakhluqkum|إدغام القاف في الكاف إدغاما ناقصا أو كاملا {نخلقكم} بالمرسلات|- `idgham_kamil` (`إدغام كامل`)<br>- `idgham_naqis` (`إدغام ناقص`)<br>|`idgham_kamil` (`إدغام كامل`)|Assimilation of the letter 'Qaf' into the letter 'Kaf,' whether incomplete (`idgham_naqis`) or complete (`idgham_kamil`), in the verse {نخلقكم} in Surah Al-Mursalat.|
|raa_firq|التفخيم والترقيق في راء {فرق} في الشعراء وصلا|- `waqf` (`وقف`)<br>- `tafkheem` (`تفخيم`)<br>- `tarqeeq` (`ترقيق`)<br>|`tafkheem` (`تفخيم`)|Emphasis and softening of the letter 'Ra' in the word {فرق} in Surah Ash-Shu'ara' when connected (wasl).This refers to the recitation rules concerning whether the letter "Ra" (ر) in the word "فرق"  is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when reciting the specific verse from Surah Ash-Shu'ara' in connected speech. `waqf`: means pasuing so we only have one way (tafkheem of Raa)|
|raa_alqitr|التفخيم والترقيق في راء {القطر} في سبأ وقفا|- `wasl` (`وصل`)<br>- `tafkheem` (`تفخيم`)<br>- `tarqeeq` (`ترقيق`)<br>|`wasl` (`وصل`)|Emphasis and softening of the letter 'Ra' in the word {القطر} in Surah Saba' when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "القطر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Saba'. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)|
|raa_misr|التفخيم والترقيق في راء {مصر} في يونس وموضعي يوسف والزخرف  وقفا|- `wasl` (`وصل`)<br>- `tafkheem` (`تفخيم`)<br>- `tarqeeq` (`ترقيق`)<br>|`wasl` (`وصل`)|Emphasis and softening of the letter 'Ra' in the word {مصر} in Surah Yunus, and in the locations of Surah Yusuf and Surah Az-Zukhruf when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "مصر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) at the specific pauses in these Surahs. `wasl`: means not pasuing so we only have one way (tafkheem of Raa)|
|raa_nudhur|التفخيم والترقيق  في راء {نذر} بالقمر وقفا|- `wasl` (`وصل`)<br>- `tafkheem` (`تفخيم`)<br>- `tarqeeq` (`ترقيق`)<br>|`tafkheem` (`تفخيم`)|Emphasis and softening of the letter 'Ra' in the word {نذر} in Surah Al-Qamar when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "نذر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Al-Qamar. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)|
|raa_yasr|التفخيم والترقيق في راء {يسر} بالفجر و{أن أسر} بطه والشعراء و{فأسر} بهود والحجر والدخان  وقفا|- `wasl` (`وصل`)<br>- `tafkheem` (`تفخيم`)<br>- `tarqeeq` (`ترقيق`)<br>|`tarqeeq` (`ترقيق`)|Emphasis and softening of the letter 'Ra' in the word {يسر} in Surah Al-Fajr when pausing (waqf).This refers to the recitation rules regarding whether the letter "Ra" (ر) in the word "يسر" is pronounced with emphasis (`tafkheem`) or softening (`tarqeeq`) when pausing at this word in Surah Al-Fajr. `wasl`: means not pasuing so we only have one way (tarqeeq of Raa)|


> **Note:** This documentation is auto generated using `python generate_moshaf_docs.py`.

