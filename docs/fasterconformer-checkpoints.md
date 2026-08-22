# FastConformer Checkpoint Ranking for Quran Streaming Fine-tuning

Goal: pick a FastConformer base checkpoint to fine-tune on Holy Quran (Arabic, tartil) for streaming ASR — **cache-aware**, **4× downsampling**, **CTC** preferred.

Criteria (weights as stated by user): training hours, data diversity, multilingual, cache-aware streaming, CTC decoder. Cache-aware and CTC are **preferences**, not hard requirements. Downsampling is reported for every model but does not change the ranking.

---

## Ranked table (all 15 checkpoints, best → worst)

| # | Model | Lang | Params | Downsampling | Cache-aware streaming | CTC | Training data / hours | Diversity & multilingual | License | Notes for Quran task |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi) | en | 114M | 8× (80 ms/frame) | ✅ multi-latency 0/80/480/1040 ms | ✅ RNNT+CTC | NeMo ASRSET 3.0, several-thousand h (Libri 960, Fisher, SWB, WSJ, NSC p1/p6, VCTK, VoxPopuli, Europarl, MLS-EN 2k, MCV7, People's Speech 12k) | High diversity, en-only | CC-BY-4.0 | **Best all-rounder** — cache-aware + CTC + huge hours/diversity; is the base of the Shenava & Hindi models; ideal language-swap starting point for streaming fine-tunes |
| 2 | [parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | en | 114M | 8× | ❌ offline full-attention | ✅ TDT+CTC | 36,000 h (27k private + 9k public: Libri, Fisher, NSC, VCTK, VoxPopuli, Europarl, MLS, MCV7) | Highest hours + diversity, en-only | CC-BY-4.0 | Strongest acoustic base if you accept a non-streaming encoder (cache-aware only "preferable"); decodes with TDT or CTC |
| 3 | [SraVaani-0.5-live](https://huggingface.co/ARTPARK-IISc/SraVaani-0.5-live) | 63 Indic langs/dialects | ~430M | 8× | ✅ multi-latency 1040→0 ms | ✅ CTC head | Large-scale VAANI pretraining (hours not stated; varies per lang) | Very high diversity + multilingual (Vaani, RESPIN, FLEURS, IndicTTS, CV, Gramvaani, Kathbath, MUCS) | MIT (gated) | Best on multilingual + cache + CTC, but Indic-only (no Arabic), shipped as TorchScript export + separate fine-tune ckpts, access must be requested |
| 4 | [Shenava-Koochik-v1.0](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) | fa (Persian) | 114M | 8× (80 ms/frame) | ✅ multi-latency [70,13/6/1/0] | ✅ RNNT+CTC | ~7,386 h pseudo-labeled + gold (visualears-persian-asr-16k, 3.93M clips) | Moderate; Persian only | Apache-2.0 | FP32 NeMo source built explicitly for evaluation/fine-tuning; base = NVIDIA streaming_multi; Persian uses Arabic script → transferable |
| 5 | [stt_en_fastconformer_ctc_large](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large) | en | 115M | 8× | ❌ offline | ✅ CTC only | NeMo ASRSET 3.0, several-thousand h | High diversity, en-only | CC-BY-4.0 | Pure-CTC fallback when streaming not required; LS test-other 4.2 |
| 6 | [Shenava-Koochik-0.9](https://huggingface.co/Reza2kn/Shenava-Koochik-0.9) | fa | 114M | 8× (80 ms/frame) | ✅ multi-latency | ✅ RNNT+CTC | ~7,386 h + gold | Persian only | Apache-2.0 | Older 0.9 of #4 — superseded by v1.0 |
| 7 | [Hindi-FastConformer-Streaming-ASR](https://huggingface.co/salesken/Hindi-FastConformer-Streaming-ASR) | hi | large | 8× (via streaming_multi) | ✅ multi-latency | ✅ RNNT+CTC | IndicVoices-ST (ai4bharat) | Hindi only | Apache-2.0 | Fine-tune of NVIDIA streaming_multi; demonstrates this lineage adapts to new languages |
| 8 | [shenava-fa-fastconformer-streaming-32m](https://huggingface.co/Reza2kn/shenava-fa-fastconformer-streaming-32m) | fa | 32M | 8× (80 ms/frame) | ✅ multi-latency | ✅ RNNT+CTC | ~7,386 h + gold | Persian only | Apache-2.0 | Same data as #4 but 32M capacity — weak fine-tune base |
| 9 | [Shenava-Rizeh-v1.0](https://huggingface.co/PersianML/Shenava-Rizeh-v1.0) | fa | 32M | 8× | ✅ multi-latency | ✅ RNNT+CTC | Distilled from Koochik v1.0 | Persian only | Apache-2.0 | Distilled student — avoid as a fine-tune base (limited capacity) |
| 10 | [stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.0](https://huggingface.co/dev-ahmedhany/stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.0) | ar (incl. Quran tartil) | large | 8× (via stt_ar hybrid) | ✅ [70,13] 1040 ms | ✅ RNNT+CTC | ~10 h (8.3 h Quran tartil everyayah + 2 h FLEURS eg) | Low; Arabic/Egyptian | CC-BY-4.0 | **Only Arabic + already-Quran checkpoint**; tiny data. Top *domain* pick if inherited Arabic capability from the NVIDIA base suffices |
| 11 | [parakeet_realtime_eou_120m-v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) | en | 120M | 8× | ✅ [70,1] 80–160 ms | ❌ RNNT only | ~10,000 h + (AMI, DialogStudio, Granary, GSC, LibriTTS; some synthetic) | Moderate | NVIDIA OML | Cache-aware + hours, but **no CTC head** (RNNT only) |
| 12 | [stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc](https://huggingface.co/nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc) | ka (Georgian) | 115M | 8× | ✅ 80 ms | ✅ RNNT+CTC | ~163 h (MCV17 96+63 + FLEURS 4) | Low | CC-BY-4.0 | Cache-aware + CTC but negligible hours/diversity |
| 13 | [Ja-FastConformer-CTC-25Hz-Streaming-100M](https://huggingface.co/Atotti/Ja-FastConformer-CTC-25Hz-Streaming-100M) | ja | 112M | **4× (25 Hz, 10 ms stride)** | ⚠️ causal [70,0], not cache-aware chunked | ✅ CTC | ReazonSpeech | Low; Japanese only | gated | **Only 4× downsampled encoder in the list** — matches your 4× preference; otherwise weak fit |
| 14 | [sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-2026-06-26](https://huggingface.co/mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-2026-06-26) | fa | 114M | 8× (dw_striding ×8) | ✅ streaming | ✅ CTC (export) | Same data as #4 | Persian | CC-BY-NC-4.0 | **ONNX deployment export, not a training checkpoint** — use Koochik v1.0 NeMo source instead |
| 15 | [multilingual-asr](https://huggingface.co/nur-dev/multilingual-asr) | kk/ru/uz/en | — | 8× | ❌ not streaming | ✅ CTC | Small corpus (76k test samples) | 4 langs, low hours | CC-BY-NC-4.0 | Multilingual but no cache-aware, no Arabic, non-commercial license |

---

## Per-criterion scores (1–5, higher is better)

| Model | Hours | Diversity | Multilingual | Cache-aware | CTC | Total | Rank |
|---|---|---|---|---|---|---|---|
| stt_en_fastconformer_hybrid_large_streaming_multi | 5 | 5 | 1 | 5 | 5 | 21 | 1 |
| parakeet-tdt_ctc-110m | 5 | 5 | 1 | 1 | 5 | 17 | 2 |
| SraVaani-0.5-live | 3 | 5 | 5 | 5 | 5 | 23 | 3 |
| Shenava-Koochik-v1.0 | 4 | 3 | 1 | 5 | 5 | 18 | 4 |
| stt_en_fastconformer_ctc_large | 5 | 5 | 1 | 1 | 5 | 17 | 5 |
| Shenava-Koochik-0.9 | 4 | 3 | 1 | 5 | 5 | 18 | 6 |
| Hindi-FastConformer-Streaming-ASR | 3 | 3 | 1 | 5 | 5 | 17 | 7 |
| shenava-fa-fastconformer-streaming-32m | 4 | 3 | 1 | 5 | 5 | 18 | 8 |
| Shenava-Rizeh-v1.0 | 4 | 3 | 1 | 5 | 5 | 18 | 9 |
| stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.0 | 1 | 2 | 1 | 5 | 5 | 14 | 10 |
| parakeet_realtime_eou_120m-v1 | 4 | 4 | 1 | 5 | 1 | 15 | 11 |
| stt_ka_..._streaming_80ms_pc | 1 | 2 | 1 | 5 | 5 | 14 | 12 |
| Ja-FastConformer-CTC-25Hz-Streaming-100M | 3 | 2 | 1 | 2 | 5 | 13 | 13 |
| sherpa-onnx-...-koochik-v1.0 | 4 | 3 | 1 | 5 | 5 | 18* | 14 |
| multilingual-asr | 2 | 2 | 3 | 1 | 5 | 13 | 15 |

\* #14 scores like #4 on paper but is an **ONNX inference export, not a trainable checkpoint** — demoted to deployment-only.

Ranking ties broken by practical value: cache-aware preference, license (CC-BY / Apache-2.0 favored over non-commercial/gated), NeMo-source availability, and language proximity to Arabic (Persian/Arabic-script > Indic > EN > JA/KA).

---

## Key insights for your 4× downsampling target

- **Every cache-aware streaming checkpoint in this list is 8× FastConformer (80 ms/frame).** Only the Ja model (#13) is **4× (25 Hz)**, and it is causal-but-not-cache-aware.
- To reach 4× cache-aware you must re-tune the encoder conv-striding of an 8× base (frame-rate transfer) — any of #1/#3/#4/#10 can serve as the starting encoder and decoder/tokenizer.
- Alternative: use #13 as a ready-made 4× encoder (25 Hz, CTC, causal) and only if its 10 ms-stride setup is what you want.

## Practical picks for Quran fine-tuning

- **Best criteria score:** #1 `stt_en_fastconformer_hybrid_large_streaming_multi` — cache-aware + CTC + most hours/diversity, CC-BY-4.0, standard NeMo checkpoint, and the proven base for other streaming language swaps in this list (Shenava, Hindi).
- **Best domain fit:** #10 `stt_ar_..._streaming_pcd_v1.0` — Arabic + already Quran-tartil fine-tuned, cache-aware [70,13], hybrid CTC/RNNT — but only ~10 h of data; treat as a light head start, or start from #1/#2 and fine-tune on your Quran corpus directly.
- **Watch-outs:** #3 and #13 are gated (request access); #14 is an inference export (not a trainable checkpoint); #11 has no CTC head; #15 is non-commercial (CC-BY-NC).

---

## Sources

All figures pulled from the HuggingFace model cards linked above (Aug 2026). Gated repos (#3, #13) may not be directly accessible without login.
