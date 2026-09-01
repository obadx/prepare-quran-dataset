# Streaming FastConformer SN Models - Silence/Noise Dataset Analysis

| Model | from | pre-enc | noise in train | noise in augment | max_noise_input_seconds | [left, right, C] | Test per_phonemes | Test avg_per | QDAT per_phonemes | QDAT avg_per | eval_silence_per |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|--:|--:|--:|--:|--:|
| sn-v6 | ar-pcd | bf16 | Yes | No | 12.0 | [78, 12, 0] | 0.0457 | 0.0245 | 0.2805 | 0.1647 | 0.000599 |
| sn-v7 | ar-pcd | bf16 | Yes | Yes | 12.0 | [78, 12, 0] | 0.0457 | 0.0196 | 0.2876 | 0.1640 | 0.000559 |
| sn-v8 | ar-pcd | bf16 | No | Yes | 12.0 | [78, 12, 0] | 0.0733 | 0.0727 | 0.2537 | 0.1435 | N/A |
| sn-v9 | ar-pcd | bf16 | No | No | 12.0 | [78, 12, 0] | 0.0732 | 0.0601 | 0.2782 | 0.1440 | N/A |
| v1 | ar-pcd | bf16 | Yes | Yes | 30.0 | [78, 12, 0] | 0.0453 | 0.0195 | 0.2613 | 0.1562 | 0.000526 |
| c5_v1 | ar-pcd | bf16 | Yes | Yes | 30.0 | [78, 12, 5] | 0.0530 | 0.0439 | 0.2350 | 0.1272 | 0.4962 |
| c5_v3 | en-str-m | bf16 | Yes | Yes | 30.0 | [78, 12, 5] | 0.0160 | 0.0155 | 0.1865 | 0.1187 | 0.000173 |
| v6 | en-str-m | bf16 | Yes | Yes | 30.0 | [78, 12, 0] | 0.0179 | 0.0116 | 0.1834 | 0.1155 | 0.000177 |
| c5_v4 | en-str-m | fp32 | Yes | Yes | 30.0 | [78, 12, 5] | 0.0190 | 0.0072 | 0.1792 | 0.1045 | 0.000140 |
| v7 | en-str-m | fp32 | Yes | Yes | 30.0 | [78, 12, 0] | 0.0216 | 0.0080 | 0.2018 | 0.1220 | 0.000151 |
| c2_v1 | en-str-m | fp32 | Yes | Yes | 30.0 | [78, 12, 2] | 0.0176 | 0.0069 | 0.2357 | 0.1607 | 0.000076 |

Where `from` is the pretrained NeMo checkpoint the encoder weights were loaded from:

- `ar-pcd`: [`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)
- `en-str-m`: [`nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)

`pre-enc` is the dtype used by the encoder's pre-encode front-end (the subsampling
convolutions applied to the log-mel features), controlled by the model kwarg
`fp32_pre_encode`. The default is `bf16` (`fp32_pre_encode: false`). With `fp32`, the
front-end runs in float32 with autocast disabled — since log-mel features' dynamic range
makes these convolutions numerically fragile under bf16 — and its output is cast back to
bf16 so all downstream layers are unaffected.

## Streaming Results

| Model | from | pre-enc | [left, right, C] | Test per_phonemes | Test avg_per | QDAT per_phonemes | QDAT avg_per |
|-------|:-:|:-:|:-:|--:|--:|--:|--:|
| c5_v3 | en-str-m | bf16 | [78, 12, 5] | 0.0524 | 0.0287 | 0.2544 | 0.1121 |
| v6 | en-str-m | bf16 | [78, 12, 0] | 0.0180 | 0.0118 | 0.1818 | 0.1149 |
| c5_v4 | en-str-m | fp32 | [78, 12, 5] | 0.0392 | 0.0222 | 0.2025 | 0.1031 |
| v7 | en-str-m | fp32 | [78, 12, 0] | 0.0217 | 0.0080 | 0.2002 | 0.1214 |
| c2_v1 | en-str-m | fp32 | [78, 12, 2] | 0.0448 | 0.0230 | 0.2389 | 0.1379 |

## Muallem-annotated-v3 + tlog

> Default att_context_size: `[78, 12, 0]`

| Model | off-T-phonemes | str-T-phonemes | off-Q-phonemes | str-Q-phonemes |
|-------|--:|--:|--:|--:|
| base-v1 | 0.00211 | 0.00211 | 0.0572 | 0.0572 |

* **T**: for Testset
* **Q**: for qdat
* **off**: offline
* **str**: streaming
