# Streaming FastConformer SN Models - Silence/Noise Dataset Analysis

| Model | from | noise in train | noise in augment | max_noise_input_seconds | [left, right, C] | Test per_phonemes | Test avg_per | QDAT per_phonemes | QDAT avg_per | eval_silence_per |
|-------|:-:|:-:|:-:|:-:|:-:|--:|--:|--:|--:|--:|
| sn-v6 | ar-pcd | Yes | No | 12.0 | [78, 12, 0] | 0.0457 | 0.0245 | 0.2805 | 0.1647 | 0.000599 |
| sn-v7 | ar-pcd | Yes | Yes | 12.0 | [78, 12, 0] | 0.0457 | 0.0196 | 0.2876 | 0.1640 | 0.000559 |
| sn-v8 | ar-pcd | No | Yes | 12.0 | [78, 12, 0] | 0.0733 | 0.0727 | 0.2537 | 0.1435 | N/A |
| sn-v9 | ar-pcd | No | No | 12.0 | [78, 12, 0] | 0.0732 | 0.0601 | 0.2782 | 0.1440 | N/A |
| v1 | ar-pcd | Yes | Yes | 30.0 | [78, 12, 0] | 0.0453 | 0.0195 | 0.2613 | 0.1562 | 0.000526 |
| c5_v1 | ar-pcd | Yes | Yes | 30.0 | [78, 12, 5] | 0.0530 | 0.0439 | 0.2350 | 0.1272 | 0.4962 |
| c5_v3 | en-str-m | Yes | Yes | 30.0 | [78, 12, 5] | 0.0160 | 0.0155 | 0.1865 | 0.1187 | 0.000173 |
| v6 | en-str-m | Yes | Yes | 30.0 | [78, 12, 0] | 0.0179 | 0.0116 | 0.1834 | 0.1155 | 0.000177 |

Where `from` is the pretrained NeMo checkpoint the encoder weights were loaded from:

- `ar-pcd`: [`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)
- `en-str-m`: [`nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
