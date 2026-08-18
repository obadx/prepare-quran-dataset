# Streaming FastConformer SN Models - Silence/Noise Dataset Analysis

| Model | noise in train | noise in augment | max_noise_input_seconds | [left, right, C] | Test per_phonemes | Test avg_per | QDAT per_phonemes | QDAT avg_per | eval_silence_per |
|-------|:-:|:-:|:-:|:-:|--:|--:|--:|--:|--:|
| sn-v6 | Yes | No | 12.0 | [78, 12, 0] | 0.0457 | 0.0245 | 0.2805 | 0.1647 | 0.000599 |
| sn-v7 | Yes | Yes | 12.0 | [78, 12, 0] | 0.0457 | 0.0196 | 0.2876 | 0.1640 | 0.000559 |
| sn-v8 | No | Yes | 12.0 | [78, 12, 0] | 0.0733 | 0.0727 | 0.2537 | 0.1435 | N/A |
| sn-v9 | No | No | 12.0 | [78, 12, 0] | 0.0732 | 0.0601 | 0.2782 | 0.1440 | N/A |
| v1 | Yes | Yes | 30.0 | [78, 12, 0] | 0.0453 | 0.0195 | 0.2613 | 0.1562 | 0.000526 |
| c5_v1 | Yes | Yes | 30.0 | [78, 12, 5] | 0.0530 | 0.0439 | 0.2350 | 0.1272 | 0.4962 |
