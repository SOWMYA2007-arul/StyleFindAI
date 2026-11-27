# Experiment 20261127-054826_fashion-clip_c200_candidate-comparison

* Date: 2026-11-27 05:48:26 UTC
* Model: `fashion-clip` version `hf:patrickjohncyh/fashion-clip`
* Candidates: 200, hnsw.ef_search: 400, crop: False, multicrop: off
* Ranking weights: {'visual': 0.65, 'colour': 0.15, 'pattern': 0.1, 'attribute': 0.1}
* Dataset: 215 active products, version `41550197c441`, sources ['AI-generated dresses (StyleFind demo)']
* Queries: 535 from `dataset/evaluation/queries_generated.csv` (version `af91c1598d62`, split `val`)
* Hardware: {'platform': 'Windows-10-10.0.26300-SP0', 'processor': 'Intel64 Family 6 Model 189 Stepping 1, GenuineIntel', 'cpu_count': 8, 'torch_threads': 8, 'ram_gb': 15.7, 'device': 'cpu', 'python': '3.11.9'}
* RSS memory after run: 1120.8 MB

> Queries are augmented catalogue photos (known answers). They are NOT real customer photos and
> must not be reported as real-world accuracy.

| ranking | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| visual | 97.4% | 98.1% | 98.3% | 0.978 |
| hybrid | 97.9% | 98.5% | 98.7% | 0.982 |

Candidate recall (true product inside the N candidates): 100.0%  
Category P@10 100.0%, colour C@10 25.4%, pattern C@10 0.0%  
Exact label: 10 queries, precision 100.0%, recall 1.9%  
Latency p50 162 ms, p95 191 ms (vector search p50 8.1 ms); exact-scan fallbacks: 0

## By scenario (hybrid)

| scenario | n | R@1 | R@10 | mAP | visual R@1 |
|---|---|---|---|---|---|
| angle | 107 | 100.0% | 100.0% | 1.000 | 100.0% |
| background | 107 | 92.5% | 93.5% | 0.928 | 86.9% |
| clutter | 107 | 99.1% | 100.0% | 0.995 | 100.0% |
| lighting | 107 | 100.0% | 100.0% | 1.000 | 100.0% |
| partial | 107 | 98.1% | 100.0% | 0.989 | 100.0% |

## Failure cases (true product not in hybrid top 10)

* `queries_generated/ai-dress-0513__background.jpg` -> top-1 `ai-dress-0228`, visual rank 63, hybrid rank 80
* `queries_generated/ai-dress-0262__background.jpg` -> top-1 `ai-dress-0160`, visual rank 14, hybrid rank 83
* `queries_generated/ai-dress-0063__background.jpg` -> top-1 `ai-dress-0410`, visual rank 15, hybrid rank 28
* `queries_generated/ai-dress-0049__background.jpg` -> top-1 `ai-dress-0537`, visual rank 38, hybrid rank 91
* `queries_generated/ai-dress-0482__background.jpg` -> top-1 `ai-dress-0127`, visual rank 118, hybrid rank 114
* `queries_generated/ai-dress-0359__background.jpg` -> top-1 `ai-dress-0248`, visual rank 7, hybrid rank 13
* `queries_generated/ai-dress-0199__background.jpg` -> top-1 `ai-dress-0446`, visual rank 26, hybrid rank 46
