# Experiment 20261127-055615_fashion-clip_c50_crop-comparison

* Date: 2026-11-27 05:56:15 UTC
* Model: `fashion-clip` version `hf:patrickjohncyh/fashion-clip`
* Candidates: 50, hnsw.ef_search: 100, crop: True, multicrop: off
* Ranking weights: {'visual': 0.65, 'colour': 0.15, 'pattern': 0.1, 'attribute': 0.1}
* Dataset: 215 active products, version `41550197c441`, sources ['AI-generated dresses (StyleFind demo)']
* Queries: 535 from `dataset/evaluation/queries_generated.csv` (version `af91c1598d62`, split `val`)
* Hardware: {'platform': 'Windows-10-10.0.26300-SP0', 'processor': 'Intel64 Family 6 Model 189 Stepping 1, GenuineIntel', 'cpu_count': 8, 'torch_threads': 8, 'ram_gb': 15.7, 'device': 'cpu', 'python': '3.11.9'}
* RSS memory after run: 1282.1 MB

> Queries are augmented catalogue photos (known answers). They are NOT real customer photos and
> must not be reported as real-world accuracy.

| ranking | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| visual | 82.4% | 87.7% | 90.3% | 0.851 |
| hybrid | 84.9% | 91.0% | 92.9% | 0.874 |

Candidate recall (true product inside the N candidates): 95.1%  
Category P@10 100.0%, colour C@10 26.4%, pattern C@10 0.0%  
Exact label: 2 queries, precision 100.0%, recall 0.4%  
Latency p50 221 ms, p95 379 ms (vector search p50 7.9 ms); exact-scan fallbacks: 0

## By scenario (hybrid)

| scenario | n | R@1 | R@10 | mAP | visual R@1 |
|---|---|---|---|---|---|
| angle | 107 | 93.5% | 96.3% | 0.940 | 92.5% |
| background | 107 | 84.1% | 89.7% | 0.860 | 79.4% |
| clutter | 107 | 98.1% | 99.1% | 0.984 | 98.1% |
| lighting | 107 | 78.5% | 91.6% | 0.830 | 78.5% |
| partial | 107 | 70.1% | 87.9% | 0.757 | 63.6% |

## Failure cases (true product not in hybrid top 10)

* `queries_generated/ai-dress-0081__clutter.jpg` -> top-1 `ai-dress-0071`, visual rank None, hybrid rank None
* `queries_generated/ai-dress-0387__partial.jpg` -> top-1 `ai-dress-0290`, visual rank None, hybrid rank None
* `queries_generated/ai-dress-0511__lighting.jpg` -> top-1 `ai-dress-0178`, visual rank 8, hybrid rank 12
* `queries_generated/ai-dress-0398__partial.jpg` -> top-1 `ai-dress-0442`, visual rank 13, hybrid rank 21
* `queries_generated/ai-dress-0089__angle.jpg` -> top-1 `ai-dress-0290`, visual rank None, hybrid rank None
* `queries_generated/ai-dress-0513__background.jpg` -> top-1 `ai-dress-0291`, visual rank 18, hybrid rank 20
* `queries_generated/ai-dress-0262__background.jpg` -> top-1 `ai-dress-0104`, visual rank None, hybrid rank None
* `queries_generated/ai-dress-0318__partial.jpg` -> top-1 `ai-dress-0262`, visual rank None, hybrid rank None
* `queries_generated/ai-dress-0318__lighting.jpg` -> top-1 `ai-dress-0448`, visual rank None, hybrid rank None
* `queries_generated/ai-dress-0095__lighting.jpg` -> top-1 `ai-dress-0178`, visual rank 16, hybrid rank 15
