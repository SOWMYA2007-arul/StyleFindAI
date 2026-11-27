# Experiment 20261127-062217_fashion-clip_c100_final-test-candidate-comparison

* Date: 2026-11-27 06:22:17 UTC
* Model: `fashion-clip` version `hf:patrickjohncyh/fashion-clip`
* Candidates: 100, hnsw.ef_search: 200, crop: False, multicrop: off
* Ranking weights: {'visual': 0.65, 'colour': 0.15, 'pattern': 0.1, 'attribute': 0.1}
* Dataset: 215 active products, version `41550197c441`, sources ['AI-generated dresses (StyleFind demo)']
* Queries: 540 from `dataset/evaluation/queries_generated.csv` (version `af91c1598d62`, split `test`)
* Hardware: {'platform': 'Windows-10-10.0.26300-SP0', 'processor': 'Intel64 Family 6 Model 189 Stepping 1, GenuineIntel', 'cpu_count': 8, 'torch_threads': 8, 'ram_gb': 15.7, 'device': 'cpu', 'python': '3.11.9'}
* RSS memory after run: 1122.0 MB

> Queries are augmented catalogue photos (known answers). They are NOT real customer photos and
> must not be reported as real-world accuracy.

| ranking | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| visual | 97.2% | 98.0% | 98.0% | 0.977 |
| hybrid | 98.0% | 98.9% | 99.1% | 0.984 |

Candidate recall (true product inside the N candidates): 99.8%  
Category P@10 100.0%, colour C@10 21.4%, pattern C@10 0.0%  
Exact label: 11 queries, precision 100.0%, recall 2.0%  
Latency p50 131 ms, p95 159 ms (vector search p50 7.9 ms); exact-scan fallbacks: 0

## By scenario (hybrid)

| scenario | n | R@1 | R@10 | mAP | visual R@1 |
|---|---|---|---|---|---|
| angle | 108 | 100.0% | 100.0% | 1.000 | 100.0% |
| background | 108 | 90.7% | 95.4% | 0.926 | 88.0% |
| clutter | 108 | 100.0% | 100.0% | 1.000 | 100.0% |
| lighting | 108 | 100.0% | 100.0% | 1.000 | 100.0% |
| partial | 108 | 99.1% | 100.0% | 0.994 | 98.1% |

## Failure cases (true product not in hybrid top 10)

* `queries_generated/ai-dress-0395__background.jpg` -> top-1 `ai-dress-0221`, visual rank 93, hybrid rank 61
* `queries_generated/ai-dress-0451__background.jpg` -> top-1 `ai-dress-0396`, visual rank 12, hybrid rank 57
* `queries_generated/ai-dress-0336__background.jpg` -> top-1 `ai-dress-0290`, visual rank 23, hybrid rank 17
* `queries_generated/ai-dress-0044__background.jpg` -> top-1 `ai-dress-0127`, visual rank 31, hybrid rank 14
* `queries_generated/ai-dress-0410__background.jpg` -> top-1 `ai-dress-0199`, visual rank None, hybrid rank None
