# Experiment 20261127-053609_fashion-clip_c50_smoke-test

* Date: 2026-11-27 05:36:09 UTC
* Model: `fashion-clip` version `hf:patrickjohncyh/fashion-clip`
* Candidates: 50, hnsw.ef_search: 100, crop: False, multicrop: off
* Ranking weights: {'visual': 0.65, 'colour': 0.15, 'pattern': 0.1, 'attribute': 0.1}
* Dataset: 215 active products, version `41550197c441`, sources ['AI-generated dresses (StyleFind demo)']
* Queries: 10 from `dataset/evaluation/queries_generated.csv` (version `af91c1598d62`, split `val`)
* Hardware: {'platform': 'Windows-10-10.0.26300-SP0', 'processor': 'Intel64 Family 6 Model 189 Stepping 1, GenuineIntel', 'cpu_count': 8, 'torch_threads': 8, 'ram_gb': 15.7, 'device': 'cpu', 'python': '3.11.9'}
* RSS memory after run: 1109.9 MB

> Queries are augmented catalogue photos (known answers). They are NOT real customer photos and
> must not be reported as real-world accuracy.

| ranking | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| visual | 100.0% | 100.0% | 100.0% | 1.000 |
| hybrid | 100.0% | 100.0% | 100.0% | 1.000 |

Candidate recall (true product inside the N candidates): 100.0%  
Category P@10 100.0%, colour C@10 27.0%, pattern C@10 0.0%  
Exact label: 0 queries, precision n/a, recall 0.0%  
Latency p50 166 ms, p95 484 ms (vector search p50 8.3 ms); exact-scan fallbacks: 0

## By scenario (hybrid)

| scenario | n | R@1 | R@10 | mAP | visual R@1 |
|---|---|---|---|---|---|
| angle | 2 | 100.0% | 100.0% | 1.000 | 100.0% |
| background | 2 | 100.0% | 100.0% | 1.000 | 100.0% |
| clutter | 2 | 100.0% | 100.0% | 1.000 | 100.0% |
| lighting | 2 | 100.0% | 100.0% | 1.000 | 100.0% |
| partial | 2 | 100.0% | 100.0% | 1.000 | 100.0% |

## Failure cases (true product not in hybrid top 10)

* none
