# Experiment 20261122-115506_cv-baseline_c50_final

* Date: 2026-11-22 11:55:06 UTC
* Model: `cv-baseline` version `cv-v2`
* Candidates: 50, hnsw.ef_search: 100, crop: False, multicrop: off
* Ranking weights: {'visual': 0.65, 'colour': 0.15, 'pattern': 0.1, 'attribute': 0.1}
* Dataset: 360 active products, version `384a65cb262d`, sources ['SYNTHETIC_TEST_FIXTURE']
* Queries: 300 from `dataset/evaluation/queries.csv` (version `029c3c83aa8f`, split `all`)
* Hardware: {'platform': 'Linux-6.18.44-fc-v726-x86_64-with-glibc2.39', 'processor': 'x86_64', 'cpu_count': 2, 'torch_threads': 2, 'ram_gb': 7.8, 'device': 'cpu', 'python': '3.13.18'}
* RSS memory after run: 686.1 MB

> **Synthetic fixture.** These numbers only show that the pipeline works.

> Queries are augmented catalogue photos (known answers). They are NOT real customer photos and
> must not be reported as real-world accuracy.

| ranking | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| visual | 72.3% | 89.0% | 94.7% | 0.800 |
| hybrid | 75.7% | 91.7% | 95.7% | 0.828 |

Candidate recall (true product inside the N candidates): 99.3%  
Category P@10 33.7%, colour C@10 74.2%, pattern C@10 36.6%  
Exact label: 41 queries, precision 100.0%, recall 13.7%  
Latency p50 63 ms, p95 81 ms (vector search p50 3.8 ms); exact-scan fallbacks: 0

## By scenario (hybrid)

| scenario | n | R@1 | R@10 | mAP | visual R@1 |
|---|---|---|---|---|---|
| angle | 60 | 98.3% | 100.0% | 0.992 | 93.3% |
| background | 60 | 60.0% | 90.0% | 0.716 | 60.0% |
| clutter | 60 | 83.3% | 96.7% | 0.879 | 83.3% |
| lighting | 60 | 56.7% | 93.3% | 0.694 | 50.0% |
| partial | 60 | 80.0% | 98.3% | 0.860 | 75.0% |

## Failure cases (true product not in hybrid top 10)

* `queries/syn-900190__background.jpg` -> top-1 `syn-900198`, visual rank 15, hybrid rank 32
* `queries/syn-900243__clutter.jpg` -> top-1 `syn-900048`, visual rank 9, hybrid rank 12
* `queries/syn-900034__lighting.jpg` -> top-1 `syn-900223`, visual rank None, hybrid rank None
* `queries/syn-900133__background.jpg` -> top-1 `syn-900219`, visual rank 20, hybrid rank 18
* `queries/syn-900204__background.jpg` -> top-1 `syn-900023`, visual rank 39, hybrid rank 30
* `queries/syn-900008__lighting.jpg` -> top-1 `syn-900062`, visual rank 17, hybrid rank 13
* `queries/syn-900082__partial.jpg` -> top-1 `syn-900314`, visual rank 6, hybrid rank 18
* `queries/syn-900303__lighting.jpg` -> top-1 `syn-900154`, visual rank 25, hybrid rank 19
* `queries/syn-900022__lighting.jpg` -> top-1 `syn-900029`, visual rank 42, hybrid rank 18
* `queries/syn-900228__clutter.jpg` -> top-1 `syn-900048`, visual rank 13, hybrid rank 11
