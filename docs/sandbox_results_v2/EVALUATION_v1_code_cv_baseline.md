# Retrieval evaluation report

Generated: 2026-11-22 05:01 UTC by `scripts/evaluate.py` (numbers are measured, not hand-written).

* Catalogue: 360 products; source(s): ['SYNTHETIC_TEST_FIXTURE']
* Queries: 300 from `dataset/evaluation/queries.csv` (transforms: angle, background, clutter, lighting, partial)
* Hardware: cpu; latency = full search pipeline per query with the model already loaded

> **Warning:** this run used the SYNTHETIC TEST FIXTURE, not real products. It verifies that the
> pipeline and metrics work; it says nothing about real-world accuracy. Re-run on the Kaggle catalogue.

## Summary

| model | ranking | R@1 | R@5 | R@10 | mAP | category P@10 | colour C@10 | pattern C@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| cv-baseline | visual | 69.3% | 88.0% | 93.0% | 0.776 | - | - | - | - | - |
| cv-baseline | hybrid | 71.3% | 90.3% | 94.0% | 0.795 | 32.0% | 72.9% | 30.0% | 45 | 59 |

## Exact-match labelling

| model | queries labelled exact | precision | recall |
|---|---|---|---|
| cv-baseline | 19 | 100.0% | 6.3% |

## By scenario (hybrid ranking)

### cv-baseline

| scenario | n | R@1 | R@5 | R@10 | mAP | visual-only R@1 |
|---|---|---|---|---|---|---|
| angle | 60 | 96.7% | 100.0% | 100.0% | 0.983 | 93.3% |
| background | 60 | 45.0% | 81.7% | 86.7% | 0.611 | 46.7% |
| clutter | 60 | 65.0% | 93.3% | 98.3% | 0.758 | 61.7% |
| lighting | 60 | 53.3% | 78.3% | 86.7% | 0.648 | 48.3% |
| partial | 60 | 96.7% | 98.3% | 98.3% | 0.976 | 96.7% |

Failure cases (true product not in hybrid top 10), first 10:

* `queries/syn-900067__lighting.jpg` -> top-1 `syn-900242`, visual rank 11, hybrid rank 15
* `queries/syn-900190__background.jpg` -> top-1 `syn-900198`, visual rank 16, hybrid rank 28
* `queries/syn-900034__lighting.jpg` -> top-1 `syn-900223`, visual rank None, hybrid rank None
* `queries/syn-900244__lighting.jpg` -> top-1 `syn-900259`, visual rank 36, hybrid rank 29
* `queries/syn-900204__background.jpg` -> top-1 `syn-900023`, visual rank 37, hybrid rank 40
* `queries/syn-900119__lighting.jpg` -> top-1 `syn-900223`, visual rank 18, hybrid rank 11
* `queries/syn-900268__background.jpg` -> top-1 `syn-900218`, visual rank 20, hybrid rank 14
* `queries/syn-900268__clutter.jpg` -> top-1 `syn-900184`, visual rank 17, hybrid rank 19
* `queries/syn-900008__partial.jpg` -> top-1 `syn-900049`, visual rank 36, hybrid rank 19
* `queries/syn-900008__lighting.jpg` -> top-1 `syn-900227`, visual rank 17, hybrid rank 15

## How to read this

* **Recall@K** - how often the exact source product appears in the first K results.
* **visual vs hybrid** - hybrid adds colour/pattern/attribute re-ranking; if hybrid is *worse*, lower the
  non-visual weights in `.env` (RANK_W_*).
* Queries are augmented catalogue photos, so they are easier than real street photos; DeepFashion2
  consumer-to-shop pairs (or your own photos) give a harder, more honest test.
