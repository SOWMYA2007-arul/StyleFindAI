# Experiment comparison

Generated 2026-11-22 11:55 UTC by `scripts/compare_experiments.py` from 11 stored experiments. Numbers are copied from each experiment's result.json.

Rows in the same table used the same query set and catalogue version and are directly comparable.

## Query set `029c3c83aa8f`, catalogue `384a65cb262d`

| experiment | model | N | ef | crop | multicrop | split | visual R@1 | hybrid R@1 | R@5 | R@10 | mAP | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261122-115220_cv-baseline_c50_candidates | cv-baseline | 50 | 100 | False | off | all | 72.3% | 75.7% | 91.7% | 95.7% | 0.828 | 59 | 77 |
| 20261122-115221_cv-baseline_c100_candidates | cv-baseline | 100 | 200 | False | off | all | 72.3% | 75.7% | 91.7% | 95.7% | 0.828 | 68 | 87 |
| 20261122-115221_cv-baseline_c200_candidates | cv-baseline | 200 | 400 | False | off | all | 72.3% | 75.7% | 91.7% | 95.7% | 0.828 | 84 | 104 |
| 20261122-115254_cv-baseline_c50_ef40_ef | cv-baseline | 50 | 40 | False | off | all | 72.3% | 75.7% | 91.7% | 95.7% | 0.828 | 58 | 78 |
| 20261122-115255_cv-baseline_c50_ef100_ef | cv-baseline | 50 | 100 | False | off | all | 72.3% | 75.7% | 91.7% | 95.7% | 0.828 | 57 | 77 |
| 20261122-115255_cv-baseline_c50_ef400_ef | cv-baseline | 50 | 400 | False | off | all | 72.3% | 75.7% | 91.7% | 95.7% | 0.828 | 57 | 77 |
| 20261122-115334_cv-baseline_c50_crop | cv-baseline | 50 | 100 | True | off | all | 73.7% | 74.0% | 91.7% | 95.0% | 0.820 | 112 | 159 |
| 20261122-115410_cv-baseline_c50_multicrop | cv-baseline | 50 | 100 | False | max_score | all | 72.3% | 82.7% | 95.0% | 98.0% | 0.883 | 99 | 123 |
| 20261122-115441_cv-baseline_c50_multicrop-mean | cv-baseline | 50 | 100 | False | mean_embedding | all | 74.3% | 78.0% | 92.3% | 96.7% | 0.845 | 86 | 111 |
| 20261122-115506_cv-baseline_c50_final | cv-baseline | 50 | 100 | False | off | all | 72.3% | 75.7% | 91.7% | 95.7% | 0.828 | 63 | 81 |

Best in this group by hybrid mAP (ties -> lower latency): **20261122-115410_cv-baseline_c50_multicrop**

## Query set `unknown (legacy)`, catalogue `unknown (legacy)` (legacy import)

| experiment | model | N | ef | crop | multicrop | split | visual R@1 | hybrid R@1 | R@5 | R@10 | mAP | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_cv-baseline_v1-code | cv-baseline | 50 (visual) / max(50, SEARCH_CANDIDATES) (hybrid) | max(100, 2k) | False | off | all | 69.3% | 71.3% | 90.3% | 94.0% | 0.795 | 45 | 59 |

Best in this group by hybrid mAP (ties -> lower latency): **legacy_cv-baseline_v1-code**

## How to choose

* Prefer the setting with the best hybrid mAP / R@1 on the **val** split, then confirm it on **test**.
* If two settings are equal within ~1 percentage point, choose the faster one (p95 latency).
* All queries are augmented catalogue photos unless an experiment says otherwise; they are easier than
  real customer photos.
