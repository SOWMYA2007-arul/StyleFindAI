# Experiment comparison

Generated 2026-11-27 07:40 UTC by `scripts/compare_experiments.py` from 14 stored experiments. Numbers are copied from each experiment's result.json.

Rows in the same table used the same query set and catalogue version and are directly comparable.

## Query set `af91c1598d62`, catalogue `41550197c441`

| experiment | model | N | ef | crop | multicrop | split | visual R@1 | hybrid R@1 | R@5 | R@10 | mAP | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261127-053609_fashion-clip_c50_smoke-test | fashion-clip | 50 | 100 | False | off | val | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 166 | 484 |
| 20261127-053834_fashion-clip_c50_validation-baseline | fashion-clip | 50 | 100 | False | off | val | 97.4% | 97.8% | 98.3% | 98.7% | 0.981 | 160 | 200 |
| 20261127-054825_fashion-clip_c100_candidate-comparison | fashion-clip | 100 | 200 | False | off | val | 97.4% | 97.8% | 98.3% | 98.5% | 0.981 | 143 | 167 |
| 20261127-054825_fashion-clip_c50_candidate-comparison | fashion-clip | 50 | 100 | False | off | val | 97.4% | 97.8% | 98.3% | 98.7% | 0.981 | 132 | 155 |
| 20261127-054826_fashion-clip_c200_candidate-comparison | fashion-clip | 200 | 400 | False | off | val | 97.4% | 97.9% | 98.5% | 98.7% | 0.982 | 162 | 191 |
| 20261127-055228_fashion-clip_c50_ef100_ef-comparison | fashion-clip | 50 | 100 | False | off | val | 97.4% | 97.8% | 98.3% | 98.7% | 0.981 | 196 | 291 |
| 20261127-055228_fashion-clip_c50_ef200_ef-comparison | fashion-clip | 50 | 200 | False | off | val | 97.4% | 97.8% | 98.3% | 98.7% | 0.981 | 196 | 293 |
| 20261127-055228_fashion-clip_c50_ef40_ef-comparison | fashion-clip | 50 | 40 | False | off | val | 97.4% | 97.8% | 98.3% | 98.7% | 0.981 | 196 | 292 |
| 20261127-055615_fashion-clip_c50_crop-comparison | fashion-clip | 50 | 100 | True | off | val | 82.4% | 84.9% | 91.0% | 92.9% | 0.874 | 221 | 379 |
| 20261127-060140_fashion-clip_c50_multicrop-max-score | fashion-clip | 50 | 100 | False | max_score | val | 97.4% | 97.8% | 98.3% | 98.7% | 0.981 | 340 | 400 |
| 20261127-060946_fashion-clip_c50_final-test | fashion-clip | 50 | 100 | False | off | test | 97.2% | 98.1% | 98.9% | 99.4% | 0.985 | 169 | 214 |
| 20261127-062217_fashion-clip_c100_final-test-candidate-comparison | fashion-clip | 100 | 200 | False | off | test | 97.2% | 98.0% | 98.9% | 99.1% | 0.984 | 131 | 159 |
| 20261127-062217_fashion-clip_c200_final-test-candidate-comparison | fashion-clip | 200 | 400 | False | off | test | 97.2% | 98.0% | 98.9% | 99.1% | 0.984 | 149 | 188 |
| 20261127-062217_fashion-clip_c50_final-test-candidate-comparison | fashion-clip | 50 | 100 | False | off | test | 97.2% | 98.1% | 98.9% | 99.4% | 0.985 | 122 | 152 |

## Interpretation and selected configuration

The 10-query smoke test is excluded from model selection because it is only a pipeline verification run.

The validation experiments support using 50 candidates: it achieved 97.8% hybrid Recall@1 and 98.7% Recall@10, with lower p95 latency than the 100- and 200-candidate settings in that comparison.

The selected configuration was then confirmed on all 540 held-out test queries:

- Hybrid Recall@1: 98.1%
- Hybrid Recall@5: 98.9%
- Hybrid Recall@10: 99.4%
- mAP: 0.985
- p50 latency: 122 ms
- p95 latency: 152 ms

The final selected configuration is FashionCLIP with 50 candidates, ef_search=100, cropping disabled, and multi-crop disabled.

Latency measurements are machine-dependent and represent individual experiment runs. All queries are generated catalogue-image augmentations, not independent customer photographs.
## How to choose

* Prefer the setting with the best hybrid mAP / R@1 on the **val** split, then confirm it on **test**.
* If two settings are equal within ~1 percentage point, choose the faster one (p95 latency).
* All queries are augmented catalogue photos unless an experiment says otherwise; they are easier than
  real customer photos.
