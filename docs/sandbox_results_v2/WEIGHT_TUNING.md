# Ranking-weight tuning (cv-baseline, N=50)

Generated 2026-11-22T11:55:37.044952+00:00. Chosen on val, reported on test.

| weights | split | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|---|
| current {'visual': 0.65, 'colour': 0.15, 'pattern': 0.1, 'attribute': 0.1} | val | 78.3% | 91.3% | 97.4% | 0.839 |
| current {'visual': 0.65, 'colour': 0.15, 'pattern': 0.1, 'attribute': 0.1} | test | 74.1% | 91.9% | 94.6% | 0.821 |
| best on val {'visual': 0.75, 'colour': 0.05, 'pattern': 0.1, 'attribute': 0.1} | val | 80.9% | 93.9% | 96.5% | 0.859 |
| best on val {'visual': 0.75, 'colour': 0.05, 'pattern': 0.1, 'attribute': 0.1} | test | 75.1% | 90.8% | 95.1% | 0.826 |

**Recommendation:** keep current weights (no reliable improvement on the test split).

