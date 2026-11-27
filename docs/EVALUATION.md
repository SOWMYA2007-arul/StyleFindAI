# StyleFindAI — Evaluation Report

**Evaluation date:** 27 November 2026  
**Model:** FashionCLIP (`patrickjohncyh/fashion-clip`)  
**Dataset:** 215 AI-generated dress catalogue products  
**Evaluation queries:** 1,075 synthetic queries  
**Vector database:** PostgreSQL with pgvector  
**Selected configuration:** 50 candidates, garment cropping disabled, multi-crop disabled

---

## 1. Evaluation Objective

The objective of this evaluation is to measure the image-retrieval performance of StyleFindAI and identify a practical retrieval configuration.

The experiments evaluate:

- Visual similarity retrieval
- Hybrid re-ranking performance
- Recall at different values of K
- Mean Average Precision (mAP)
- Retrieval latency
- Candidate-count and HNSW search parameter effects
- Garment isolation and multi-crop retrieval effects

## 2. Dataset Description

The catalogue contains 215 AI-generated dress products.

Five synthetic query variations were generated for each product:

| Transformation | Queries |
|---|---:|
| Background variation | 215 |
| Angle variation | 215 |
| Partial-view variation | 215 |
| Clutter variation | 215 |
| Lighting variation | 215 |
| **Total** | **1,075** |

### Dataset Split

| Split | Queries |
|---|---:|
| Validation | 535 |
| Test | 540 |
| **Total** | **1,075** |

The split was assigned at product level so that queries belonging to the same product remain in the same split.

All generated query images were verified, with no missing images.

**Important:** All queries are augmented catalogue photos. They are easier than independent real customer photographs. These results measure performance on synthetic visual variations, not real-world customer-photo accuracy.

## 3. Evaluation Methodology

FashionCLIP generates image embeddings for query images. The system retrieves visually similar products through PostgreSQL with pgvector and then applies the existing hybrid re-ranking pipeline.

Both visual retrieval ranking and hybrid re-ranking are evaluated.

### Metrics

| Metric | Meaning |
|---|---|
| Recall@1 | Correct product appears first |
| Recall@5 | Correct product appears within the first five results |
| Recall@10 | Correct product appears within the first ten results |
| mAP | Mean Average Precision; with one relevant product per query, AP is the reciprocal of its rank if retrieved |
| p50 (ms) | Median query latency |
| p95 (ms) | 95th-percentile query latency |

The evaluation script also records category, colour, pattern and exact-match metrics. These additional metrics require careful interpretation because some attribute labels are not independently verified.

## 4. Validation Baseline

The full validation baseline used 535 queries, FashionCLIP and 50 retrieval candidates.

| Metric | Result |
|---|---:|
| Visual Recall@1 | 97.4% |
| Hybrid Recall@1 | 97.8% |
| Hybrid Recall@5 | 98.3% |
| Hybrid Recall@10 | 98.7% |
| mAP | 0.981 |
| p50 latency | 160 ms |
| p95 latency | 200 ms |
| Stable ordering | Yes |

## 5. Candidate-Count Comparison — Validation

Three candidate counts were compared using the validation split.

| Candidates | Visual R@1 | Hybrid R@1 | Hybrid R@5 | Hybrid R@10 | mAP | p50 (ms) | p95 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 97.4% | 97.8% | 98.3% | 98.7% | 0.981 | 132 | 155 |
| 100 | 97.4% | 97.8% | 98.3% | 98.5% | 0.981 | 143 | 167 |
| 200 | 97.4% | 97.9% | 98.5% | 98.7% | 0.982 | 162 | 191 |

The 200-candidate setting offered only marginal retrieval improvements and had higher measured latency.

The 50-candidate setting was selected as the practical validation configuration.

## 6. HNSW ef_search Comparison

The HNSW `ef_search` parameter was evaluated at 40, 100 and 200.

| ef_search | Visual R@1 | Hybrid R@1 | Hybrid R@10 | mAP | p50 (ms) | p95 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 97.4% | 97.8% | 98.7% | 0.981 | 196 | 292 |
| 100 | 97.4% | 97.8% | 98.7% | 0.981 | 196 | 291 |
| 200 | 97.4% | 97.8% | 98.7% | 0.981 | 196 | 293 |

No meaningful retrieval-quality difference was observed.

The default value of 100 was retained for consistency with the baseline and final test configuration.

## 7. Garment Isolation Experiment

Garment isolation was tested by enabling the cropping option.

| Metric | Without cropping | With cropping |
|---|---:|---:|
| Visual Recall@1 | 97.4% | 82.4% |
| Hybrid Recall@1 | 97.8% | 84.9% |
| Hybrid Recall@5 | 98.3% | 91.0% |
| Hybrid Recall@10 | 98.7% | 92.9% |
| mAP | 0.981 | 0.874 |
| p50 latency | 160 ms | 221 ms |
| p95 latency | 200 ms | 379 ms |

Garment isolation reduced retrieval performance and increased latency in this experiment.

It was therefore disabled in the selected configuration.

## 8. Multi-Crop Experiment

The max-score multi-crop strategy was evaluated separately.

| Metric | Normal retrieval | Multi-crop |
|---|---:|---:|
| Visual Recall@1 | 97.4% | 97.4% |
| Hybrid Recall@1 | 97.8% | 97.8% |
| Hybrid Recall@5 | 98.3% | 98.3% |
| Hybrid Recall@10 | 98.7% | 98.7% |
| mAP | 0.981 | 0.981 |
| p50 latency | 160 ms | 340 ms |
| p95 latency | 200 ms | 400 ms |

Multi-crop did not improve retrieval quality and increased latency. It was therefore disabled.

## 9. Final Test Results

The final test evaluation used 540 held-out synthetic queries.

| Metric | Test result |
|---|---:|
| Visual Recall@1 | 97.2% |
| Hybrid Recall@1 | 98.1% |
| Hybrid Recall@5 | 98.9% |
| Hybrid Recall@10 | 99.4% |
| mAP | 0.985 |
| p50 latency | 169 ms |
| p95 latency | 214 ms |
| Stable ordering | Yes |

### Validation vs Test

| Metric | Validation | Test |
|---|---:|---:|
| Visual Recall@1 | 97.4% | 97.2% |
| Hybrid Recall@1 | 97.8% | 98.1% |
| Hybrid Recall@5 | 98.3% | 98.9% |
| Hybrid Recall@10 | 98.7% | 99.4% |
| mAP | 0.981 | 0.985 |
| p50 latency | 160 ms | 169 ms |
| p95 latency | 200 ms | 214 ms |

The validation and test results are consistent on this synthetic benchmark.

### Test-Split Candidate-Count Confirmation

A separate candidate-count comparison was subsequently performed on all 540 held-out test queries.

| Candidates | Hybrid R@1 | Hybrid R@5 | Hybrid R@10 | mAP | p50 (ms) | p95 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 98.1% | 98.9% | 99.4% | 0.985 | 122 | 152 |
| 100 | 98.0% | 98.9% | 99.1% | 0.984 | 131 | 159 |
| 200 | 98.0% | 98.9% | 99.1% | 0.984 | 149 | 188 |

The 50-candidate configuration achieved the best measured retrieval quality and lowest measured p95 latency in this comparison.

Increasing the candidate count to 100 or 200 did not improve retrieval quality.

Therefore, 50 candidates is selected as the final retrieval configuration for this benchmark.

Latency measurements are machine-dependent and represent individual experiment runs.

## 10. Selected Configuration

The final practical configuration is:

| Component | Setting |
|---|---|
| Image encoder | FashionCLIP |
| Retrieval candidates | 50 |
| HNSW ef_search | 100 |
| Garment isolation | Disabled |
| Multi-crop retrieval | Disabled |
| Ranking | Existing hybrid re-ranking pipeline |

The selection prioritizes strong retrieval quality and lower measured latency.

## 11. Limitations

1. The catalogue contains 215 AI-generated dress products, not a large production retail catalogue.
2. Query images are synthetic transformations of catalogue images.
3. Synthetic transformations do not fully represent real customer photos, different cameras, complex backgrounds, occlusion or varied image quality.
4. Colour and pattern metadata partly originates from generation prompts and has not been independently verified. Related attribute metrics are provisional.
5. All products belong to the same broad dress category, so category precision is not a useful discriminator for this dataset.
6. Latency measurements depend on the local CPU, database configuration, caching and system load.
7. These results should not be interpreted as production-scale performance or real-world customer-photo accuracy.

## 12. Future Improvements

- Collect independent real-world dress photographs.
- Obtain human-verified product matches and attribute labels.
- Increase catalogue size and product diversity.
- Include additional clothing categories.
- Test different camera angles, distances, occlusion and lighting conditions.
- Evaluate on different hardware and deployment environments.
- Investigate improvements to garment segmentation.
- Compare additional embedding models through a separate, controlled indexing workflow.

## 13. Conclusion

StyleFindAI achieved **98.1% Hybrid Recall@1, 98.9% Hybrid Recall@5, 99.4% Hybrid Recall@10 and 0.985 mAP** on its 540-query synthetic test benchmark using FashionCLIP.

The experiments showed that the existing hybrid retrieval pipeline performs strongly on generated evaluation queries. Increasing the candidate count provided no meaningful improvement on the test split, while garment isolation and multi-crop retrieval increased latency without improving retrieval quality.

The current practical configuration uses FashionCLIP with 50 candidates, garment cropping disabled and multi-crop retrieval disabled.

Independent real-world photographs, verified ground-truth labels and testing on larger catalogues are needed before making stronger claims about real-world retrieval performance.

## 14. Experiment Reproducibility

Individual experiment results are stored in:

`dataset/evaluation/experiments/`

Each experiment contains:

- `result.json` — detailed metrics and configuration.
- `report.md` — individual experiment report.

The experiment index is:

`dataset/evaluation/experiments/index.jsonl`

To regenerate the comparison report:

```powershell
Push-Location ".\ai-service"

& "$env:LOCALAPPDATA\StyleFindAI-Test\Scripts\python.exe" ".\scripts\compare_experiments.py"

Pop-Location
```

**Note:** Running the comparison script regenerates the experiment comparison output. The manually maintained interpretation in `docs/EXPERIMENTS.md` may need to be restored afterward.
