# Sandbox results (v2 build) - synthetic fixture, cv-baseline encoder

These reports were produced while building v2 in an environment that could not download FashionCLIP or the
Kaggle dataset. They prove the scripts work and show the effect of the code changes on ONE fixed setup:

* catalogue: 360 synthetic test garments (`SYNTHETIC_TEST_FIXTURE`), not real products
* encoder: `cv-baseline` (classical OpenCV features), not FashionCLIP
* queries: 300 augmented catalogue photos (background / angle / partial / clutter / lighting)

They are **not** your results and must not be reported as FashionCLIP or real-customer accuracy.
Your own runs (README "Upgrade" steps) write `docs/EVALUATION.md`, `docs/EXPERIMENTS.md`,
`docs/INDEX_BENCHMARK.md` and `docs/WEIGHT_TUNING.md` from YOUR catalogue and model.

| file | what |
|---|---|
| EVALUATION_v1_code_cv_baseline.md | the old (v1) code on this setup - "before" |
| EVALUATION_final_cv_baseline.md | v2 code, default settings - "after" |
| EXPERIMENTS.md | all v2 experiments side by side (candidates, ef_search, crop, multi-crop) |
| INDEX_BENCHMARK.md | EXPLAIN ANALYZE / recall of HNSW vs exact, with and without filters |
| WEIGHT_TUNING.md | val/test weight tuning (result: keep current weights) |
| FOREGROUND_BENCHMARK.md | mask IoU vs ground truth on hard synthetic scenes |
