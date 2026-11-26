# Models: compatibility, licences, hardware

Status column: **tested** = executed in the build environment; **code-path tested** = the class ran with
randomly initialised weights of the same architecture (weights could not be downloaded there); **untested** =
written against the library's documented API, not executed. Run the commands in README on your laptop to complete testing.

| Role | Model | Dims | Licence | Download | CPU feasible on 16 GB? | Status |
|---|---|---|---|---|---|---|
| Primary retrieval | FashionCLIP `patrickjohncyh/fashion-clip` (CLIP ViT-B/32 fine-tuned on fashion) | 512 | MIT (model card) | ~600 MB | Yes (~1.5 GB RAM) | code-path tested |
| Baseline | OpenCLIP ViT-B/32 `laion2b_s34b_b79k` | 512 | MIT | ~600 MB | Yes | untested (open_clip API) |
| Experimental | DINOv2 ViT-B/14 `facebook/dinov2-base` | 768 | Apache-2.0 | ~350 MB | Yes | untested (transformers API) |
| Sanity baseline | `cv-baseline` (HSV histogram + HOG + LBP, no neural net) | 512 | n/a | none | Yes | tested |
| Detection (optional) | Ultralytics YOLO (`yolo11n.pt` default; set `YOLO_WEIGHTS=yolo26n.pt` if your ultralytics version ships YOLO26) | - | **AGPL-3.0** | ~6 MB | Yes | untested |
| Segmentation (optional) | SAM 2.1 hiera-tiny | - | Apache-2.0 | ~150 MB | Slow but yes | untested |
| Attributes (optional) | Qwen2.5-VL-3B-Instruct | - | **Qwen Research licence (non-commercial)** | ~7.5 GB | Barely (bf16 ≈ 7 GB RAM, tens of seconds per image) → use Colab | untested |

## Important compatibility facts

* **Never mix embedding spaces.** Every vector row stores `model_key`, `model_version`, `dims`; queries filter on both,
  and the DB has a `CHECK (vector_dims(embedding) = dims)`.
* **Per-model HNSW indexes.** The `embedding` column is dimension-less so 512-d and 768-d models coexist; each model has a
  partial HNSW index on `embedding::vector(<dims>)`. pgvector's `hnsw.ef_search` defaults to 40, which would silently
  cut a top-50 query to 40 rows. The search sets `SET LOCAL hnsw.ef_search = max(100, 2k)`.
* For ~1,200 products PostgreSQL usually chooses an exact scan (cheaper than HNSW at this size), which is also exact. The
  index takes over automatically as the catalogue grows.
* **COCO YOLO weights have no garment classes** (only `person`, `handbag`, `tie`...). With them, the detector can only crop
  the person. For real garment boxes use weights trained on DeepFashion2-style classes; the code automatically prefers
  class names containing dress/skirt/top/sleeve/trousers/etc.
* **SAM 2 on Windows**: install with `pip install "git+https://github.com/facebookresearch/sam2.git"` (requires torch ≥ 2.5).
  If the optional CUDA extension fails to build, set `SAM2_BUILD_CUDA=0` before installing; CPU inference still works. WSL2 is the
  most reliable route.
* **Ultralytics is AGPL-3.0**: fine for a student project; a closed-source commercial deployment would need an Ultralytics licence.
* **Qwen2.5-VL-3B** is non-commercial. For commercial use evaluate Qwen2.5-VL-7B (Apache-2.0) on a GPU.
* Python 3.11 is supported by all listed libraries. The build environment used Python 3.13 and transformers 5.x; the code
  handles both transformers 4.x and 5.x output types.

## How the pipeline uses them

1. Search always runs **one** encoder (`AI_PRIMARY_MODEL`). Other encoders are only loaded by `generate_embeddings.py`,
   `evaluate.py`, `/ai/compare-models` or the admin "Generate" button. `AI_MAX_LOADED_MODELS=1` unloads the previous one first.
2. Attributes: colour always from pixels (k-means in CIELAB on the estimated garment region); category / pattern / sleeves /
   neckline by zero-shot text prompts when the encoder has a text side (FashionCLIP, OpenCLIP), kept only above a probability
   threshold, otherwise `unknown`. Without a text side, category comes from a weighted vote of the nearest neighbours.
3. Image + text search (`text_delta`): `q = normalise(img + λ·(text(target) − text(source)))`, where source describes what was
   predicted for the photo and target is the same description with the requested change. Requested attributes also get triple
   weight in the attribute score. This is a training-free baseline; a trained composed-retrieval model (e.g. CLIP4Cir on
   FashionIQ) would be the next step if evaluation shows it is needed.


## v2 additions

### Model management
* Each model is loaded on first use and at most `AI_MAX_LOADED_MODELS` stay in RAM (LRU unload).
* After loading, one probe image is embedded; if the vector size or `model_version` does not match the registry,
  the model is refused (`ModelUnavailableError`, HTTP 503) - embedding spaces are never mixed.
* Download / load failures return 503 with the approximate download size and RAM need; the model is not retried
  for 60 s so a broken download cannot slow every request.
* `FASHION_CLIP_REVISION` / `DINOV2_REVISION` pin a Hugging Face commit. Empty = the original version string
  (`hf:patrickjohncyh/fashion-clip`), so your existing embeddings stay valid. A pinned revision is a NEW
  `model_version` and needs `generate_embeddings.py`; old vectors stay in the DB but are never compared.
* `cv-baseline` is now `cv-v2` (its features depend on the improved garment mask).

### Garment detection choices (CPU, licences)
| option | classes | licence | CPU cost | status |
|---|---|---|---|---|
| `foreground` (default) | garment vs backdrop, no person/garment separation | own code | ~6 ms (+~35 ms GrabCut when cropping) | tested, ground-truth IoU in docs/sandbox_results_v2 |
| Ultralytics YOLO, COCO weights | `person` only | AGPL-3.0 | ~50-100 ms (n model) | code path + fallback tested; weights not run here |
| YOLO trained on DeepFashion2 (13 clothing classes, e.g. community checkpoints) | short/long sleeve top, dress, skirt, trousers... | check each checkpoint; DeepFashion2 data is non-commercial | ~50-100 ms | supported via `YOLO_WEIGHTS`; verify licence first |
| SAM 2.1 hiera-tiny (box prompt) | mask for the detected box | Apache-2.0 | ~0.5-1.5 s on CPU | optional, untested here |
| Qwen2.5-VL-3B | text attributes | Qwen Research (non-commercial) | 10-60 s/image on CPU, ~7 GB RAM (bf16) | optional, Colab recommended |

Indian garments (saree, lehenga, kurta) are not classes in DeepFashion2; a detector would still give a box
around the garment region (e.g. "long sleeve dress"), which is what the crop needs.

### Memory / latency (sandbox measurements, CPU)
| stage | cost |
|---|---|
| foreground mask (256 px) | ~6 ms |
| colour palette (CIEDE2000 merge) | ~12 ms |
| pattern descriptor + classifier | ~15 ms |
| GrabCut refinement (crop mode only) | ~35-40 ms |
| multi-crop `max_score` | +2 encoder passes + 2 vector queries (+~40 ms with cv-baseline; FashionCLIP ~+2 x embed time) |
