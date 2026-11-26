# Garment detection and segmentation (v3)

This is the part of the AI service that answers: *which pixels of this photo are the garment the shopper
wants to find?* It is used when a search asks for **"Focus on the garment"** (`crop=true`) and by the new
`POST /ai/segment` endpoint. Search without `crop` is unchanged (bit-for-bit the same results as v2).

Nothing here touches login, cart, checkout, orders or any other shop feature.

---

## 1. How a photo flows through the pipeline

```
upload ──► quality check ─► foreground mask (weight-free, ~10 ms) ─► scene analysis
                                                                      │
          ┌───────────────────────────┬───────────────────────────────┼──────────────────────────┐
          ▼                           ▼                               ▼                          ▼
   product_only                    person                        multi_garment               difficult
 (garment on a plain or      (someone wearing it)           (several pieces, no person)  (cluttered / low contrast)
  gradient backdrop)                                                                       
 foreground mask only        human parser ─► clothing YOLO   clothing YOLO ─► colour +     person segmenter can
 + speck / hand removal      ─► person mask (YOLO-seg or     connectivity split            re-route to "person";
 NO models are run           GrabCut) ─► skin/face/hair                                     else GrabCut + split
                             removal  (+ pose, optional)
          └───────────────────────────┴───────────────┬───────────────┴──────────────────────────┘
                                                      ▼
                 masks to full resolution (guided-filter boundary refinement, keeps straps / tassels)
                                                      ▼
                 choose target garment (your `target`, else the most prominent) ─► SAM 2 refine
                 (optional, only for a doubtful target, never for product photos)
                                                      ▼
                 confidence ≥ MIN_SEGMENTATION_CONFIDENCE ?  yes ─► crop on white for the search model
                                                             no  ─► full image + the reason why
```

Rules the code follows:

* **Cheapest adequate method first.** A catalogue-style product photo never loads YOLO, the parser, SAM 2
  or pose (a test proves it). Expensive models are only tried when the photo needs them *and* you enabled
  them in `.env`.
* **Every optional model is optional.** Missing package, missing weights, a crash or a timeout
  (`SEGMENTATION_TIMEOUT_S`) is recorded in `attempts` and the next method is used. Search never breaks.
* **No silent bad masks.** Each result has a confidence. Below `MIN_SEGMENTATION_CONFIDENCE` (0.4) the
  full image is used and `fallback_reason` says why.
* **No invented pixels.** A hand in front of a garment is *removed* from the mask (and lowers confidence);
  it is never painted over.
* **Without a `target`, the crop keeps every garment region.** One multi-colour garment (a lehenga with a
  contrast border) is often split by colour; cutting it apart hurt search in testing, so the default crop
  covers all regions. Set `ISOLATION_DEFAULT=main_garment` to crop only the most prominent piece, or send
  `target`.
* **Cached.** Masks are cached by (image hash, settings, target) and query embeddings by (image hash,
  model, model version, crop result, multi-crop mode), so paging or changing filters does not recompute.
* **Timed.** `timings_ms` has every stage (`foreground`, `scene`, `person_segmenter`, `classical_person`,
  `sam2`, `total`, …).

### Code map (`ai-service/app/services/segmentation/`)

| file | what it does |
|---|---|
| `pipeline.py` | routing, fallbacks, target choice, SAM 2 step, confidence check, cache, `isolate()` crop |
| `classical.py` | weight-free methods: scene analysis, skin model (adapts to the wearer's own skin tone from the head area), face/hair removal, colour + connectivity instance split, occluder (hand) removal, skin-toned lower garment rescue |
| `backends.py` | optional models behind one interface: Ultralytics YOLO (garments / person masks), SegFormer human parser, SAM 2 with point prompts + candidate scoring, MediaPipe pose; lazy loading, timeouts |
| `refine.py` | guided-filter boundary refinement, speck removal, hole filling, occlusion ratio, metrics (IoU, Dice, precision, recall, boundary F1) |
| `quality.py` | sharpness, brightness, contrast, garment share, warnings |
| `types.py` | `GarmentInstance` / `SegmentationResult` (JSON-safe output) |
| `cache.py` | small thread-safe LRU cache |

Also changed: `detection.py` (now calls the pipeline, keeps the v2 functions), `foreground.py` (v3 mode:
black-on-black retry, GrabCut keeps border rows, filled outlines; **v2 mode reproduces v2 exactly** and is
still used for catalogue features and `cv-baseline` vectors, so no stored data has to be recomputed),
`search.py` (target + embedding cache), `routes.py` (new endpoints).

---

## 2. Which models are integrated, and what was actually run

| component | status in this build | licence |
|---|---|---|
| Weight-free pipeline (foreground, skin/face/hair removal, instance split, occluder removal, guided refinement) | **default, tested** (unit tests + 260-image synthetic benchmark + one real photo) | project code |
| Ultralytics YOLO **person** masks (`PERSON_SEGMENTER=yolo`, `yolo11n-seg.pt`, COCO) | **run for real** in the sandbox: real weights, real street photo, test included (skipped if weights absent) | **AGPL-3.0** (Ultralytics code and weights) |
| Ultralytics YOLO **garment** detector (`GARMENT_DETECTOR=yolo`) | code path tested with a fake detector that returns DeepFashion2-style class names. **No clothing weights are bundled or were run.** | AGPL-3.0 + the licence of whatever weights you use |
| SegFormer human parser (`HUMAN_PARSER=segformer`, default `mattmdjaga/segformer_b2_clothes`) | code path tested with a small randomly initialised SegFormer (shapes, label mapping). **Real weights were not downloadable in the sandbox, so its accuracy here is unverified.** | model card says MIT; it was trained on the ATR dataset - check ATR's terms before commercial use |
| SAM 2 refinement (`ENABLE_SAM2=true`) | prompt + candidate-scoring logic tested with a fake predictor; real checkpoint not run | Apache-2.0 (code and checkpoints) |
| MediaPipe pose (`POSE_ESTIMATOR=mediapipe`) | code path present; not run (model file not downloadable in the sandbox); only ever used on person photos | MediaPipe code Apache-2.0; read the model card of the `.task` bundle you download |

**Standard COCO YOLO weights cannot detect garments.** COCO has a `person` class and no clothing classes. With
`yolo11n.pt` / `yolo11n-seg.pt` the garment detector correctly finds *nothing* (a test checks this). COCO
weights are only useful as the **person segmenter**. For garment boxes you need clothing-trained weights,
e.g. a YOLO model trained on DeepFashion2 (13 classes like `short_sleeve_top`, `trousers`, `sling_dress`).
DeepFashion2 is released for non-commercial research use only, so weights trained on it inherit that
limitation. No such weights ship with this project.

---

## 3. Installing the optional models (Windows PowerShell)

Everything below is optional. The default (`GARMENT_ISOLATION=auto`, all models `none`) needs no download.
Run these from the `ai-service` folder with your virtual environment active (`.\.venv\Scripts\Activate.ps1`).

### 3.1 Person masks with YOLO (recommended first step, ~6 MB, CPU-friendly)

```powershell
# installs Ultralytics (AGPL-3.0 licence - fine for a student project; read it before commercial use)
pip install "ultralytics>=8.3"
# make a folder for weights inside the project's model cache
mkdir ..\.model-cache\yolo -Force
# download the small COCO segmentation model (6 MB)
Invoke-WebRequest https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt -OutFile ..\.model-cache\yolo\yolo11n-seg.pt
```

Then in `.env`:

```
PERSON_SEGMENTER=yolo
PERSON_YOLO_WEIGHTS=../.model-cache/yolo/yolo11n-seg.pt
```

Measured in the sandbox (CPU): first call ~1.8 s (model load), then ~100 ms per photo, about +260 MB RAM.

### 3.2 Human parser (SegFormer, ~110 MB download on first use)

`transformers` and `torch` are already installed for FashionCLIP. Only `.env` changes:

```
HUMAN_PARSER=segformer
HUMAN_PARSER_MODEL=mattmdjaga/segformer_b2_clothes
```

The first person photo downloads the model into `.model-cache`. On a CPU expect a few hundred ms per photo
at `HUMAN_PARSER_SIZE=512`.

### 3.3 Clothing detector (only if you have clothing-trained YOLO weights)

```
GARMENT_DETECTOR=yolo
YOLO_WEIGHTS=C:\path\to\your-clothing-model.pt
```

Class names containing top/shirt/kurta/trousers/skirt/dress/dupatta/... are mapped automatically to the groups
`upper`, `lower`, `full_body`, `outer`, `accessory`. Shoes, bags, hats, belts and jewellery are ignored.

### 3.4 SAM 2 (GPU recommended; slow on CPU)

```powershell
pip install "git+https://github.com/facebookresearch/sam2.git"
mkdir ..\.model-cache\sam2 -Force
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -OutFile ..\.model-cache\sam2\sam2.1_hiera_tiny.pt
```

```
ENABLE_SAM2=true
SAM2_CHECKPOINT=../.model-cache/sam2/sam2.1_hiera_tiny.pt
SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
```

SAM 2 is used only for the target garment, only when its confidence is below 0.75, and only if its mask agrees
with the pipeline's estimate (consistency ≥ 0.5); otherwise the estimate is kept and the reason recorded.

### 3.5 Pose (MediaPipe)

```powershell
pip install mediapipe
mkdir ..\.model-cache\mediapipe -Force
Invoke-WebRequest https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -OutFile ..\.model-cache\mediapipe\pose_landmarker_lite.task
```

```
POSE_ESTIMATOR=mediapipe
POSE_MODEL_PATH=../.model-cache/mediapipe/pose_landmarker_lite.task
```

### CPU vs GPU

Your laptop has no NVIDIA GPU: keep `AI_DEVICE=cpu` and use the YOLO person segmenter (+ the parser if you
want). SAM 2 is practical only on a GPU (e.g. Google Colab, `AI_DEVICE=cuda`). Check what is configured and
loaded at any time:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/ai/segmentation/status
```

---

## 4. Configuration (`.env`)

| setting | default | meaning |
|---|---|---|
| `GARMENT_ISOLATION` | `auto` | `auto` = pipeline with the models enabled below; `foreground` = pipeline, weight-free methods only (your old v2 setting keeps working and maps here); `yolo` = also switch on the garment detector; `off` = never crop |
| `SEGMENTATION_STRATEGY` | `auto` | `auto` / `light` (never load a model) / `off` |
| `ISOLATION_DEFAULT` | `all_garments` | crop scope when no `target` is sent: `all_garments` or `main_garment` |
| `ISOLATION_PADDING` | `0.06` | margin around the crop |
| `GARMENT_DETECTOR` | `none` | `yolo` with clothing weights in `YOLO_WEIGHTS` |
| `DETECTION_MIN_CONFIDENCE` / `DETECTION_MIN_BOX_AREA` / `DETECTION_MAX_INSTANCES` | `0.35` / `0.02` / `10` | detector thresholds (box area as share of the image) |
| `PERSON_SEGMENTER`, `PERSON_YOLO_WEIGHTS` | `none`, `yolo11n-seg.pt` | YOLO-seg person masks |
| `HUMAN_PARSER`, `HUMAN_PARSER_MODEL`, `HUMAN_PARSER_SIZE` | `none`, `mattmdjaga/segformer_b2_clothes`, `512` | SegFormer parsing |
| `ENABLE_SAM2`, `SAM2_CHECKPOINT`, `SAM2_CONFIG` | `false`, empty, tiny config | SAM 2 refinement |
| `POSE_ESTIMATOR`, `POSE_MODEL_PATH` | `none`, empty | MediaPipe pose |
| `MIN_SEGMENTATION_CONFIDENCE` | `0.4` | below this: full image + reason |
| `SEGMENTATION_TIMEOUT_S` | `20` | per optional model call |
| `BOUNDARY_REFINEMENT` | `guided` | `guided` / `none` |
| `SEGMENTATION_CACHE_SIZE` / `EMBEDDING_CACHE_SIZE` | `64` / `256` | 0 disables the embedding cache |
| `FACE_CASCADE_PATH` | empty | optional OpenCV Haar face cascade file (not needed; some OpenCV builds lack the class) |

---

## 5. API

### `POST /ai/segment` (new)

Form fields: `image` (required), `target` (optional: instance index or a word like `upper`, `lower`,
`dupatta`), `strategy` (`auto`/`light`/`off`), `include_masks` (`true` adds a PNG mask per instance as base64).

```powershell
curl.exe -s -F "image=@C:\photos\me_in_kurta.jpg" -F include_masks=false http://127.0.0.1:8001/ai/segment
```

Response (shortened):

```json
{"path": "person", "method": "classical_person", "confidence": 0.57, "target_index": 1,
 "bbox": [43, 341, 111, 518], "fallback_used": false, "fallback_reason": null,
 "instances": [{"label": "garment region", "group": "lower", "confidence": 0.57, "bbox": [...],
                "area_share": 0.03, "occlusion": 0.0, "source": "classical_person"}],
 "attempts": [{"method": "person_segmenter", "ok": true, "reason": null, "ms": 97.6}],
 "timings_ms": {"foreground": 11.6, "scene": 1.9, "person_segmenter": 97.6, "total": 183.3},
 "quality": {...}, "scene": {...}, "cached": false}
```

### `GET /ai/segmentation/status` (new)

Configured backends, load errors, mask-cache statistics.

### `POST /ai/search` (unchanged contract, one optional field added)

`target` (with `crop=true`) chooses the garment in a multi-garment photo. `query.garment_isolation` keeps the
v2 fields (`method` = `foreground_crop` / `yolo_crop` / `full_image` / `classical_person_crop` / ...,
`reason`, `label`) and adds `segmentation` with the full structured result. The Node backend passes `target`
through from `POST /api/search/image`; the website does not send it yet, so nothing changes in the UI.

---

## 6. Testing and benchmarking

```powershell
cd ai-service
# all AI tests (117 + 1 that runs only when yolo11n-seg.pt is present)
python -m pytest -q
# only the segmentation tests
python -m pytest -q tests\test_segmentation.py
# regenerate the synthetic regression set (260 drawings, 13 scene types) and benchmark it
python scripts\make_segmentation_fixture.py
python scripts\evaluate_segmentation.py --tag synthetic
# retrieval with and without the crop (stored side by side in dataset\evaluation\experiments\)
python scripts\evaluate.py --models fashion-clip --tag v3_nocrop
python scripts\evaluate.py --models fashion-clip --crop --tag v3_crop
python scripts\compare_experiments.py
```

### Real-world evaluation (needed before claiming real accuracy)

The synthetic set is a regression check only. For real accuracy use a **licensed, human-annotated** set of
at least 1,000 photos. Fashionpedia's annotations are CC BY 4.0 (the images keep their own Flickr licences)
and come in COCO format:

```powershell
python scripts\evaluate_segmentation.py --format coco --images D:\fashionpedia\val2020 `
    --annotations D:\fashionpedia\instances_attributes_val2020.json --limit 1000 --tag fashionpedia_val
```

Garment parts (sleeve, neckline, pocket...) are ignored automatically for Fashionpedia; only whole garments
count. For per-category results (product photo / worn / multi-garment ...) pass `--meta file.csv` with
`file_name,category`. For your own photos use `--format maskdir` (`images/`, `masks/<name>.png`, optional
`instances/<name>__<k>__<label>.png`, optional `meta.csv`). Output: IoU, Dice, precision, recall, boundary F1
(normal and strict 2 px), instance IoU, AP50 and mAP@[.5:.95] (class-agnostic), fallback rate, p50/p95
latency, peak CPU RSS (and GPU memory on CUDA), per category and per method, in
`dataset\evaluation\experiments\<time>_segmentation_<tag>\` and `docs\SEGMENTATION_BENCHMARK.md`.

---

## 7. Results measured in the build sandbox

**No real-world segmentation benchmark was run** (no licensed annotated dataset was available in the sandbox).
What *was* measured:

**a) Synthetic regression set** (260 drawings with exact masks; full table in `docs/SEGMENTATION_BENCHMARK.md`):

| method | IoU | boundary F1 | strict BF1 (2 px) | p50 ms |
|---|---|---|---|---|
| v1 foreground | 0.781 | 0.587 | 0.546 | 4 |
| v2 foreground | 0.879 | 0.833 | 0.734 | 10 |
| v3 pipeline (weight-free) | **0.919** | **0.881** | **0.862** | 18 |

Biggest gains: worn garments 0.69 → 0.79, multi-garment 0.66 → 0.73, hand in front 0.83 → 0.97, white on white
0.90 → 0.98. Still weak: busy backgrounds (0.63) - that needs the parser/detector on real photos.
Adding the real YOLO person model changed nothing on these drawings (it does not see cartoon people) and
cost ~+80 ms p95 and ~+260 MB RAM; it is meant for real photos.

**b) One real photo** (Ultralytics' public `bus.jpg`, used only locally, not shipped): without a person model
the weight-free cues missed the people; with `PERSON_SEGMENTER=yolo` the main person's jacket and jeans were
isolated and face/hands excluded, ~180 ms total on CPU. One photo is a smoke test, not a benchmark.

**c) Retrieval impact** (sandbox catalogue: 360 synthetic products, `cv-baseline` encoder, 300 queries,
hybrid ranking; FashionCLIP could not be downloaded there):

| | v2 | v3 |
|---|---|---|
| no crop R@1 / R@10 / mAP | 0.757 / 0.957 / 0.828 | **identical** |
| crop R@1 | 0.740 | **0.783** |
| crop R@10 | 0.950 | 0.947 |
| crop mAP | 0.820 | **0.845** |
| crop p50 latency | 120 ms | **90 ms** |

By query type (crop, R@1): clutter 0.85 → 0.98, background 0.62 → 0.77, lighting 0.57 → 0.58, angle unchanged,
**partial/close-up 0.70 → 0.62 (worse)**. Close-ups are the one place the v3 crop is behind v2; if your users
mostly upload close-ups, compare with your own data before turning crop on by default.

Your FashionCLIP numbers must be measured on your machine (commands in section 6).

---

## 8. Known limitations

* Without a parser or clothing detector, person photos rely on colour: a garment in the wearer's skin tone is
  ambiguous. The result then carries a low confidence and a note (and a wide skin-toned block below the waist
  is kept as a low-confidence `lower` garment).
* The weight-free person cue (face/head shape) misses many real photos; enable `PERSON_SEGMENTER=yolo` for
  real users.
* A hand that overlaps the garment's outline (not fully inside it) is not removed.
* Garments that run off the frame on a low-contrast background fall back to the full image.
* Instance labels from the weight-free path are generic ("garment region" + group by position); real class
  names (kurta, dupatta...) come only from a clothing detector or parser.
* Transparent dupattas are segmented as solid regions (no alpha matting model is included); the guided filter
  only softens the boundary band.
* Pose and SAM 2 are integrated but were not run with real weights in this build.
