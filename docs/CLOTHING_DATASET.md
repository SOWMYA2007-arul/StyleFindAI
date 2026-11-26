# Adding the Clothing Dataset (500 diverse images)

Dataset: **"Clothing dataset (full, high quality)"** by Alexey Grigorev - 5,403 photos, 20 garment types,
**CC0 1.0 public domain** (free for any use, including commercial). Source:
https://github.com/alexeygrigorev/clothing-dataset (also on Kaggle). Files: `images.csv`
(`image,sender_id,label,kids`), `images_original/` (full resolution), `images_compressed/` (400 px wide).

The import **adds** 500 products next to your existing catalogue. Nothing is deleted, no existing
product, image or embedding is changed, and you can switch which products the website shows at any time.

---

## 1. Where to put your files

Copy your `archive (2)` contents into the prepared folder, so it looks exactly like this:

```
StyleFindAI\
└── dataset\
    └── raw\
        └── clothing-dataset\          <- this folder already exists in the zip
            ├── images.csv             <- from archive (2)
            ├── images_original\       <- from archive (2)  (all 5,403 .jpg)
            └── images_compressed\     <- from archive (2)  (optional, makes the analysis faster)
```

PowerShell (change the first path to where your `archive (2)` folder is):

```powershell
$src = "C:\Users\you\Desktop\dress_match\StyleFindAI-v3 (1)\archive (2)"
$dst = "C:\Projects\StyleFindAI\dataset\raw\clothing-dataset"
# copy the csv and both image folders; robocopy prints a summary at the end
Copy-Item "$src\images.csv" $dst
robocopy "$src\images_original"   "$dst\images_original"   /E
robocopy "$src\images_compressed" "$dst\images_compressed" /E
```

You copy the whole dataset, and the script picks the 500 (it needs to see all of them to choose
different ones). Only the 500 chosen photos are copied into `dataset\products\`, resized to max 1024 px
(about 50-100 MB in total). The `raw` folder can be deleted afterwards if you need the space.

The folder name does not have to match: `--raw` can point anywhere, and the script also finds
`images.csv` one level down (e.g. `--raw "C:\...\StyleFindAI-v3 (1)"`).

---

## 2. Run it (one command)

Back up the database first (it is quick, and the import changes the database):

```powershell
cd C:\Projects\StyleFindAI
powershell -ExecutionPolicy Bypass -File scripts\backup-db.ps1
```

Close the AI service window if it is running (the import loads FashionCLIP itself; two copies would need
~3 GB RAM). Then **preview first** (copies nothing, writes nothing to the database):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\import-clothing-dataset.ps1 -PreviewOnly
```

Open `docs\CLOTHING_DATASET_IMPORT.md` (counts per type, pattern, colour, what was rejected and why) and
`dataset\metadata\clothing_dataset_preview.jpg` (thumbnails of the 500 chosen photos). If you are happy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\import-clothing-dataset.ps1
```

This runs six steps and stops at the first error:

| step | what happens | time on a 16 GB CPU laptop (estimate) |
|---|---|---|
| 1 | `migrate.py` adds the categories outerwear, pants, shorts, skirts (+ footwear, hats) | seconds |
| 2 | `import_clothing_dataset.py --load` chooses, copies and inserts the 500 products | ~10-20 min (≈2,000 photos analysed with FashionCLIP) |
| 3 | `generate_embeddings.py --model fashion-clip` embeds **only the new 500 images** | a few minutes |
| 4 | `make_eval_queries.py` makes test queries for the new products (`queries_clothing.csv`; your baseline `queries.csv` is untouched) | ~1 min |
| 5 | `evaluate.py` without and with garment crop on those queries, then `compare_experiments.py` → `docs\EXPERIMENTS.md` | a few minutes |
| 6 | shows how many products of each source are visible | seconds |

Then start the site as usual: `powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1`.

Options: `-Count 300` (fewer images), `-Model cv-baseline` (no model download; much weaker search),
`-Raw "D:\somewhere\archive (2)"` (dataset elsewhere).

### The same steps by hand (if you prefer to see each one)

```powershell
cd C:\Projects\StyleFindAI\ai-service
.\.venv\Scripts\Activate.ps1
python scripts\migrate.py
python scripts\import_clothing_dataset.py --raw ..\dataset\raw\clothing-dataset --dry-run
python scripts\import_clothing_dataset.py --raw ..\dataset\raw\clothing-dataset --load
python scripts\generate_embeddings.py --model fashion-clip
python scripts\make_eval_queries.py --catalogue ..\dataset\metadata\clothing_dataset_catalogue.csv --name queries_clothing --n 100
python scripts\evaluate.py --models fashion-clip --queries ..\dataset\evaluation\queries_clothing.csv --tag clothing_nocrop
python scripts\evaluate.py --models fashion-clip --queries ..\dataset\evaluation\queries_clothing.csv --crop --tag clothing_crop
python scripts\compare_experiments.py
```

---

## 3. How the 500 are chosen

1. **Labels kept:** T-Shirt, Longsleeve, Polo, Shirt, Blouse, Top (→ Tops & Shirts), Dress (→ Western Dresses),
   Outwear, Hoodie, Blazer (→ Outerwear & Jackets), Pants, Shorts, Skirt.
   **Dropped:** Not sure, Skip, Other (no clear garment), Undershirt, Body (underwear / baby bodysuits).
   **Opt-in:** Shoes (`--include-footwear`), Hat (`--include-hats`). Kids' clothes are kept and marked
   "(kids)" (`--kids exclude` to drop them, `--kids only` for kids only).
2. **Checks:** file exists, decodes, at least 200 px; byte-identical and near-identical photos removed
   (pHash plus a colour-thumbnail check, so a red and a blue shirt with the same cut are not treated as duplicates).
3. **Analysis:** the garment is isolated (the v3 segmentation pipeline, weight-free mode), then its colour is
   measured from pixels and its pattern by the texture classifier plus FashionCLIP zero-shot (floral, paisley,
   printed...). Without FashionCLIP (`--pattern-model none`) only solid / striped / checked / polka dot are
   named, and other prints stay "unknown".
4. **Quota per type:** proportional to the square root of how many exist, so T-shirts do not swamp the set and
   rare types (Blouse, Top) still appear. Example for the full dataset: T-Shirt ≈ 75, Longsleeve ≈ 62, Pants ≈ 62,
   Shirt ≈ 46, Dress ≈ 45, Outwear ≈ 42, Shorts ≈ 41, Skirt ≈ 29, Polo ≈ 25, Blazer ≈ 24, Hoodie ≈ 23, Top ≈ 15,
   Blouse ≈ 11 (the report shows the real numbers).
5. **Diversity inside each type:** round-robin over pattern groups (rare patterns first), and in each group the
   photo whose colour + texture is farthest from everything already chosen. Photos where segmentation had to
   fall back go last.

The same `--seed` gives the same 500 every time; change it (e.g. `--seed 8`) for a different selection.

### What goes into the database

| field | value |
|---|---|
| product_code | `cd-<image id from images.csv>` (re-running updates, never duplicates) |
| name | measured colour + pattern + type, e.g. "Navy blue striped T-shirt (kids)" |
| category / subcategory | from the dataset's **human label** |
| colour, pattern | measured from the photo → stored as `model_prediction`, **unverified**, with confidence |
| sleeve type | "full sleeves" for the dataset's Longsleeve label (human metadata, verified); otherwise a hint, unverified |
| source_dataset / licence | `clothing-dataset-full (Alexey Grigorev)` / CC0 |
| price | empty (the dataset has no prices; you can add them in the admin panel) |

**Website behaviour:** the site shows and filters only *verified* attributes (a project rule, so a model guess
never looks like catalogue truth). The new products can be found by category and by photo search, and their name
contains the measured colour, but the colour/pattern filters do not include them. If you want measured values to
be shown and filterable for your demo, add `--trust-measured` to the import command: only confident measurements
(colour ≥ 0.35, pattern ≥ 0.70) become verified, still labelled `model_prediction`. To undo, run the import again
without the flag.

---

## 4. Choosing what the website shows (nothing is deleted)

```powershell
cd C:\Projects\StyleFindAI\ai-service
python scripts\catalogue_visibility.py --status
# only the new 500 products:
python scripts\catalogue_visibility.py --only-source "clothing-dataset-full (Alexey Grigorev)"
# undo the last change (back to exactly how it was):
python scripts\catalogue_visibility.py --restore
# or show everything:
python scripts\catalogue_visibility.py --all
```

This only flips `is_active`; embeddings stay, so switching back is instant. Each change saves the previous state
to `dataset\metadata\visibility_backup_<time>.json`.

---

## 5. What was tested before you got this

* The importer ran on **1,259 real photos from this same dataset** (the 400 px versions fetched from the author's
  GitHub, plus up-scaled copies standing in for `images_original`), with the full 5,403-row `images.csv`:
  500 selected, 3,471 rows correctly reported as "file missing", a deliberately corrupted file rejected, 13 garment
  types, 261 distinct type + pattern + colour combinations, 10 segmentation fallbacks among the chosen photos,
  about 3 minutes without FashionCLIP.
* Loaded into the database twice: the second run updated the same 500 rows (no duplicates). Embedded with
  `cv-baseline` (only the 500 new images). Website API: new categories listed, products and filters served,
  photo search returns the new products.
* 6 new tests (`tests/test_clothing_import.py`); all 123 AI tests and 16 backend tests pass.
* **Not run here:** FashionCLIP (cannot be downloaded in the build sandbox), so the pattern names from zero-shot and
  the FashionCLIP search quality on these photos are untested. On cv-baseline (a weak, model-free encoder) the new
  catalogue scored hybrid R@1 0.66 / R@10 0.85; your FashionCLIP numbers will be different. Measure them in step 5.

## 6. Things to know

* These are **real home photos** (beds, floors, hangers), not studio shots. Search queries that are photos of the
  same garment will mostly work well; the "Focus on the garment" crop matters more here than for studio photos,
  so compare the `clothing_nocrop` and `clothing_crop` runs: step 5 writes the comparison to `docs\EXPERIMENTS.md`.
* Test queries in step 4 are edited versions of the catalogue photos (crop, blur, lighting...). They measure
  robustness, not "a different photo of the same garment", because the dataset has one photo per garment.
* The catalogue's colour/pattern features (used by hybrid ranking) are measured on the whole photo with the
  v2 background method, like all other products. A busy background can affect them.
* Mixing 60×80 Myntra product shots and these photos in one catalogue works, but results look more consistent
  if you show one source at a time (section 4).
