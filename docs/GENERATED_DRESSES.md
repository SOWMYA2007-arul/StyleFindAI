# 500 AI-generated dress photos (Amazon-style) for StyleFind

These images are **made by an AI model**. They are not photos of real products, and every one is labelled as
AI-generated in the database and on its product page. Use them for demos and testing, not as a real shop catalogue.

## Your images are already in this folder

`dataset\raw\generated-dresses\` contains the **215 dresses you generated** (SDXL-Turbo, 512 × 512): 16 dress
types, 11 patterns, 30+ colours. About a quarter show a generated model wearing the dress; the rest are mannequin,
hanger, flat-lay or dress-form shots. You can import them right away (Step 3). They were tested in the build
environment:

* all 215 pass the importer's checks (no broken files, no duplicates)
* the requested colour is confirmed in the image for **92 %** of them (the other 8 % really came out in a different
  colour, e.g. turquoise instead of blue; those get the colour measured in the image)
* a rotated and cropped copy of a dress, used as a search photo, finds that exact dress first
* with the sandbox's simple encoder: R@1 0.67 / R@10 0.84 on 500 edited test queries (FashionCLIP on your laptop
  will give different numbers; step 5 of the import measures them)

**Why 215 and not 540?** The first version of the quality check demanded a pure white background and threw away 323
good images that had off-white or beige studio walls. That rule is fixed. To fill up to 500+, run the notebook again
(Step 1) with the **same** `OUT` folder and `SEED`: it keeps the 215 and generates only the missing ones (~10 minutes).
If you deleted the Google Drive folder, first upload `dataset\raw\generated-dresses` back to
`MyDrive/stylefind/generated-dresses`. Then import again (Step 3); the 215 already imported are updated, not duplicated.

---

The images are generated on **Google Colab's free GPU**. This project folder does not contain them, because generating
500 realistic photos needs an NVIDIA GPU and a 7 GB model download. Your laptop has no NVIDIA GPU, so Colab is the
practical way. It takes about 30 minutes in total.

```
 Colab (GPU)                                   Your laptop
 ───────────                                   ───────────
 notebooks/generate_dresses_colab.ipynb   ──►  dataset\raw\generated-dresses\
   540 prompts (balanced designs)              images.csv + images_original\*.jpg
   SDXL-Turbo, 512x512                                    │
   CLIP quality check                                     ▼
   download generated-dresses.zip              scripts\import-generated-dresses.ps1
                                               (check, choose best 500, database,
                                                embeddings, test queries, evaluation)
```

---

## Step 1: generate on Colab (≈ 25 min)

1. Open https://colab.research.google.com → **File → Upload notebook** → choose
   `StyleFindAI\notebooks\generate_dresses_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the cells from top to bottom (`Shift + Enter`). The notebook asks for Google Drive access. Allow it, so the
   images survive a disconnect.
4. Cell 7 is the long step. You will see progress such as `120/540  1.4 s/image  ~10 min left`.
   **If Colab disconnects**, run all cells again: the script skips the images that already exist.
5. Cell 8 shows 40 of the images. Cell 9 downloads `generated-dresses.zip`. If the browser download is slow,
   get the folder `MyDrive/stylefind/generated-dresses` from Google Drive instead.

What gets generated:
* **540 different designs**: 16 dress types (A-line, wrap, shift, sheath, fit-and-flare, maxi, shirt dress, slip,
  tiered, bodycon, kaftan, anarkali, smocked, peplum, sweater, skater), 29 colours, 11 patterns (solid, striped,
  checked, floral, polka dot, geometric, abstract, paisley, block print, embroidered, textured), 4 sleeve types,
  6 necklines, 3 lengths and 10 fabrics. Each value is used about equally often, and no two designs are identical.
* **Amazon-style presentation**: about 50 % on an invisible "ghost" mannequin on white, 25 % flat lay on white,
  15 % on a wooden hanger against a light wall, and 10 % on a headless dress form. The prompts ask for no people, but
  SDXL-Turbo sometimes draws a model wearing the dress anyway (see `--no-people`).
* **Quality check (`--qc`)**: CLIP rejects images that do not look like a dress (another garment, text, broken
  image), and pixels reject dark or cluttered backgrounds. A rejected image is regenerated with a new seed, up to 2
  times. Rejects are listed in `rejected.json`. Options: `--no-people` also rejects images with a generated model
  (SDXL-Turbo ignores the "no person" negative prompt, so ~25 % show one); `--strict-white` restores the old
  pure-white rule.
* 540 are made so that 500 good ones remain after the checks and duplicate removal.

### Presets

| preset | model | size | time on free T4 | licence |
|---|---|---|---|---|
| `fast` (default) | SDXL-Turbo, 4 steps | 512 × 512 | ~1-2 s per image → 15-20 min | Stability AI Community License: free for research, non-commercial, and commercial use under USD 1M yearly revenue (commercial users register with Stability AI) |
| `quality` | SDXL base 1.0, 25 steps | 1024 × 1024 | ~25 s per image → ~4 h (resumable over several sessions) | CreativeML Open RAIL++-M |

To switch, change `PRESET = "quality"` in cell 5. Keep the same `SEED` when resuming.

Running the generator on your laptop's CPU is possible (`python scripts\generate_dress_images.py --out ... --device cpu`).
It is not recommended: SDXL in full precision needs about 10 GB of RAM and takes several hours for 540 images.

---

## Step 2: put the images into the project

Unzip `generated-dresses.zip` so that you get exactly:

```
StyleFindAI\dataset\raw\generated-dresses\
    images.csv
    images_original\dress-0000.jpg ... dress-0539.jpg
    plan.json, rejected.json        (kept for reference)
```

PowerShell (adjust the download path):

```powershell
Expand-Archive "$env:USERPROFILE\Downloads\generated-dresses.zip" -DestinationPath "C:\Projects\StyleFindAI\dataset\raw\generated-dresses"
```

---

## Step 3: import (one command)

Close the AI service window first. The import loads FashionCLIP itself.

```powershell
cd C:\Projects\StyleFindAI
powershell -ExecutionPolicy Bypass -File scripts\backup-db.ps1
powershell -ExecutionPolicy Bypass -File scripts\import-generated-dresses.ps1 -PreviewOnly
powershell -ExecutionPolicy Bypass -File scripts\import-generated-dresses.ps1
```

`-PreviewOnly` copies nothing. Check `docs\IMPORT_REPORT_AI.md` and `dataset\metadata\ai_preview.jpg` first.
The full run then:

1. runs the migrations (the "Western Dresses" category already exists)
2. checks every image (decodes, size, duplicates), isolates the dress, measures its colour and pattern, and keeps the
   500 most varied. The products are added as `ai-dress-0000` … in **Western Dresses**, with
   `source_dataset = "AI-generated dresses (StyleFind demo)"`.
3. creates FashionCLIP embeddings for the new images only
4. creates test queries `queries_generated.csv` (your baseline `queries.csv` is untouched)
5. evaluates without and with garment crop → `docs\EXPERIMENTS.md`
6. shows what the website displays

Start the site: `powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1`.

To show **only** the generated dresses on the website (nothing is deleted, `--restore` undoes it):

```powershell
cd ai-service
python scripts\catalogue_visibility.py --only-source "AI-generated dresses (StyleFind demo)"
```

### What is stored for each dress

| field | value |
|---|---|
| name | e.g. "Navy blue polka dot midi wrap dress", built from the requested design. The colour word is the requested colour only if that colour covers at least 8 % of the image centre (where the dress is); otherwise the colour measured on the image centre is used, because Stable Diffusion does not always follow the prompt. |
| description | starts with "AI-GENERATED product image (not a real product)" and includes the prompt |
| colour | the confirmed requested colour, or the colour measured on the image centre; unverified `model_prediction` with a confidence. The description says which. |
| pattern | the texture classifier / FashionCLIP result; where they cannot tell, the requested pattern (confidence 0.5). Unverified `model_prediction`. |
| website filters | the site shows and filters only verified attributes. Add `--trust-measured` to the import (edit the `$common` line in `scripts\import-generated-dresses.ps1`) so confident colours are shown and filterable; see docs/CLOTHING_DATASET.md. |
| sleeve type, neckline | what the prompt asked for, stored as unverified `model_prediction` |
| price | empty (add prices in the admin panel if you want them) |

The report shows how often the measured colour matched the requested colour.

---

## What was tested and what was not

* **Tested here:** the prompt plan (540 unique, balanced designs; no clashing combinations; every prompt within the 77
  tokens SDXL's text encoder reads, checked with the CLIP tokenizer); writing the folder layout; resuming after a lost
  image without duplicate rows; quality-check retries and rejects; the importer reading the generated folder (dry run
  and a real database load with stand-in images, then removed); naming and colour agreement. Tests:
  `tests/test_generate_dresses.py` (8). All 131 AI tests pass. Your 215 real generated images were imported, embedded,
  evaluated and searched through the website backend in the build environment.
* **Not tested here:** the actual Stable Diffusion and CLIP downloads and image quality. The build machine has no GPU
  and cannot reach Hugging Face. How realistic the images look is up to the model; check the preview in notebook
  cell 8 before importing. The PowerShell scripts were written for Windows and reviewed, but not executed here.
* Search scores measured on generated images say how well the system finds *these* pictures. They are not evidence of
  how it performs on real shop photos.
