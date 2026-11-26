# Dataset notes

## Primary source: Kaggle "Fashion Product Images"

* Small version (≈ 44k images, 60×80 to 1080px variants): https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
* High-resolution version: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
* Images and metadata originate from Myntra product listings. Read the licence shown on the Kaggle page
  before any public or commercial deployment; `prepare_dataset.py` records the source and a licence note
  in every catalogue row (`source_dataset`, `licence_information`).
* The small version's images are low resolution (most are 60×80). Retrieval works, but the website looks
  better with the high-resolution version. Both have the same `styles.csv` + `images/<id>.jpg` layout, so the
  same command works: `python scripts/prepare_dataset.py --raw dataset/raw/<folder>`.

### What prepare_dataset.py does

| Step | Rule |
|---|---|
| Read CSV | repairs ~20 rows whose product name contains unquoted commas instead of dropping them |
| Women only | `gender == "Women"` |
| Clothing only | `masterCategory == "Apparel"`; footwear, accessories (bags, wallets, watches, jewellery), personal care, free items, sporting goods excluded |
| No innerwear / sleepwear | subCategory `Innerwear`, `Loungewear and Nightwear`, `Socks` and article types like Bra, Briefs, Nightdress excluded |
| Category mapping | Kurtas/Kurtis/Kurta Sets/Tunics → kurtas-kurtis; Sarees; Lehenga Choli → lehengas; Dresses/Jumpsuit → western-dresses; Tops/Shirts/Tshirts → tops-shirts; Salwar/Churidar/Dupatta/Patiala/Shrug/Nehru Jackets → other-ethnic. Anything unmapped is excluded |
| Duplicate IDs | first occurrence kept |
| Images | must exist, decode, be JPEG/PNG/WEBP, shortest side ≥ 60 px |
| Duplicate images | SHA-256 (byte-identical) + perceptual hash ≤ 3 bits **and** 32×32 colour-thumbnail check (pHash alone is colour-blind: same-cut garments in different colours collide) |
| Quotas | default 300/200/150/200/150/200; a shortfall is filled from categories with spare products and reported |
| Files | copied to `dataset/products/<category>/<original id>.jpg`; the id-to-metadata link is never broken |
| Attributes | colour = dataset `baseColour` (verified metadata); pattern / sleeve / neckline parsed from the product name, else `unknown` |

### Expected shortfall

The Kaggle catalogue has very few women's **Lehenga Choli** items (a few dozen at most), so the 150-lehenga
quota cannot be met. `prepare_report.json` lists available vs selected per category. Legitimate ways to add data:

1. Use the high-resolution Kaggle version (same products, so it doesn't add lehengas, but better images).
2. Your own photographs of garments you own, or with written permission from a shop. Add them through the admin panel
   (recorded as `admin_upload`).
3. Open-licence datasets that allow redistribution and list licence per image, e.g. Hugging Face datasets with
   an explicit CC-BY / CC0 licence. Check the licence of each source before importing.
4. Retailer partner feeds (for example from ThreadKart sellers) with a signed data-use agreement.

Do **not** scrape images from shopping sites without permission.

## DeepFashion2 (evaluation)

https://github.com/switchablenorms/DeepFashion2 provides consumer-to-shop pairs (the same garment photographed
by a customer and by a shop), which is the most honest test for this project. Access requires filling in the
authors' agreement form, and use is limited to non-commercial research. The data is not included and no loader
pretends to use it. To evaluate with it (or with your own photos):

1. load the shop images as catalogue products (`load_catalogue.py` reads `catalogue.csv`),
2. write a CSV `query_path,product_id,transform` with the consumer photos under `dataset/evaluation/`,
3. run `python scripts/evaluate.py --queries dataset/evaluation/<your file>.csv`.

## Synthetic test fixture (development only)

`scripts/make_test_fixtures.py` draws simple garment silhouettes with colours and prints, in the Kaggle file layout,
plus deliberate junk rows (footwear, innerwear, missing file, corrupt file, duplicate image, duplicate id, men's item).
It exists only to test the pipeline on a machine without the real dataset. Every row is tagged
`SYNTHETIC_TEST_FIXTURE`; the evaluation report warns when it was used. Do not ship it as the catalogue. To switch to
the real data, run `prepare_dataset.py --clean` on the Kaggle folder, then `load_catalogue.py`, and delete fixture rows:
`DELETE FROM products WHERE source_dataset = 'SYNTHETIC_TEST_FIXTURE';` (or recreate the database).


## Integrating a new high-resolution dataset (plan - nothing is done automatically)

1. **Inspect (read-only):** `python scripts/inspect_dataset.py --root <folder>` -> `docs/NEW_DATASET_REPORT.md`
   (resolution vs the current 60x80, categories and their mapping to the six StyleFind categories, women-only
   share, duplicates, corrupt files, licence text). Decide only after reading the licence.
2. **Prepare as a separate catalogue version:** run `prepare_dataset.py` with its own `--id-prefix` (e.g. `hr`) and
   `--source` / `--licence` values into a copy of the project or a second database (`POSTGRES_DB=stylefind_hr`), so the
   live 1,200-product catalogue is untouched. If the dataset's columns differ from Kaggle's `styles.csv`, the
   column mapping is added to `prepare_dataset.py` first (send me the report).
3. **Embed + evaluate:** `generate_embeddings.py --model fashion-clip`, generate queries for the new catalogue,
   `evaluate.py`, and compare with the saved FashionCLIP baseline (`compare_experiments.py`). Note that a
   different catalogue changes the task, so compare per-category results and real photos, not only R@1.
4. **Category balance:** only add garments that really belong to a category. If lehengas remain scarce, the
   report says so; do not relabel other garments to fill the quota.
5. **Switch only with your approval:** either point `.env` at the new database or load the new products into the
   live one alongside the old ones (different `product_code` prefix, `source_dataset` recorded). Old products can be
   hidden (`is_active = false`) instead of deleted, so nothing is lost.
