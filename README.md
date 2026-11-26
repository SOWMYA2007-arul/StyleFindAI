# StyleFindAI 👗

**AI-Powered Visual Fashion Search | Full-Stack Application | Computer Vision | Deep Learning**

StyleFindAI is an AI-powered visual fashion search engine that helps users discover visually similar clothing products by uploading an image of a kurta, saree, lehenga, or dress.

Instead of relying only on text descriptions or manually selected filters, StyleFindAI uses deep learning, computer vision, image embeddings, vector similarity search, and hybrid ranking to retrieve relevant fashion products from a real catalogue.

### How it works

1. Accepts and validates an uploaded clothing image.
2. Processes the image and extracts its visual representation using **FashionCLIP**.
3. Converts the image into a **512-dimensional embedding vector**.
4. Searches catalogue embeddings stored in **PostgreSQL with pgvector** using cosine similarity.
5. Retrieves the closest visual candidates.
6. Re-ranks the candidates using visual similarity, colour, pattern, and clothing attributes.
7. Returns actual product records and their stored images.

The search results are retrieved dynamically from the database rather than hard-coded or mocked.

> **New here or upgrading?** Read [START_HERE.md](START_HERE.md) first. The repository includes 215 AI-generated dress images under `dataset/raw/generated-dresses/`.

> **Want 500 Amazon-style AI-generated dress photos?** Generate them using free Google Colab with `notebooks/generate_dresses_colab.ipynb`, then follow [docs/GENERATED_DRESSES.md](docs/GENERATED_DRESSES.md).

> **Adding the Clothing Dataset (500 diverse photos)?** Place it in `dataset/raw/clothing-dataset/` and follow [docs/CLOTHING_DATASET.md](docs/CLOTHING_DATASET.md).

> **Already running v2?** Follow [docs/UPGRADE_V3.md](docs/UPGRADE_V3.md) for the garment detection and segmentation upgrade. From v1, follow [docs/UPGRADE_V2.md](docs/UPGRADE_V2.md) first.

See [CHANGELOG.md](CHANGELOG.md) for changes and [docs/SEGMENTATION.md](docs/SEGMENTATION.md) for segmentation details.

---

## 1. Project Architecture

StyleFindAI uses a modular architecture with separate frontend, backend, AI service, and database components.

```text
                    ┌──────────────────────────┐
                    │   React + Vite Frontend  │
                    │       Port: 5173         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Node.js + Express API  │
                    │       Port: 4000         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Python FastAPI       │
                    │      AI Service          │
                    │       Port: 8001         │
                    └────────────┬─────────────┘
                                 │
                  ┌──────────────┼───────────────┐
                  ▼              ▼               ▼
             FashionCLIP      OpenCLIP         DINOv2
                  │              │               │
                  └──────────────┼───────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ PostgreSQL 16 + pgvector │
                    │                          │
                    │ Embeddings + HNSW Index  │
                    └──────────────────────────┘
```

### Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Backend | Node.js 22, Express, TypeScript |
| AI Service | Python 3.11, FastAPI |
| Deep Learning | PyTorch, FashionCLIP, OpenCLIP, DINOv2 |
| Computer Vision | OpenCV, image preprocessing, foreground extraction, segmentation |
| Vector Search | PostgreSQL 16, pgvector, cosine similarity, HNSW |
| Authentication | JWT-based authentication, HttpOnly cookies, CSRF protection |
| Testing | Pytest, Vitest, Playwright |
| Deployment | Docker, Docker Compose, PowerShell scripts |

### Repository Structure

| Folder | Description |
|---|---|
| `ai-service/` | Python AI service, image preprocessing, model encoders, vector search, ranking, dataset and evaluation scripts |
| `backend/` | Node.js REST API, authentication, uploads, product management and AI-service integration |
| `frontend/` | React and TypeScript user interface |
| `database/` | PostgreSQL schema and ordered database migrations |
| `dataset/` | Raw datasets, processed images, product images, metadata and evaluation data |
| `docs/` | API documentation, model notes, datasets, evaluation reports and screenshots |
| `scripts/` | PowerShell scripts for setup, environment checks, database backup and application startup |
| `tests/` | Browser-based end-to-end tests |
| `notebooks/` | Google Colab notebook for generating fashion images |

---

## 2. AI Architecture & Techniques

The main objective of StyleFindAI is to retrieve fashion products based on visual appearance rather than depending exclusively on text-based search.

The AI pipeline combines deep learning-based image representations, computer vision, vector similarity search and hybrid ranking.

### 2.1 Image Preprocessing

Before an image is passed to the deep learning model, it goes through an image preprocessing pipeline.

Techniques include:

- Image validation and input handling.
- Image resizing and normalization.
- Image preprocessing compatible with the selected encoder.
- Optional foreground extraction and garment isolation.
- Optional segmentation-based preprocessing.

These steps help prepare uploaded images for consistent feature extraction.

### 2.2 FashionCLIP — Visual Feature Extraction

**FashionCLIP** is the primary image encoder used by StyleFindAI.

FashionCLIP is a vision-language model adapted for the fashion domain. It learns meaningful representations of fashion images and text descriptions.

In the primary search pipeline:

1. The uploaded image is passed through FashionCLIP.
2. The model extracts a visual feature representation.
3. The representation is stored as a 512-dimensional embedding.
4. The embedding is compared with the precomputed catalogue embeddings.

This allows the system to identify products with similar visual characteristics even when their filenames or product descriptions are different.

**Why FashionCLIP?**

General-purpose image models may not capture fashion-specific visual similarities as effectively. FashionCLIP provides a representation designed for fashion-related images and concepts.

### 2.3 Alternative Deep Learning Encoders

StyleFindAI also supports alternative encoders for experimentation and retrieval evaluation.

| Model | Technique | Purpose |
|---|---|---|
| FashionCLIP | Vision-language representation learning | Primary fashion image retrieval |
| OpenCLIP ViT-B/32 | Contrastive image-text representation learning | Alternative visual embedding model |
| DINOv2 ViT-B/14 | Self-supervised Vision Transformer | Alternative visual feature extraction |
| CV Baseline | Traditional computer vision-based representation | Baseline comparison |

The alternative encoders can be used to compare retrieval performance and evaluate which representation works better for the fashion catalogue.

The primary configured model is FashionCLIP.

### 2.4 Image Embeddings & Vector Search

An image embedding is a numerical representation of an image.

Instead of comparing two images pixel by pixel, StyleFindAI compares their learned feature vectors.

For the primary FashionCLIP model:

\[
E(I) \in \mathbb{R}^{512}
\]

Where:

- \(I\) represents an image.
- \(E(I)\) represents its embedding.
- 512 is the embedding dimensionality.

The system stores catalogue embeddings in PostgreSQL using the **pgvector** extension.

For image retrieval, it uses cosine similarity:

\[
\operatorname{Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

A higher cosine similarity indicates that two embedding vectors point in more similar directions.

**Vector database techniques:**

- PostgreSQL for persistent product and embedding storage.
- pgvector for vector operations.
- Cosine similarity for visual similarity comparison.
- HNSW (Hierarchical Navigable Small World) indexing for efficient approximate nearest-neighbour retrieval.

The system retrieves an initial set of visually similar candidates before applying additional ranking.

### 2.5 Hybrid Ranking & Re-ranking

Visual similarity alone may not always produce the most relevant fashion recommendations.

For example, two dresses may have similar shapes but completely different colours or patterns.

To improve relevance, StyleFindAI applies a hybrid ranking approach.

The initial candidate set is retrieved using vector similarity and then re-ranked using:

- Visual similarity.
- Colour similarity.
- Pattern similarity.
- Clothing attributes.

The configured ranking weights are:

| Ranking component | Weight |
|---|---:|
| Visual similarity | 0.65 |
| Colour similarity | 0.15 |
| Pattern similarity | 0.10 |
| Attribute similarity | 0.10 |
| **Total** | **1.00** |

The weights are configurable through environment variables and normalized automatically.

The ranking configuration is designed to prioritize learned visual representations while incorporating additional fashion-specific characteristics.

### 2.6 Colour & Pattern Analysis

StyleFindAI includes additional image-analysis techniques to improve fashion retrieval.

**Colour analysis**

The system analyses colour information to help distinguish products with similar visual structures but different colour combinations.

**Pattern analysis**

Pattern-related features help distinguish clothing such as:

- Plain garments.
- Printed garments.
- Striped patterns.
- Other visually distinguishable fabric patterns.

These signals contribute to the hybrid ranking stage.

### 2.7 Garment Detection, Foreground Extraction & Segmentation

Backgrounds, people and unrelated objects can affect visual similarity.

StyleFindAI includes a garment-isolation pipeline intended to focus more strongly on the clothing item.

The pipeline supports configurable preprocessing and optional segmentation backends.

Techniques and optional components include:

- Foreground extraction.
- Classical computer vision-based segmentation.
- Garment detection using YOLO-based components.
- Human parsing and person segmentation.
- SAM2-based segmentation support.
- Segmentation refinement and quality evaluation.

These components are configurable and are not all enabled by default.

The purpose is to reduce background interference and improve the visual representation of the garment.

See [docs/SEGMENTATION.md](docs/SEGMENTATION.md) for the implementation details and configuration.

### 2.8 Multi-Crop Retrieval (Optional)

StyleFindAI also supports experimental multi-crop search strategies.

Instead of relying on a single image representation, multiple image crops can be evaluated.

Supported strategies include:

- `off`
- `max_score`
- `mean_embedding`

These strategies should be enabled only when retrieval evaluation demonstrates an improvement.

### 2.9 Model Evaluation & Experimentation

StyleFindAI includes scripts for comparing different encoders, retrieval configurations and vector-index settings.

Evaluation and experimentation include:

- Comparing FashionCLIP, OpenCLIP and DINOv2.
- Evaluating different candidate retrieval sizes.
- Comparing retrieval experiments.
- Tuning hybrid ranking weights.
- Benchmarking HNSW vector-index behaviour.
- Evaluating segmentation quality.
- Generating evaluation queries with background, crop, angle and lighting variations.

The evaluation scripts help measure whether a model or configuration change actually improves retrieval rather than relying only on visual assumptions.

See the following documentation:

- [Model Documentation](docs/MODELS.md)
- [Dataset Documentation](docs/DATASET.md)
- [Segmentation Documentation](docs/SEGMENTATION.md)
- [Phase Report](docs/PHASE_REPORT.md)

---

## 3. What You Need

| Tool | Version | PowerShell check |
|---|---|---|
| Python | 3.11.x | `py -3.11 --version` |
| Node.js | 22 LTS (20+ works) | `node --version` |
| Docker Desktop | Recent version | `docker --version` |
| Disk space | Approximately 6 GB free | |
| RAM | 16 GB recommended | |

Run the automatic environment check:

```powershell
cd D:\Projects\StyleFindAI

powershell -ExecutionPolicy Bypass -File scripts\check-environment.ps1
```

> **Why Docker for the database?** The pgvector extension is not included in a standard PostgreSQL Windows installation. The `pgvector/pgvector:pg16` Docker image provides PostgreSQL with pgvector already installed.

---

## 4. First-Time Setup

All commands below are intended for Windows PowerShell.

### 4.1 Configuration File

Create your local environment configuration:

```powershell
cd D:\Projects\StyleFindAI

Copy-Item .env.example .env

notepad .env
```

Configure at least:

- `POSTGRES_PASSWORD`
- `JWT_SECRET`
- `AI_INTERNAL_TOKEN`

Use strong, private values and never commit your `.env` file to GitHub.

### 4.2 Start PostgreSQL

```powershell
docker compose up -d

docker compose ps
```

Wait for the database container to become healthy.

### 4.3 Set Up the Python AI Service

```powershell
cd ai-service

py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python scripts/migrate.py
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

### 4.4 Download the Fashion Dataset

Download **Fashion Product Images (Small)** from Kaggle:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Extract the dataset into:

```text
dataset/
└── raw/
    └── fashion-product-images-small/
        ├── styles.csv
        └── images/
```

A nested `myntradataset` folder is also supported.

Prepare and validate the dataset:

```powershell
python scripts/prepare_dataset.py --raw dataset/raw/fashion-product-images-small

python scripts/validate_images.py --min-count 1000

python scripts/dataset_statistics.py

python scripts/load_catalogue.py
```

The preparation script filters clothing images, removes duplicates, organizes products and prepares the catalogue.

See [docs/DATASET.md](docs/DATASET.md) for dataset details and category selection.

### 4.5 Generate Image Embeddings

Generate FashionCLIP embeddings for catalogue images:

```powershell
python scripts/generate_embeddings.py --model fashion-clip
```

The script processes catalogue images and stores their embeddings in PostgreSQL.

Only images without an existing embedding are processed, making the operation suitable for incremental use.

### 4.6 Start the AI Service

From the `ai-service` folder, with the Python environment activated:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Keep this terminal open.

Open another PowerShell terminal and test the service:

```powershell
curl.exe http://127.0.0.1:8001/health
```

Interactive API documentation:

http://127.0.0.1:8001/docs

### 4.7 Set Up the Backend

```powershell
cd D:\Projects\StyleFindAI\backend

npm install
```

Create an administrator account with your own password:

```powershell
npm run create-admin -- --email admin@stylefind.local --password "choose-a-long-password"
```

### 4.8 Set Up the Frontend

```powershell
cd D:\Projects\StyleFindAI\frontend

npm install
```

---

## 5. Run the Application

After completing the first-time setup, start the application using:

```powershell
cd D:\Projects\StyleFindAI

docker compose up -d

powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1
```

The development script starts the AI service, backend and frontend.

Open the application:

**Frontend:** http://localhost:5173

Other endpoints:

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Admin Login | http://localhost:5173/admin/login |
| Backend Health | http://localhost:4000/api/health |
| AI Service Docs | http://127.0.0.1:8001/docs |

To stop the application, close the service terminals and run:

```powershell
docker compose stop
```

---

## 6. Testing & Evaluation

StyleFindAI includes automated tests and evaluation scripts for validating the backend, AI service, dataset and search functionality.

### Python Tests

From the `ai-service` directory:

```powershell
.\.venv\Scripts\Activate.ps1

python -m pytest tests -v
```

### Retrieval Evaluation

Generate evaluation queries:

```powershell
python scripts/make_eval_queries.py --n 60
```

Evaluate the primary model:

```powershell
python scripts/evaluate.py --models fashion-clip --tag baseline
```

Compare different candidate counts:

```powershell
python scripts/evaluate.py --models fashion-clip --candidates 50 100 200 --tag candidates
```

Generate alternative model embeddings:

```powershell
python scripts/generate_embeddings.py --model openclip-b32 --no-features

python scripts/generate_embeddings.py --model dinov2-b14 --no-features
```

Compare models:

```powershell
python scripts/evaluate.py --models fashion-clip openclip-b32 dinov2-b14 --tag models
```

Run additional experiments:

```powershell
python scripts/compare_experiments.py

python scripts/benchmark_vector_index.py

python scripts/tune_weights.py --model fashion-clip
```

### Backend Tests

With the AI service running:

```powershell
cd D:\Projects\StyleFindAI\backend

npm test
```

### Browser End-to-End Tests

With all three services running:

```powershell
cd D:\Projects\StyleFindAI\tests

npm install

npx playwright install chromium
```

Then run:

```powershell
node ui-smoke.mjs --query ..\dataset\evaluation\queries\<query-file>.jpg --email admin@stylefind.local --password "<your-password>"
```

Replace the query filename and password with your actual values.

---

## 7. Configuration Highlights

Important environment variables in `.env`:

| Variable | Purpose |
|---|---|
| `AI_PRIMARY_MODEL` | Primary encoder: `fashion-clip`, `openclip-b32`, `dinov2-b14` or `cv-baseline` |
| `RANK_W_VISUAL` | Visual similarity ranking weight |
| `RANK_W_COLOUR` | Colour similarity ranking weight |
| `RANK_W_PATTERN` | Pattern similarity ranking weight |
| `RANK_W_ATTRIBUTE` | Attribute similarity ranking weight |
| `SEARCH_CANDIDATES` | Number of vector-search candidates to re-rank |
| `ENABLE_GARMENT_DETECTION` | Optional garment detection |
| `ENABLE_SAM2` | Optional SAM2 segmentation support |
| `ENABLE_VLM_ATTRIBUTES` | Optional vision-language attribute extraction |
| `STORE_SEARCH_HISTORY` | Controls anonymous search logging |
| `HNSW_EF_SEARCH` | HNSW search configuration |
| `HNSW_ITERATIVE_SCAN` | Iterative vector-index scanning configuration |
| `FILTERED_EXACT_FALLBACK` | Controls exact-search fallback behaviour |
| `SEARCH_MULTICROP` | Multi-crop strategy: `off`, `max_score` or `mean_embedding` |
| `GARMENT_ISOLATION` | Garment-isolation strategy |
| `PERSON_SEGMENTER` | Optional person segmentation model |
| `HUMAN_PARSER` | Optional human parsing model |
| `GARMENT_DETECTOR` | Optional garment detection model |
| `POSE_ESTIMATOR` | Optional pose estimation component |
| `FASHION_CLIP_REVISION` | Optional FashionCLIP model revision |
| `AI_MAX_LOADED_MODELS` | Maximum number of encoders kept in memory |

See [docs/MODELS.md](docs/MODELS.md) and [docs/SEGMENTATION.md](docs/SEGMENTATION.md) for more configuration information.

---

## 8. Integrating with ThreadKart

StyleFindAI is designed as a modular visual-search service that can be integrated into a larger fashion-commerce application such as ThreadKart.

The Python AI service communicates through HTTP and JSON APIs.

ThreadKart can integrate with either:

- `POST /api/search/image` through the Node.js backend.
- `POST /ai/search` directly through the Python AI service.

The catalogue can be loaded using the existing dataset and embedding-generation scripts.

StyleFindAI is maintained independently and does not directly modify ThreadKart files.

See [docs/API.md](docs/API.md) for API details.

---

## 9. Security, Backups & Deployment

### Security

- Environment secrets are stored in `.env`, not source code.
- `.env` is excluded from Git using `.gitignore`.
- Admin sessions use HttpOnly cookies and CSRF protection.
- Logout revokes tokens.
- The API also supports bearer-mode authentication for scripts.

### Database Backups

Create a database backup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\backup-db.ps1
```

Restore a backup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\restore-db.ps1 -File <backup-file>
```

Always verify the backup and target database before restoring.

### Production-Style Startup

For a production-style local startup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-prod.ps1
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the deployment checklist and HTTPS configuration.

---

## 10. Native PostgreSQL Setup (Without Docker)

If you prefer not to use Docker:

1. Install PostgreSQL 16 from https://www.postgresql.org/.
2. Build and install pgvector using the Windows instructions: https://github.com/pgvector/pgvector#windows.
3. Create the database:

```sql
CREATE DATABASE stylefind;
```

4. Configure the database connection in `.env`.
5. Run the database migration:

```powershell
cd ai-service

python scripts/migrate.py
```

---

## 11. Troubleshooting

| Problem | Suggested solution |
|---|---|
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| AI search service is not running | Start the FastAPI service using Uvicorn |
| HTTP 409: No catalogue embeddings | Generate embeddings for the selected model |
| Connection refused on port 5432 | Start Docker Compose and wait for PostgreSQL to become healthy |
| First search is slow | The model may need to load into memory on the first request |
| No relevant products are returned | Check the catalogue, generated embeddings and selected primary model |

---

## 12. Documentation

| Document | Description |
|---|---|
| [START_HERE.md](START_HERE.md) | Beginner-friendly setup and upgrade guide |
| [CHANGELOG.md](CHANGELOG.md) | Project changes and improvements |
| [docs/API.md](docs/API.md) | API reference |
| [docs/MODELS.md](docs/MODELS.md) | AI model information |
| [docs/DATASET.md](docs/DATASET.md) | Dataset preparation and organization |
| [docs/CLOTHING_DATASET.md](docs/CLOTHING_DATASET.md) | Clothing dataset instructions |
| [docs/GENERATED_DRESSES.md](docs/GENERATED_DRESSES.md) | Generated dress dataset |
| [docs/SEGMENTATION.md](docs/SEGMENTATION.md) | Garment segmentation pipeline |
| [docs/PHASE_REPORT.md](docs/PHASE_REPORT.md) | Development and implementation report |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide |

---

## 13. Future Improvements

Potential areas for further development include:

- Improving retrieval accuracy through larger and more diverse fashion datasets.
- Further experimentation with vision-language models and image encoders.
- Improving garment segmentation and background removal.
- Optimizing vector retrieval and hybrid ranking.
- Evaluating multimodal search combining image and text queries.
- Deploying the application to a cloud platform.
- Improving scalability and response times for larger product catalogues.

---

## About the Project

**StyleFindAI** explores the practical application of artificial intelligence, deep learning, computer vision and full-stack development to solve a real-world fashion discovery problem.

It combines a modular AI architecture with a complete web application, database-backed visual retrieval, model evaluation and experimentation.

**Core areas:** Artificial Intelligence · Deep Learning · Computer Vision · Vision-Language Models · Image Embeddings · Vector Databases · Semantic Search · Full-Stack Development
