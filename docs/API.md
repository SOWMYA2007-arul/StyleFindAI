# StyleFind AI - API reference

All responses are JSON. Errors look like `{"error": "message"}` (plus `details` for validation errors).
Internal file-system paths are never returned; images are addressed as `/media/<category>/<file>`.

## Node.js backend (`http://localhost:4000`)

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/health` | - | DB + AI service status |
| POST | `/api/search/image` | - (30/min) | multipart: `image` (jpg/png/webp, ≤ MAX_UPLOAD_MB), optional `text`, `model`, `top_k` (1-50, default 20), `filters` (JSON `{"category":[...],"colour":[...],"pattern":[...],"sleeve_type":[...],"neckline":[...]}`), `crop` (`true`/`false`), `target` (optional, with `crop`: which garment, e.g. `upper`, `lower`, `dupatta`, or an index) |
| GET | `/api/products` | - | query: `category`, `colour`, `pattern`, `sleeve_type`, `neckline` (comma lists), `q`, `min_price`, `max_price`, `sort` (`relevance,name,newest,price_asc,price_desc`), `page`, `limit` (≤100) |
| GET | `/api/products/:id` | - | product details |
| GET | `/api/products/:id/similar` | - | visually similar products (uses the stored embedding) |
| GET | `/api/categories` | - | categories with counts and a cover image |
| GET | `/api/filters` | - | facet values with counts, price range if prices exist |
| POST | `/api/auth/login` | - (10/15 min) | `{email, password}` → sets HttpOnly `sf_session` cookie + readable `sf_csrf` cookie, returns `{email, csrfToken}`. `?mode=bearer` returns `{token}` instead (scripts / API clients) |
| GET | `/api/auth/session` | - | `{authenticated, email?}` - always 200 (used by the admin UI) |
| POST | `/api/auth/logout` | admin | revokes all tokens of this admin (`token_version`), clears cookies |
| GET | `/api/auth/me` | admin | current admin + csrf token |
| GET | `/api/admin/stats` | admin | catalogue + index + search statistics |
| GET | `/api/admin/models` | admin | available encoders and embedding counts |
| GET | `/api/admin/products` | admin | like `/api/products`, includes hidden products and `hasEmbedding` |
| POST | `/api/admin/products` | admin | multipart: `image` + `name`, `category` (slug), optional `productCode`, `subcategory`, `price`, `description`, `colour`, `pattern`, `sleeveType`, `neckline`, `isActive`. Saves the image under a server-generated name and generates the embedding. 409 on duplicate code or identical image |
| PUT | `/api/admin/products/:id` | admin | same fields, all optional; a new `image` replaces the old one, deletes stale embeddings and re-indexes |
| DELETE | `/api/admin/products/:id` | admin | deletes the product (admin-uploaded image files are removed; dataset images are kept on disk) |
| POST | `/api/admin/embeddings/generate` | admin | `{productIds?: number[], force?: boolean, model?: string}` - embed missing (or all, with force) |
| POST | `/api/admin/products/:id/reindex` | admin | regenerate this product's embedding now |
| GET | `/api/admin/indexing-failures` | admin | products whose primary image is `pending` / `failed`, with the error |

Admin routes accept the session cookie (browser; state-changing requests must send `X-CSRF-Token` equal to the
`sf_csrf` cookie, otherwise 403) or `Authorization: Bearer <token>` (scripts; no CSRF needed).

Search `filters` also accept `min_price` / `max_price` (numbers). Filters are applied inside the vector search;
if a selective filter leaves fewer than N candidates, an exact comparison of all matching products is used
(`retrieval.mode = "exact_fallback"`). The response echoes the applied `filters`, includes `retrieval`
(`mode`, `ef_search`, `returned`) and `timingsMs` per stage; the same timings are in the `Server-Timing` header.
Every response carries `X-Request-Id` (also forwarded to the AI service).

### Search response (abridged)

```json
{
  "searchId": "uuid or null",
  "model": {"key": "fashion-clip", "version": "hf:patrickjohncyh/fashion-clip", "dims": 512},
  "query": {
    "predicted_attributes": {"colour": {"value": "navy blue", "confidence": 0.62, "source": "pixels"},
                             "category": {"value": "kurtas-kurtis", "confidence": 0.71, "source": "zero_shot"}},
    "palette": [{"lab": [30.1, 8.2, -30.5], "weight": 0.7, "name": "navy blue"}],
    "requested_attributes": {"colour": "blue"},
    "composition": {"method": "text_delta"},
    "quality_warnings": [],
    "garment_isolation": {"method": "full_image"}
  },
  "candidatesConsidered": 50,
  "timingsMs": {"embed_ms": 140.2, "vector_search_ms": 4.1, "total_ms": 260.3},
  "count": 20,
  "results": [{
    "id": 12, "productCode": "fpi-15970", "name": "...", "category": "kurtas-kurtis",
    "colour": "navy blue", "pattern": "printed", "imageUrl": "/media/kurtas-kurtis/15970.jpg",
    "matchType": "exact | similar | related",
    "scores": {"final": 0.83, "visual": 0.91, "visual_cosine": 0.955, "colour": 0.88, "pattern": 0.74, "attribute": 1.0},
    "reasons": ["Similar overall look (visual score 0.91)", "Similar colours (navy blue)"]
  }]
}
```

`scores.final` is a ranking score in [0, 1], not a probability. A null component was not available and was left out of the weighted sum.

## Python AI service (`http://127.0.0.1:8001`, Swagger at `/docs`)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | DB, pgvector version, products, embeddings per model, `outdated_features` (+ warning) |
| POST | `/ai/embed` | multipart `image`, `model` → embedding vector |
| POST | `/ai/search` | multipart `image`, `text`, `model`, `top_k`, `filters`, `crop`, optional `candidates` (top_k..500), `ef_search` (1..1000) and `target` (with `crop`: instance index or garment word such as `upper`, `lower`, `dupatta`) → ranked results (snake_case version of the above, with `image_key`; `query.garment_isolation.segmentation` holds the segmentation result) |
| POST | `/ai/segment` | multipart `image`, optional `target`, `strategy` (`auto`/`light`/`off`), `include_masks` → route, garment instances (label, group, confidence, bbox, occlusion, optional base64 PNG mask), target, fallbacks, per-stage timings (see docs/SEGMENTATION.md) |
| GET | `/ai/segmentation/status` | configured / loaded segmentation backends and cache statistics |
| POST | `/ai/extract-attributes` | multipart `image`, `model`, `crop` → predicted attributes + palette |
| POST | `/ai/compare-models` | multipart `image`, `top_k` → results from every model that has catalogue embeddings |
| GET | `/ai/products/{id}/similar` | similar products from stored embedding |
| POST | `/ai/index` | header `x-internal-token`; `{product_ids, model, force}` → embeds images (used by the admin panel) |
| GET | `/ai/models` | encoders, dims, text support, catalogue embedding counts |

A model that cannot be loaded (missing weights, failed download, wrong dimensions) returns **503** with an
explanation instead of a crash, and is not retried for 60 s.

Status codes: 200 OK, 201 created, 400/422 invalid input, 401 auth, 404 not found, 409 conflict / no embeddings,
413 too large, 415 wrong file type, 429 rate limited, 502/503/504 AI service problems, 500 internal (details only in server logs).


### Colour typed in the search text (v3.4)

When the text names a colour ("same dress in green"), it is a requirement: `/ai/search` and `/api/search/image`
return only products in that colour family, and add

```json
"colour_requirement": {"colour": "green", "category": "western-dresses", "checked": 165, "kept": 43, "removed": 122}
```

(`colourRequirement` in the backend response). `category` is the garment type that was also required (typed, or
kept from the photo), `null` if none. The field is absent / `null` when no colour was typed.
