import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type pg from "pg";
import { withTransaction, query } from "../config/db.js";
import { env } from "../config/env.js";
import { ApiError } from "../middleware/errors.js";
import { inspectImage } from "../middleware/upload.js";
import { aiIndex } from "./aiClient.js";
import { ATTRIBUTE_FILTERS, getProduct } from "./productService.js";

/** Admin uploads live in their own sub-folder so dataset images are never overwritten. */
const ADMIN_DIR = "admin-uploads";

export interface ProductInput {
  productCode?: string;
  name: string;
  category: string;
  subcategory?: string | null;
  price?: number | null;
  description?: string | null;
  colour?: string;
  pattern?: string;
  sleeveType?: string;
  neckline?: string;
  isActive?: boolean;
}

async function saveImage(buf: Buffer): Promise<{ key: string; width: number; height: number; sha256: string }> {
  const { type, width, height } = inspectImage(buf); // signature, corrupt check, dimensions, pixel budget
  const sha256 = crypto.createHash("sha256").update(buf).digest("hex");
  const dup = await query("SELECT p.product_code FROM product_images pi JOIN products p ON p.id = pi.product_id WHERE pi.sha256 = $1", [sha256]);
  if (dup.rowCount) throw new ApiError(409, `This exact image is already used by product ${dup.rows[0].product_code}`);
  // Server-generated filename - never trust the uploaded filename.
  const key = `${ADMIN_DIR}/${crypto.randomUUID()}.${type === "jpeg" ? "jpg" : type}`;
  const abs = path.join(env.productsDir, key);
  await fs.mkdir(path.dirname(abs), { recursive: true });
  await fs.writeFile(abs, buf, { flag: "wx" });
  return { key, width, height, sha256 };
}

const removeFile = (key: string) => fs.rm(path.join(env.productsDir, key), { force: true });

async function categoryId(client: pg.PoolClient, slug: string): Promise<number> {
  const r = await client.query("SELECT id FROM categories WHERE slug = $1", [slug]);
  if (!r.rowCount) throw new ApiError(400, `Unknown category '${slug}'`);
  return r.rows[0].id;
}

async function writeAttributes(client: pg.PoolClient, productId: number, input: ProductInput) {
  const values: Record<string, string | undefined> = {
    colour: input.colour, pattern: input.pattern, sleeve_type: input.sleeveType, neckline: input.neckline,
  };
  for (const attr of ATTRIBUTE_FILTERS) {
    const v = values[attr];
    if (v === undefined) continue;
    await client.query(
      `INSERT INTO product_attributes (product_id, attribute, value, source, verified)
       VALUES ($1, $2, $3, 'admin', TRUE)
       ON CONFLICT (product_id, attribute, source) DO UPDATE SET value = EXCLUDED.value, updated_at = now()`,
      [productId, attr, v.trim().toLowerCase() || "unknown"],
    );
  }
}

async function setIndexStatus(productId: number, status: "pending" | "failed", error: string | null) {
  try {
    await query(
      `UPDATE product_images SET index_status = $2, index_error = $3 WHERE product_id = $1 AND is_primary`,
      [productId, status, error?.slice(0, 500) ?? null],
    );
  } catch {
    /* migration 003 not applied yet - status columns missing */
  }
}

/**
 * Generates the embedding right away. If the AI service is down or fails, the product is saved but
 * clearly marked 'failed' (admin panel shows it with a Retry button) - never silently "searchable".
 */
async function tryIndex(productId: number, force = false) {
  try {
    const r = await aiIndex([productId], { force });
    if (r.indexed > 0) return { status: "generated" as const, model: r.model_key, indexed: r.indexed };
    if ((r as { failed?: number }).failed) return { status: "failed" as const, error: "the AI service could not read this image" };
    return { status: "generated" as const, model: r.model_key, indexed: 0, note: "already indexed" };
  } catch (e) {
    const msg = e instanceof ApiError ? e.message : "AI service unavailable";
    await setIndexStatus(productId, "failed", msg);
    return { status: "failed" as const, error: msg };
  }
}

export async function reindexProduct(id: number) {
  const exists = await query("SELECT 1 FROM products WHERE id = $1", [id]);
  if (!exists.rowCount) throw new ApiError(404, "Product not found");
  await setIndexStatus(id, "pending", null);
  return { productId: id, embedding: await tryIndex(id, true) };
}

export async function indexingFailures() {
  try {
    const r = await query(`
      SELECT p.id, p.product_code, p.name, pi.index_status, pi.index_error, pi.updated_at
      FROM product_images pi JOIN products p ON p.id = pi.product_id
      WHERE pi.is_primary AND pi.index_status <> 'indexed'
      ORDER BY (pi.index_status = 'failed') DESC, pi.updated_at DESC LIMIT 100`);
    return r.rows.map((x) => ({ id: Number(x.id), productCode: x.product_code, name: x.name, status: x.index_status,
                                error: x.index_error, updatedAt: x.updated_at }));
  } catch {
    return [];
  }
}

export async function createProduct(input: ProductInput, image: Buffer) {
  const code = input.productCode?.trim() || `adm-${crypto.randomUUID().slice(0, 8)}`;
  const exists = await query("SELECT 1 FROM products WHERE product_code = $1", [code]);
  if (exists.rowCount) throw new ApiError(409, `Product code '${code}' already exists`);
  const img = await saveImage(image);
  let id: number;
  try {
    id = await withTransaction(async (c) => {
      const cat = await categoryId(c, input.category);
      const p = await c.query(
        `INSERT INTO products (product_code, name, category_id, subcategory, price, description, source_dataset,
                               licence_information, is_active)
         VALUES ($1,$2,$3,$4,$5,$6,'admin_upload','Uploaded by StyleFind admin - owner must hold image rights',$7)
         RETURNING id`,
        [code, input.name, cat, input.subcategory ?? null, input.price ?? null, input.description ?? null, input.isActive ?? true],
      );
      const pid = Number(p.rows[0].id);
      await c.query(
        `INSERT INTO product_images (product_id, storage_key, is_primary, width, height, sha256) VALUES ($1,$2,TRUE,$3,$4,$5)`,
        [pid, img.key, img.width, img.height, img.sha256],
      );
      await writeAttributes(c, pid, input);
      return pid;
    });
  } catch (e) {
    await removeFile(img.key); // undo the file if the DB insert failed
    throw e;
  }
  const embedding = await tryIndex(id);
  return { product: await getProduct(id, true), embedding };
}

export async function updateProduct(id: number, input: Partial<ProductInput>, image?: Buffer) {
  const current = await query("SELECT id FROM products WHERE id = $1", [id]);
  if (!current.rowCount) throw new ApiError(404, "Product not found");
  const img = image ? await saveImage(image) : null;
  let oldKey: string | null = null;
  try {
    await withTransaction(async (c) => {
      const sets: string[] = [];
      const vals: unknown[] = [];
      const set = (col: string, v: unknown) => (vals.push(v), sets.push(`${col} = $${vals.length}`));
      if (input.name !== undefined) set("name", input.name);
      if (input.category !== undefined) set("category_id", await categoryId(c, input.category));
      if (input.subcategory !== undefined) set("subcategory", input.subcategory);
      if (input.price !== undefined) set("price", input.price);
      if (input.description !== undefined) set("description", input.description);
      if (input.isActive !== undefined) set("is_active", input.isActive);
      if (sets.length) {
        vals.push(id);
        await c.query(`UPDATE products SET ${sets.join(", ")} WHERE id = $${vals.length}`, vals);
      }
      await writeAttributes(c, id, input as ProductInput);
      if (img) {
        const old = await c.query("SELECT id, storage_key FROM product_images WHERE product_id = $1 AND is_primary FOR UPDATE", [id]);
        if (old.rowCount) {
          oldKey = old.rows[0].storage_key;
          // New image -> old embeddings and colour features describe the OLD picture: remove them in the
          // same transaction (migration 003 also enforces this with a trigger).
          await c.query("DELETE FROM image_embeddings WHERE product_image_id = $1", [old.rows[0].id]);
          await c.query("DELETE FROM image_features WHERE product_image_id = $1", [old.rows[0].id]);
          await c.query(
            "UPDATE product_images SET storage_key = $1, width = $2, height = $3, sha256 = $4, phash = NULL WHERE id = $5",
            [img.key, img.width, img.height, img.sha256, old.rows[0].id],
          );
        } else {
          await c.query(
            `INSERT INTO product_images (product_id, storage_key, is_primary, width, height, sha256) VALUES ($1,$2,TRUE,$3,$4,$5)`,
            [id, img.key, img.width, img.height, img.sha256],
          );
        }
      }
    });
  } catch (e) {
    if (img) await removeFile(img.key); // transaction rolled back -> the new file must not be left behind
    throw e;
  }
  // Only after a successful commit is the old admin-uploaded file deleted (dataset images are kept).
  if (oldKey && (oldKey as string).startsWith(`${ADMIN_DIR}/`)) await removeFile(oldKey);
  const embedding = img ? await tryIndex(id) : undefined;
  return { product: await getProduct(id, true), ...(embedding ? { embedding } : {}) };
}

export async function deleteProduct(id: number) {
  const r = await query<{ storage_key: string | null }>(
    `DELETE FROM products p USING product_images pi WHERE p.id = $1 AND pi.product_id = p.id RETURNING pi.storage_key`, [id]);
  if (!r.rowCount) {
    const plain = await query("DELETE FROM products WHERE id = $1", [id]);
    if (!plain.rowCount) throw new ApiError(404, "Product not found");
  }
  // Only files uploaded through the admin panel are removed; dataset images are left untouched.
  for (const row of r.rows)
    if (row.storage_key?.startsWith(`${ADMIN_DIR}/`)) await removeFile(row.storage_key);
  return { deleted: id };
}

export async function catalogueStats() {
  const r = await query(`
    SELECT (SELECT count(*) FROM products) AS products,
           (SELECT count(*) FROM products WHERE is_active) AS active_products,
           (SELECT count(*) FROM product_images) AS images,
           (SELECT count(*) FROM search_history) AS searches,
           (SELECT count(*) FROM search_history WHERE created_at > now() - interval '7 days') AS searches_7d,
           (SELECT round(avg(latency_ms)) FROM search_history WHERE created_at > now() - interval '7 days') AS avg_latency_ms`);
  const byCat = await query(`SELECT c.slug, c.name, count(p.id) AS n FROM categories c
                             LEFT JOIN products p ON p.category_id = c.id GROUP BY c.id ORDER BY c.id`);
  const emb = await query(`SELECT m.model_key, m.model_version, m.dims, count(e.id) AS n FROM embedding_models m
                           LEFT JOIN image_embeddings e ON e.model_key = m.model_key AND e.model_version = m.model_version
                           GROUP BY 1,2,3 ORDER BY 1`);
  const missing = await query(`SELECT count(*) AS n FROM product_images pi JOIN products p ON p.id = pi.product_id
                               WHERE p.is_active AND NOT EXISTS (SELECT 1 FROM image_embeddings e
                               WHERE e.product_image_id = pi.id AND e.model_key = $1)`, [env.AI_PRIMARY_MODEL]);
  const sources = await query(`SELECT source_dataset, count(*) AS n FROM products GROUP BY 1 ORDER BY 2 DESC`);
  let failedIndexing = 0;
  try {
    failedIndexing = Number((await query("SELECT count(*) AS n FROM product_images WHERE index_status = 'failed'")).rows[0].n);
  } catch {
    /* migration 003 not applied */
  }
  const num = (v: unknown) => (v === null ? null : Number(v));
  const t = r.rows[0];
  return {
    products: num(t.products), activeProducts: num(t.active_products), images: num(t.images),
    searches: num(t.searches), searchesLast7Days: num(t.searches_7d), avgSearchLatencyMs: num(t.avg_latency_ms),
    primaryModel: env.AI_PRIMARY_MODEL, missingPrimaryEmbeddings: num(missing.rows[0].n), failedIndexing,
    byCategory: byCat.rows.map((x) => ({ slug: x.slug, name: x.name, count: Number(x.n) })),
    embeddings: emb.rows.map((x) => ({ model: x.model_key, version: x.model_version, dims: x.dims, count: Number(x.n) })),
    sources: sources.rows.map((x) => ({ source: x.source_dataset, count: Number(x.n) })),
  };
}
