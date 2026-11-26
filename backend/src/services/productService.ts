import { query } from "../config/db.js";

export const ATTRIBUTE_FILTERS = ["colour", "pattern", "sleeve_type", "neckline"] as const;
export type AttrFilter = (typeof ATTRIBUTE_FILTERS)[number];

/** Storage keys are relative paths like "sarees/1234.jpg"; the browser loads them from /media/. */
export const mediaUrl = (key: string) => `/media/${key.split("/").map(encodeURIComponent).join("/")}`;

export interface ListParams {
  category?: string[];
  q?: string;
  attrs: Partial<Record<AttrFilter, string[]>>;
  minPrice?: number;
  maxPrice?: number;
  sort: "relevance" | "name" | "newest" | "price_asc" | "price_desc";
  page: number;
  limit: number;
  includeInactive?: boolean;
}

// Pivot verified attributes into columns (prefers metadata/admin over keyword-derived values).
const ATTR_PIVOT = `
  LEFT JOIN LATERAL (
    SELECT
      max(value) FILTER (WHERE attribute = 'colour')      AS colour,
      max(value) FILTER (WHERE attribute = 'pattern')     AS pattern,
      max(value) FILTER (WHERE attribute = 'sleeve_type') AS sleeve_type,
      max(value) FILTER (WHERE attribute = 'neckline')    AS neckline
    FROM (SELECT DISTINCT ON (attribute) attribute, value FROM product_attributes
          WHERE product_id = p.id AND verified
          ORDER BY attribute, CASE source WHEN 'admin' THEN 0 WHEN 'catalogue_metadata' THEN 1 ELSE 2 END) x
  ) a ON TRUE`;

const selectProduct = (withEmbeddingFlag = false) => `
  SELECT ${withEmbeddingFlag ? "EXISTS (SELECT 1 FROM image_embeddings e WHERE e.product_image_id = pi.id) AS has_embedding," : ""}
         p.id, p.product_code, p.name, p.subcategory, p.gender, p.price, p.description, p.is_active,
         p.source_dataset, p.licence_information, p.created_at, p.updated_at,
         c.slug AS category, c.name AS category_name, pi.storage_key,
         coalesce(a.colour,'unknown') AS colour, coalesce(a.pattern,'unknown') AS pattern,
         coalesce(a.sleeve_type,'unknown') AS sleeve_type, coalesce(a.neckline,'unknown') AS neckline
  FROM products p
  JOIN categories c ON c.id = p.category_id
  LEFT JOIN product_images pi ON pi.product_id = p.id AND pi.is_primary
  ${ATTR_PIVOT}`;

export function toApiProduct(r: Record<string, any>) {
  return {
    id: Number(r.id),
    productCode: r.product_code,
    name: r.name,
    category: r.category,
    categoryName: r.category_name,
    subcategory: r.subcategory,
    gender: r.gender ?? null,
    price: r.price ?? null,
    description: r.description ?? null,
    colour: r.colour,
    pattern: r.pattern,
    sleeveType: r.sleeve_type,
    neckline: r.neckline,
    imageUrl: r.storage_key ? mediaUrl(r.storage_key) : null,
    sourceDataset: r.source_dataset,
    licence: r.licence_information,
    isActive: r.is_active,
    ...(r.has_embedding !== undefined ? { hasEmbedding: r.has_embedding } : {}),
  };
}

export async function listProducts(p: ListParams) {
  const where: string[] = [];
  const params: unknown[] = [];
  const add = (v: unknown) => (params.push(v), `$${params.length}`);
  if (!p.includeInactive) where.push("p.is_active");
  if (p.category?.length) where.push(`c.slug = ANY(${add(p.category)})`);
  if (p.q) where.push(`p.name ILIKE ${add(`%${p.q.replace(/[%_\\]/g, (m) => "\\" + m)}%`)}`);
  for (const attr of ATTRIBUTE_FILTERS) {
    const vals = p.attrs[attr];
    if (vals?.length) where.push(`coalesce(a.${attr},'unknown') = ANY(${add(vals.map((v) => v.toLowerCase()))})`);
  }
  if (p.minPrice !== undefined) where.push(`p.price >= ${add(p.minPrice)}`);
  if (p.maxPrice !== undefined) where.push(`p.price <= ${add(p.maxPrice)}`);
  const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";
  const order = {
    relevance: "p.id",
    name: "p.name, p.id",
    newest: "p.created_at DESC, p.id DESC",
    price_asc: "p.price ASC NULLS LAST, p.id",
    price_desc: "p.price DESC NULLS LAST, p.id",
  }[p.sort];
  const total = await query<{ n: string }>(
    `SELECT count(*) AS n FROM products p JOIN categories c ON c.id = p.category_id ${ATTR_PIVOT} ${whereSql}`,
    params,
  );
  const rows = await query(
    `${selectProduct(Boolean(p.includeInactive))} ${whereSql} ORDER BY ${order} LIMIT ${add(p.limit)} OFFSET ${add((p.page - 1) * p.limit)}`,
    params,
  );
  const n = Number(total.rows[0].n);
  return { items: rows.rows.map(toApiProduct), page: p.page, limit: p.limit, total: n, pages: Math.max(1, Math.ceil(n / p.limit)) };
}

export async function getProduct(id: number, includeInactive = false) {
  const r = await query(`${selectProduct(includeInactive)} WHERE p.id = $1 ${includeInactive ? "" : "AND p.is_active"}`, [id]);
  if (!r.rows[0]) return null;
  // every image of the product (primary first) for the product page gallery
  const imgs = await query<{ storage_key: string; is_primary: boolean; width: number | null; height: number | null }>(
    `SELECT storage_key, is_primary, width, height FROM product_images WHERE product_id = $1 ORDER BY is_primary DESC, id`, [id]);
  return {
    ...toApiProduct(r.rows[0]),
    images: imgs.rows.map((i) => ({ url: mediaUrl(i.storage_key), isPrimary: i.is_primary, width: i.width, height: i.height })),
  };
}

export async function getFacets() {
  const attrs = await query<{ attribute: string; value: string; n: string }>(`
    SELECT attribute, value, count(DISTINCT product_id) AS n
    FROM product_attributes pa JOIN products p ON p.id = pa.product_id AND p.is_active
    WHERE verified AND value <> 'unknown' AND attribute = ANY($1)
    GROUP BY 1, 2 ORDER BY 1, 3 DESC`, [ATTRIBUTE_FILTERS]);
  const facets: Record<string, { value: string; count: number }[]> = {};
  for (const r of attrs.rows) (facets[r.attribute] ??= []).push({ value: r.value, count: Number(r.n) });
  const price = await query<{ min: number | null; max: number | null }>(
    "SELECT min(price) AS min, max(price) AS max FROM products WHERE is_active AND price IS NOT NULL",
  );
  return { attributes: facets, price: price.rows[0].min === null ? null : price.rows[0] };
}

export async function listCategories() {
  const r = await query(`
    SELECT c.id, c.slug, c.name, count(p.id) AS product_count,
           (SELECT pi.storage_key FROM products p2 JOIN product_images pi ON pi.product_id = p2.id AND pi.is_primary
             WHERE p2.category_id = c.id AND p2.is_active ORDER BY p2.id LIMIT 1) AS cover_key
    FROM categories c LEFT JOIN products p ON p.category_id = c.id AND p.is_active
    GROUP BY c.id ORDER BY c.id`);
  return r.rows.map((c) => ({
    id: c.id, slug: c.slug, name: c.name, productCount: Number(c.product_count),
    coverImageUrl: c.cover_key ? mediaUrl(c.cover_key) : null,
  }));
}
