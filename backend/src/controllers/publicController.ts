import type { Request, Response } from "express";
import { query } from "../config/db.js";
import { aiHealth, aiSimilar } from "../services/aiClient.js";
import { getFacets, getProduct, listCategories, listProducts } from "../services/productService.js";
import { runImageSearch, toApiResult } from "../services/searchService.js";
import { ApiError } from "../middleware/errors.js";
import { idParam, listQuerySchema, searchBodySchema } from "./validators.js";

export async function health(_req: Request, res: Response) {
  let db = "ok";
  try {
    await query("SELECT 1");
  } catch {
    db = "unavailable";
  }
  const ai = await aiHealth();
  res.status(db === "ok" ? 200 : 503).json({ status: db === "ok" && ai.status === "ok" ? "ok" : "degraded", database: db, ai });
}

export async function searchImage(req: Request, res: Response) {
  const body = searchBodySchema.parse(req.body ?? {});
  const file = req.file!;
  const out = await runImageSearch({
    image: file.buffer, filename: file.originalname, mimetype: file.mimetype, requestId: res.locals.requestId,
    text: body.text, model: body.model, topK: body.top_k, filters: body.filters, crop: body.crop, target: body.target,
  });
  // per-stage timings are visible in the browser's network tab (Server-Timing header)
  res.setHeader("Server-Timing", Object.entries(out.timingsMs).map(([k, v]) => `${k.replace(/_ms$/, "")};dur=${v}`).join(", "));
  res.json(out);
}

export async function products(req: Request, res: Response) {
  const q = listQuerySchema.parse(req.query);
  res.json(await listProducts({
    category: q.category, q: q.q, sort: q.sort, page: q.page, limit: q.limit, minPrice: q.min_price, maxPrice: q.max_price,
    attrs: { colour: q.colour, pattern: q.pattern, sleeve_type: q.sleeve_type, neckline: q.neckline },
  }));
}

export async function product(req: Request, res: Response) {
  const { id } = idParam.parse(req.params);
  const p = await getProduct(id);
  if (!p) throw new ApiError(404, "Product not found");
  res.json(p);
}

export async function similar(req: Request, res: Response) {
  const { id } = idParam.parse(req.params);
  if (!(await getProduct(id))) throw new ApiError(404, "Product not found");
  const r = await aiSimilar(id, 12);
  res.json({ productId: id, results: r.results.map(toApiResult) });
}

export async function categories(_req: Request, res: Response) {
  res.json(await listCategories());
}

export async function facets(_req: Request, res: Response) {
  res.json(await getFacets());
}
