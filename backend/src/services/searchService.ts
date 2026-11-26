import { withTransaction } from "../config/db.js";
import { env } from "../config/env.js";
import { aiSearch, type AiResult, type SearchParams } from "./aiClient.js";
import { mediaUrl } from "./productService.js";

export function toApiResult(r: AiResult) {
  return {
    id: r.product_id,
    productCode: r.product_code,
    name: r.name,
    category: r.category,
    categoryName: r.category_name,
    subcategory: r.subcategory,
    colour: r.colour,
    pattern: r.pattern,
    sleeveType: r.sleeve_type,
    neckline: r.neckline,
    price: r.price,
    imageUrl: mediaUrl(r.image_key),
    matchType: r.match_type,
    scores: r.scores,
    reasons: r.reasons,
  };
}

/**
 * Runs the AI search and (optionally) logs an anonymous record of it.
 * Privacy: only a SHA-256 of the image is stored - never the image, IP address or any user identity.
 */
export async function runImageSearch(params: SearchParams) {
  const t0 = performance.now();
  const ai = await aiSearch(params);
  const results = ai.results.map(toApiResult);
  let searchId: string | null = null;
  if (env.STORE_SEARCH_HISTORY) {
    try {
      searchId = await withTransaction(async (c) => {
        const h = await c.query(
          `INSERT INTO search_history (model_key, query_image_sha256, text_query, predicted_attributes, filters, latency_ms, result_count)
           VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id`,
          [ai.model.key, ai.query.sha256, params.text ?? null, JSON.stringify(ai.query.predicted_attributes),
           JSON.stringify(params.filters ?? {}), Math.round(ai.timings_ms.total_ms ?? 0), results.length],
        );
        const id = h.rows[0].id as string;
        for (const [i, r] of ai.results.entries())
          await c.query(
            `INSERT INTO search_results (search_id, rank, product_id, final_score, visual_score, match_type) VALUES ($1,$2,$3,$4,$5,$6)`,
            [id, i + 1, r.product_id, r.scores.final ?? 0, r.scores.visual ?? 0, r.match_type],
          );
        return id;
      });
    } catch (e) {
      console.warn("[search] could not store history:", (e as Error).message); // never fail the search for logging
    }
  }
  const { sha256: _omit, ...query } = ai.query;
  const timingsMs = { ...ai.timings_ms, backend_total_ms: Math.round((performance.now() - t0) * 10) / 10 };
  return { searchId, model: ai.model, query, filters: params.filters ?? {}, candidatesConsidered: ai.candidates_considered,
           retrieval: (ai as { retrieval?: unknown }).retrieval ?? null, colourRequirement: ai.colour_requirement ?? null,
           timingsMs, count: results.length, results };
}
