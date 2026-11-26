import axios, { AxiosError } from "axios";
import { env } from "../config/env.js";
import { ApiError } from "../middleware/errors.js";

/** HTTP client for the Python AI service (see ai-service/app/api/routes.py). */
const http = axios.create({ baseURL: env.AI_SERVICE_URL, timeout: env.AI_TIMEOUT_MS });

export interface AiResult {
  product_id: number;
  product_code: string;
  name: string;
  category: string;
  category_name: string;
  subcategory: string | null;
  colour: string;
  pattern: string;
  sleeve_type: string;
  neckline: string;
  price: number | null;
  image_key: string;
  scores: Record<string, number | null>;
  match_type: "exact" | "similar" | "related";
  reasons: string[];
}

export interface ColourRequirement {
  colour: string;
  category: string | null;
  checked: number;
  kept: number;
  removed: number;
}

export interface AiSearchResponse {
  model: { key: string; version: string; name: string; dims: number };
  query: Record<string, unknown> & { sha256: string; predicted_attributes: Record<string, unknown> };
  candidates_considered: number;
  colour_requirement?: ColourRequirement | null;
  results: AiResult[];
  timings_ms: Record<string, number>;
}

function translate(err: unknown): never {
  if (err instanceof AxiosError) {
    if (err.code === "ECONNREFUSED" || err.code === "ECONNRESET")
      throw new ApiError(503, "The AI search service is not running. Start it and try again.");
    if (err.code === "ECONNABORTED") throw new ApiError(504, "The AI service took too long to respond");
    const status = err.response?.status ?? 502;
    const msg = (err.response?.data as { error?: string } | undefined)?.error;
    // 4xx from the AI service are user-facing validation messages; 5xx are hidden.
    if (status >= 400 && status < 500 && typeof msg === "string") throw new ApiError(status, msg);
    throw new ApiError(502, "The AI service failed to process the request");
  }
  throw err;
}

export interface SearchParams {
  requestId?: string;
  image: Buffer;
  filename: string;
  mimetype: string;
  text?: string;
  model?: string;
  topK?: number;
  filters?: Record<string, string[] | number>;
  crop?: boolean;
  target?: string;
}

export async function aiSearch(p: SearchParams): Promise<AiSearchResponse> {
  const form = new FormData();
  form.append("image", new Blob([new Uint8Array(p.image)], { type: p.mimetype }), "query" + (p.filename.match(/\.\w+$/)?.[0] ?? ".jpg"));
  if (p.text) form.append("text", p.text);
  if (p.model) form.append("model", p.model);
  if (p.topK) form.append("top_k", String(p.topK));
  if (p.filters && Object.keys(p.filters).length) form.append("filters", JSON.stringify(p.filters));
  if (p.crop) form.append("crop", "true");
  if (p.crop && p.target) form.append("target", p.target);
  try {
    const t = performance.now();
    const res = await http.post<AiSearchResponse>("/ai/search", form, { headers: p.requestId ? { "x-request-id": p.requestId } : {} });
    res.data.timings_ms = { ...res.data.timings_ms, backend_to_ai_roundtrip_ms: Math.round((performance.now() - t) * 10) / 10 };
    return res.data;
  } catch (e) {
    translate(e);
  }
}

async function getWithRetry<T>(url: string, config: Parameters<typeof http.get>[1] = {}): Promise<T> {
  try {
    return (await http.get<T>(url, config)).data;
  } catch (e) {
    // one retry for transient connection resets (GET is idempotent); real errors are not retried
    if (e instanceof AxiosError && e.code === "ECONNRESET") return (await http.get<T>(url, config)).data;
    throw e;
  }
}

export async function aiSimilar(productId: number, topK = 12): Promise<{ results: AiResult[] }> {
  try {
    return await getWithRetry(`/ai/products/${productId}/similar`, { params: { top_k: topK } });
  } catch (e) {
    translate(e);
  }
}

export async function aiIndex(productIds: number[], opts: { force?: boolean; model?: string } = {}) {
  try {
    return (
      await http.post(
        "/ai/index",
        { product_ids: productIds, force: opts.force ?? false, model: opts.model },
        { headers: { "x-internal-token": env.AI_INTERNAL_TOKEN }, timeout: Math.max(env.AI_TIMEOUT_MS, 600_000) },
      )
    ).data as { model_key: string; indexed: number; seconds: number };
  } catch (e) {
    translate(e);
  }
}

export async function aiHealth() {
  try {
    return (await http.get("/health", { timeout: 5000 })).data as Record<string, unknown>;
  } catch {
    return { status: "unreachable" };
  }
}

export async function aiModels() {
  try {
    return (await http.get("/ai/models")).data as unknown[];
  } catch (e) {
    translate(e);
  }
}
