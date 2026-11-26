import axios, { AxiosError } from "axios";
import type { AdminStats, Category, Facets, IndexingFailure, ModelInfo, Paged, Product, SearchFilters, SearchResponse, SearchResult } from "../types";

/** Drop empty filter entries so the backend only receives real constraints. */
export function cleanFilters(f?: SearchFilters): SearchFilters | null {
  if (!f) return null;
  const out: SearchFilters = {};
  for (const [k, v] of Object.entries(f) as [keyof SearchFilters, unknown][]) {
    if (Array.isArray(v) ? v.length > 0 : typeof v === "number" && Number.isFinite(v)) (out as Record<string, unknown>)[k] = v;
  }
  return Object.keys(out).length ? out : null;
}

/**
 * Admin auth uses an HttpOnly session cookie set by the backend (JavaScript cannot read it).
 * For state-changing requests we copy the readable CSRF cookie into the X-CSRF-Token header.
 * Nothing secret is stored in localStorage / sessionStorage.
 */
export function csrfToken(): string | null {
  const m = document.cookie.match(/(?:^|;\s*)sf_csrf=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

export const http = axios.create({ baseURL: "/api", timeout: 120_000, withCredentials: true });
http.interceptors.request.use((cfg) => {
  const method = (cfg.method ?? "get").toUpperCase();
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    const t = csrfToken();
    if (t) cfg.headers["X-CSRF-Token"] = t;
  }
  return cfg;
});

/** Turns any request failure into one readable sentence for the UI. */
export function errorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    if (!err.response) return "Can't reach the StyleFind server. Check that the backend is running.";
    const data = err.response.data as { error?: string; details?: { field: string; message: string }[] };
    if (data?.details?.length) return `${data.error}: ${data.details.map((d) => `${d.field} ${d.message}`).join(", ")}`;
    return data?.error ?? `Request failed (${err.response.status})`;
  }
  return err instanceof Error ? err.message : "Something went wrong";
}

export type ListQuery = Record<string, string | number | undefined>;

export const api = {
  products: (q: ListQuery) => http.get<Paged<Product>>("/products", { params: q }).then((r) => r.data),
  product: (id: number) => http.get<Product>(`/products/${id}`).then((r) => r.data),
  similar: (id: number) => http.get<{ results: SearchResult[] }>(`/products/${id}/similar`).then((r) => r.data.results),
  categories: () => http.get<Category[]>("/categories").then((r) => r.data),
  filters: () => http.get<Facets>("/filters").then((r) => r.data),
  searchImage: (file: File, opts: { text?: string; topK?: number; crop?: boolean; filters?: SearchFilters }) => {
    const form = new FormData();
    form.append("image", file);
    if (opts.text) form.append("text", opts.text);
    form.append("top_k", String(opts.topK ?? 40));
    if (opts.crop) form.append("crop", "true");
    const f = cleanFilters(opts.filters);
    if (f) form.append("filters", JSON.stringify(f));
    return http.post<SearchResponse>("/search/image", form).then((r) => r.data);
  },

  login: (email: string, password: string) =>
    http.post<{ email: string; csrfToken: string }>("/auth/login", { email, password }).then((r) => r.data),
  logout: () => http.post("/auth/logout").then((r) => r.data),
  me: () => http.get<{ email: string }>("/auth/me").then((r) => r.data),
  session: () => http.get<{ authenticated: boolean; email?: string }>("/auth/session").then((r) => r.data),
  failures: () => http.get<IndexingFailure[]>("/admin/indexing-failures").then((r) => r.data),
  reindex: (id: number) => http.post<{ embedding: { status: string; error?: string } }>(`/admin/products/${id}/reindex`, {}, { timeout: 0 }).then((r) => r.data),
  stats: () => http.get<AdminStats>("/admin/stats").then((r) => r.data),
  models: () => http.get<ModelInfo[]>("/admin/models").then((r) => r.data),
  adminProducts: (q: ListQuery) => http.get<Paged<Product>>("/admin/products", { params: q }).then((r) => r.data),
  createProduct: (form: FormData) => http.post("/admin/products", form).then((r) => r.data),
  updateProduct: (id: number, form: FormData) => http.put(`/admin/products/${id}`, form).then((r) => r.data),
  deleteProduct: (id: number) => http.delete(`/admin/products/${id}`).then((r) => r.data),
  generateEmbeddings: (body: { productIds?: number[]; force?: boolean; model?: string }) =>
    http.post<{ model_key: string; indexed: number; seconds: number }>("/admin/embeddings/generate", body, { timeout: 0 }).then((r) => r.data),
};
