import { useSearchParams } from "react-router";
import { CheckboxGroup } from "../components/CheckboxGroup";
import { EmptyState, ErrorNotice, Spinner } from "../components/Feedback";
import { ProductCard } from "../components/ProductCard";
import { useAsync } from "../hooks/useAsync";
import { api } from "../services/api";

const FILTER_KEYS = ["category", "colour", "pattern", "sleeve_type", "neckline"] as const;
const LABELS: Record<string, string> = { colour: "Colour", pattern: "Pattern", sleeve_type: "Sleeves", neckline: "Neckline" };

export function CataloguePage() {
  const [params, setParams] = useSearchParams();
  const get = (k: string) => params.get(k)?.split(",").filter(Boolean) ?? [];
  const page = Number(params.get("page") ?? 1);
  const sort = params.get("sort") ?? "relevance";
  const q = params.get("q") ?? "";

  const update = (k: string, v: string | string[]) => {
    const next = new URLSearchParams(params);
    const s = Array.isArray(v) ? v.join(",") : v;
    if (s) next.set(k, s); else next.delete(k);
    if (k !== "page") next.delete("page");
    setParams(next);
  };

  const query = Object.fromEntries(FILTER_KEYS.map((k) => [k, params.get(k) ?? undefined]));
  const list = useAsync(() => api.products({ ...query, q: q || undefined, sort, page, limit: 24 }), [params.toString()]);
  const cats = useAsync(() => api.categories(), []);
  const facets = useAsync(() => api.filters(), []);

  return (
    <section className="container-page py-10">
      <h1 className="text-4xl sm:text-5xl">Catalogue</h1>
      <div className="mt-8 grid gap-10 lg:grid-cols-[230px_1fr]">
        <aside className="flex flex-col gap-4" aria-label="Filters">
          <div>
            <label htmlFor="q" className="label">Search by name</label>
            <input id="q" className="field" defaultValue={q} placeholder="e.g. floral kurta"
              onKeyDown={(e) => e.key === "Enter" && update("q", (e.target as HTMLInputElement).value.trim())}
              onBlur={(e) => e.target.value.trim() !== q && update("q", e.target.value.trim())} />
          </div>
          <CheckboxGroup legend="Category" selected={get("category")} onChange={(v) => update("category", v)}
            options={(cats.data ?? []).map((c) => ({ value: c.slug, label: c.name, count: c.productCount }))} />
          {(["colour", "pattern", "sleeve_type", "neckline"] as const).map((k) => (
            <CheckboxGroup key={k} legend={LABELS[k]} selected={get(k)} onChange={(v) => update(k, v)}
              options={(facets.data?.attributes[k] ?? []).map((f) => ({ value: f.value, count: f.count }))} />
          ))}
          {facets.data && !facets.data.price && <p className="text-xs text-muted">Price filters appear once products have prices.</p>}
          {[...params.keys()].length > 0 && (
            <button type="button" className="btn-ghost" onClick={() => setParams(new URLSearchParams())}>Clear all filters</button>
          )}
        </aside>
        <div>
          <div className="flex items-center justify-between gap-4">
            <p className="text-sm text-muted">{list.data ? `${list.data.total} products` : " "}</p>
            <label className="text-sm">
              <span className="mr-2 text-muted">Sort</span>
              <select className="field inline-block w-auto" value={sort} onChange={(e) => update("sort", e.target.value)}>
                <option value="relevance">Featured</option>
                <option value="name">Name A to Z</option>
                <option value="newest">Newest</option>
                {facets.data?.price && <option value="price_asc">Price, low to high</option>}
                {facets.data?.price && <option value="price_desc">Price, high to low</option>}
              </select>
            </label>
          </div>
          {list.loading && <Spinner label="Loading products" />}
          {list.error && <div className="mt-4"><ErrorNotice message={list.error} onRetry={list.reload} /></div>}
          {list.data?.items.length === 0 && <div className="mt-6"><EmptyState title="No products match">Try removing a filter.</EmptyState></div>}
          <div className="mt-6 grid grid-cols-2 gap-x-5 gap-y-10 sm:grid-cols-3 xl:grid-cols-4">
            {list.data?.items.map((p) => <ProductCard key={p.id} p={p} />)}
          </div>
          {list.data && list.data.pages > 1 && (
            <nav className="mt-10 flex items-center justify-center gap-3" aria-label="Pages">
              <button type="button" className="btn-ghost" disabled={page <= 1} onClick={() => update("page", String(page - 1))}>Previous</button>
              <span className="text-sm text-muted">Page {page} of {list.data.pages}</span>
              <button type="button" className="btn-ghost" disabled={page >= list.data.pages} onClick={() => update("page", String(page + 1))}>Next</button>
            </nav>
          )}
        </div>
      </div>
    </section>
  );
}
