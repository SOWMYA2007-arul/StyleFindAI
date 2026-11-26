import { Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { CheckboxGroup } from "../components/CheckboxGroup";
import { EmptyState, ErrorNotice } from "../components/Feedback";
import { titleCase } from "../components/ProductCard";
import { ResultCard } from "../components/ResultCard";
import { useSearch } from "../context/SearchContext";
import { useAsync } from "../hooks/useAsync";
import { api } from "../services/api";
import type { SearchFilters, SearchResult } from "../types";

type SortKey = "final" | "visual" | "colour" | "pattern" | "price_asc" | "price_desc";
const SORTS: { key: SortKey; label: string; needsPrice?: boolean }[] = [
  { key: "final", label: "Relevance (best overall match)" },
  { key: "visual", label: "Closest overall look" },
  { key: "colour", label: "Closest colour" },
  { key: "pattern", label: "Closest pattern" },
  { key: "price_asc", label: "Price, low to high", needsPrice: true },
  { key: "price_desc", label: "Price, high to low", needsPrice: true },
];
const ATTR_FILTERS: { key: "colour" | "pattern" | "sleeve_type" | "neckline"; legend: string }[] = [
  { key: "colour", legend: "Colour" }, { key: "pattern", legend: "Pattern" },
  { key: "sleeve_type", legend: "Sleeves" }, { key: "neckline", legend: "Neckline" },
];

function sortResults(list: SearchResult[], sort: SortKey) {
  const out = [...list];
  if (sort === "price_asc" || sort === "price_desc") {
    const dir = sort === "price_asc" ? 1 : -1;
    // products without a price go last, ties keep relevance order
    return out.sort((a, b) => (a.price === null ? 1 : b.price === null ? -1 : dir * (a.price - b.price)));
  }
  return out.sort((a, b) => (b.scores[sort] ?? -1) - (a.scores[sort] ?? -1));
}

/** CIELAB -> sRGB hex, used to draw the colours the model found in the photo. */
function labToHex([L, a, b]: number[]) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const f = (t: number) => (t ** 3 > 0.008856 ? t ** 3 : (t - 16 / 116) / 7.787);
  const [X, Y, Z] = [0.95047 * f(fx), f(fy), 1.08883 * f(fz)];
  const lin = [3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.204 * Y + 1.057 * Z];
  return "#" + lin.map((c) => {
    const v = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0");
  }).join("");
}

const ATTR_LABEL: Record<string, string> = {
  category: "Category", colour: "Main colour", secondary_colour: "Second colour", pattern: "Pattern",
  sleeve_type: "Sleeves", neckline: "Neckline", texture: "Surface",
};

export function SearchResultsPage() {
  const { result, previewUrl, filters, runSearch, loading, error, file } = useSearch();
  const [draft, setDraft] = useState<SearchFilters>(filters);
  const [sort, setSort] = useState<SortKey>("final");
  const cats = useAsync(() => api.categories(), []);
  const facets = useAsync(() => api.filters(), []);
  useEffect(() => setDraft(filters), [filters]);

  const hasPrices = useMemo(() => Boolean(facets.data?.price) || Boolean(result?.results.some((r) => r.price !== null)), [facets.data, result]);
  const shown = useMemo(() => (result ? sortResults(result.results, sort).slice(0, 20) : []), [result, sort]);
  const changed = JSON.stringify(draft) !== JSON.stringify(filters);
  const activeCount = Object.values(filters).filter((v) => (Array.isArray(v) ? v.length : v !== undefined)).length;
  const setList = (k: keyof SearchFilters) => (v: string[]) => setDraft({ ...draft, [k]: v });
  const setPrice = (k: "min_price" | "max_price", v: string) => setDraft({ ...draft, [k]: v === "" ? undefined : Math.max(0, Number(v)) });
  const apply = () => runSearch(draft);
  const clearAll = () => runSearch({});

  if (!result) {
    return (
      <section className="container-page py-16">
        <EmptyState title="No search yet">
          Upload a photo to see matching products. <Link to="/search" className="font-semibold text-indigo underline">Start a visual search</Link>
        </EmptyState>
      </section>
    );
  }

  const q = result.query;
  const predicted = Object.entries(q.predicted_attributes).filter(([, v]) => v.value && v.value !== "unknown");
  const exact = shown.filter((r) => r.matchType === "exact");
  const rest = shown.filter((r) => r.matchType !== "exact");

  return (
    <section className="container-page grid gap-10 py-10 lg:grid-cols-[300px_1fr]">
      <aside className="lg:sticky lg:top-24 lg:self-start">
        <div className="overflow-hidden rounded-xl bg-stone">
          {previewUrl ? <img src={previewUrl} alt="Your photo" className="max-h-80 w-full object-contain" /> : <p className="p-6 text-sm text-muted">Photo preview unavailable after reload.</p>}
        </div>
        <Link to="/search" className="btn-ghost mt-3 w-full">Search with another photo</Link>

        <h2 className="mt-8 font-sans text-sm font-semibold">What the AI saw in your photo</h2>
        {q.palette.length > 0 && (
          <div className="mt-3 flex h-6 overflow-hidden rounded-full" aria-label="Colours detected in the garment">
            {q.palette.map((p, i) => (
              <span key={i} title={`${p.name} ${Math.round(p.weight * 100)}%`} style={{ width: `${p.weight * 100}%`, background: labToHex(p.lab) }} />
            ))}
          </div>
        )}
        <dl className="mt-3 space-y-1.5 text-sm">
          {predicted.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-3">
              <dt className="text-muted">{ATTR_LABEL[k] ?? titleCase(k.replace(/_/g, " "))}</dt>
              <dd className="text-right font-medium">
                {k === "category" ? titleCase(v.value.replace(/-/g, " ")) : titleCase(v.value)}
                {v.confidence !== null && v.source !== "pixels" && <span className="ml-1 text-xs font-normal text-muted">{Math.round(v.confidence * 100)}%</span>}
              </dd>
            </div>
          ))}
        </dl>
        {Object.keys(q.requested_attributes).length > 0 && (
          <p className="mt-4 rounded-lg bg-indigo-soft p-3 text-sm">
            You asked for: {Object.entries(q.requested_attributes).map(([k, v]) => `${k.replace("_", " ")} ${v}`).join(", ")}
          </p>
        )}
        {q.composition?.note && <p className="mt-2 text-xs text-muted">{q.composition.note}</p>}
        {q.quality_warnings.map((w) => <p key={w} className="mt-3 rounded-lg bg-rani-soft p-3 text-xs text-rani">{w}</p>)}
        <p className="mt-6 text-xs leading-relaxed text-muted">
          Model: {result.model.name}. Compared against {result.candidatesConsidered} nearest catalogue items in{" "}
          {Math.round(result.timingsMs.total_ms)} ms. Garment isolation: {q.garment_isolation.method.replace(/_/g, " ")}.
        </p>
      </aside>

      <div>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl">Your matches</h1>
            <p className="mt-1 text-sm text-muted" aria-live="polite">
              {loading ? "Searching..." : `Showing ${shown.length} of ${result.count} products ranked for your photo`}
            </p>
            {!loading && result.colourRequirement && (
              <p className="mt-1 text-sm" data-testid="colour-requirement">
                {result.colourRequirement.kept > 0
                  ? <>Only <strong>{result.colourRequirement.colour}</strong>{" "}
                      {result.colourRequirement.category ? categoryLabel(result.colourRequirement.category) : "items"}
                      {" "}are shown, as you asked. Other colours were left out.</>
                  : <>No <strong>{result.colourRequirement.colour}</strong>{" "}
                      {result.colourRequirement.category ? categoryLabel(result.colourRequirement.category) : "items"}
                      {" "}were found in the catalogue. Try a nearby colour, or search without a colour.</>}
              </p>
            )}
          </div>
          <label className="text-sm">
            <span className="mr-2 text-muted">Sort by</span>
            <select className="field inline-block w-auto" value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
              {SORTS.filter((o) => !o.needsPrice || hasPrices).map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            </select>
          </label>
        </div>

        <details className="mt-6 rounded-xl bg-stone px-4 py-3" open={activeCount > 0 || undefined}>
          <summary className="flex cursor-pointer flex-wrap items-center justify-between gap-2 text-sm font-semibold">
            <span>Filter the catalogue search{activeCount > 0 && <span className="ml-2 font-normal text-muted">{activeCount} active</span>}</span>
            <span className="text-xs font-normal text-muted">Filters re-run the AI search over the whole catalogue</span>
          </summary>
        <form
          className="pb-1"
          onSubmit={(e) => { e.preventDefault(); apply(); }}
          aria-label="Filter results"
        >
          <div className="mt-3 grid gap-x-8 sm:grid-cols-2 xl:grid-cols-3">
            <CheckboxGroup legend="Category" selected={draft.category ?? []} onChange={setList("category")}
              options={(cats.data ?? []).map((c) => ({ value: c.slug, label: c.name, count: c.productCount }))} />
            {ATTR_FILTERS.map((f) => (
              <CheckboxGroup key={f.key} legend={f.legend} selected={draft[f.key] ?? []} onChange={setList(f.key)} max={10}
                options={(facets.data?.attributes[f.key] ?? []).map((x) => ({ value: x.value, count: x.count }))} />
            ))}
            {hasPrices && (
              <fieldset className="border-t border-stone-deep pt-4">
                <legend className="mb-2 text-sm font-semibold">Price (INR)</legend>
                <div className="flex items-center gap-2">
                  <label className="sr-only" htmlFor="pmin">Minimum price</label>
                  <input id="pmin" type="number" min={0} className="field" placeholder="Min" value={draft.min_price ?? ""} onChange={(e) => setPrice("min_price", e.target.value)} />
                  <span className="text-muted">to</span>
                  <label className="sr-only" htmlFor="pmax">Maximum price</label>
                  <input id="pmax" type="number" min={0} className="field" placeholder="Max" value={draft.max_price ?? ""} onChange={(e) => setPrice("max_price", e.target.value)} />
                </div>
              </fieldset>
            )}
          </div>
          {facets.error && <div className="mt-3"><ErrorNotice message={facets.error} onRetry={facets.reload} /></div>}
          <div className="mt-4 flex flex-wrap gap-3">
            <button type="submit" className="btn-primary" disabled={loading || !changed || !file}>
              {loading && <Loader2 className="size-4 animate-spin" aria-hidden />} Apply filters
            </button>
            {activeCount > 0 && <button type="button" className="btn-ghost" disabled={loading} onClick={clearAll}>Clear filters</button>}
            {!file && <p className="self-center text-xs text-muted">Upload the photo again to change filters.</p>}
          </div>
        </form>
        </details>
        {error && <div className="mt-4"><ErrorNotice message={error} /></div>}
        {result.retrieval?.mode === "exact_fallback" && (
          <p className="mt-3 text-xs text-muted">Your filters are selective, so every matching product was compared exactly.</p>
        )}

        {shown.length === 0 && <div className="mt-8"><EmptyState title="No products match these filters">Remove a filter and apply again.</EmptyState></div>}

        {exact.length > 0 && (
          <div className="mt-8">
            <h2 className="font-sans text-sm font-semibold text-rani">We found this exact product</h2>
            <div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-10 sm:grid-cols-3 xl:grid-cols-4">
              {exact.map((r, i) => <ResultCard key={r.id} r={r} index={i} />)}
            </div>
          </div>
        )}
        {rest.length > 0 && (
          <div className="mt-8">
            {exact.length > 0 && <h2 className="font-sans text-sm font-semibold">Similar pieces</h2>}
            <div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-10 sm:grid-cols-3 xl:grid-cols-4">
              {rest.map((r, i) => <ResultCard key={r.id} r={r} index={i + exact.length} />)}
            </div>
          </div>
        )}
        <p className="mt-10 max-w-2xl text-xs leading-relaxed text-muted">
          The match score ranks products against each other for this photo. It is not the probability that a product is the
          same item. "Exact match" appears only when the photo is nearly identical to one catalogue image.
        </p>
      </div>
    </section>
  );
}

function categoryLabel(slug: string): string {
  const words: Record<string, string> = { "western-dresses": "dresses", "kurtas-kurtis": "kurtas", "tops-shirts": "tops and shirts",
    "lehengas": "lehengas", "sarees": "sarees", "other-ethnic": "ethnic wear", "outerwear": "jackets", "pants": "pants",
    "shorts": "shorts", "skirts": "skirts", "footwear": "shoes", "hats": "hats" };
  return words[slug] ?? slug.replace(/-/g, " ");
}
