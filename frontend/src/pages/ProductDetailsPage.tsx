import { ScanSearch } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { EmptyState, ErrorNotice, Spinner } from "../components/Feedback";
import { formatPrice, titleCase } from "../components/ProductCard";
import { ResultCard } from "../components/ResultCard";
import { useSearch } from "../context/SearchContext";
import { useAsync } from "../hooks/useAsync";
import { api } from "../services/api";

export function ProductDetailsPage() {
  const id = Number(useParams().id);
  const nav = useNavigate();
  const s = useSearch();
  const product = useAsync(() => api.product(id), [id]);
  const similar = useAsync(() => api.similar(id), [id]);
  const [active, setActive] = useState(0);

  // Re-use this product's photo as a new visual search query.
  const searchWithThis = async () => {
    const p = product.data;
    const url = p?.images?.[active]?.url ?? p?.imageUrl;
    if (!url) return;
    const blob = await fetch(url).then((r) => r.blob());
    s.setFile(new File([blob], "product.jpg", { type: blob.type || "image/jpeg" }));
    nav("/search");
  };

  if (product.loading) return <div className="container-page"><Spinner label="Loading product" /></div>;
  if (product.error || !product.data)
    return <div className="container-page py-16"><EmptyState title="Product not found"><Link to="/catalogue" className="text-indigo underline">Back to the catalogue</Link></EmptyState></div>;
  const p = product.data;
  const gallery = p.images?.length ? p.images.map((i) => i.url) : p.imageUrl ? [p.imageUrl] : [];
  const mainImage = gallery[Math.min(active, gallery.length - 1)];
  const rows: [string, string][] = [
    ["Category", p.categoryName], ["Type", p.subcategory ?? "unknown"], ["Colour", p.colour], ["Pattern", p.pattern],
    ["Sleeves", p.sleeveType], ["Neckline", p.neckline],
  ];

  return (
    <>
      <section className="container-page grid gap-10 py-10 md:grid-cols-2">
        <div>
          <div className="overflow-hidden rounded-2xl bg-stone">
            {mainImage && <img src={mainImage} alt={p.name} className="mx-auto max-h-[640px] w-full object-contain" />}
          </div>
          {gallery.length > 1 && (
            <div className="mt-3 flex gap-2" role="group" aria-label="Product images">
              {gallery.map((u, i) => (
                <button key={u} type="button" onClick={() => setActive(i)} aria-label={`Show image ${i + 1}`} aria-pressed={i === active}
                  className={`h-20 w-16 overflow-hidden rounded-lg bg-stone ring-2 ${i === active ? "ring-indigo" : "ring-transparent"}`}>
                  <img src={u} alt="" className="size-full object-contain" />
                </button>
              ))}
            </div>
          )}
        </div>
        <div>
          <Link to={`/catalogue?category=${p.category}`} className="text-sm text-muted hover:text-indigo">{p.categoryName}</Link>
          <h1 className="mt-2 text-3xl leading-tight sm:text-4xl">{p.name}</h1>
          {p.price !== null && <p className="mt-3 text-2xl font-semibold">{formatPrice(p.price)}</p>}
          {p.description && <p className="mt-4 leading-relaxed text-muted">{p.description}</p>}
          <dl className="mt-8 divide-y divide-stone-deep border-y border-stone-deep text-sm">
            {rows.map(([k, v]) => (
              <div key={k} className="flex justify-between py-2.5">
                <dt className="text-muted">{k}</dt>
                <dd className={v === "unknown" ? "text-muted" : "font-medium"}>{v === "unknown" ? "Not listed" : titleCase(v)}</dd>
              </div>
            ))}
          </dl>
          <button type="button" className="btn-primary mt-8" onClick={searchWithThis}>
            <ScanSearch className="size-4" aria-hidden /> Search with this photo
          </button>
          <p className="mt-8 text-xs leading-relaxed text-muted">
            Product code {p.productCode}. Source: {p.sourceDataset}. Licence: {p.licence}
          </p>
        </div>
      </section>
      <section className="container-page py-10" aria-labelledby="sim-h">
        <h2 id="sim-h" className="text-3xl">Visually similar</h2>
        {similar.loading && <Spinner label="Finding similar products" />}
        {similar.error && <div className="mt-4"><ErrorNotice message={similar.error} onRetry={similar.reload} /></div>}
        <div className="mt-6 grid grid-cols-2 gap-x-5 gap-y-10 sm:grid-cols-3 lg:grid-cols-4">
          {similar.data?.slice(0, 8).map((r, i) => <ResultCard key={r.id} r={r} index={i} />)}
        </div>
      </section>
    </>
  );
}
