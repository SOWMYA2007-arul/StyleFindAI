import { Loader2, ScanSearch } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { ErrorNotice, Spinner } from "../components/Feedback";
import { ImageDropzone } from "../components/ImageDropzone";
import { ProductCard } from "../components/ProductCard";
import { useSearch } from "../context/SearchContext";
import { useAsync } from "../hooks/useAsync";
import { api } from "../services/api";

const STEPS = [
  { title: "Read your photo", body: "The image is checked, converted to RGB and resized the way the model expects." },
  { title: "Turn it into numbers", body: "A fashion-trained vision model (FashionCLIP) turns the photo into a 512-number embedding that captures cut, drape and print." },
  { title: "Compare with every product", body: "PostgreSQL with pgvector finds the 50 catalogue embeddings closest to yours by cosine distance." },
  { title: "Re-rank like a stylist", body: "Colour, pattern and garment details re-order those 50, and each result shows which of these it matched." },
];

export function HomePage() {
  const s = useSearch();
  const nav = useNavigate();
  const [localErr, setLocalErr] = useState<string | null>(null);
  const cats = useAsync(() => api.categories(), []);
  const fresh = useAsync(() => api.products({ sort: "newest", limit: 8 }), []);

  const go = async () => {
    if (await s.runSearch()) nav("/search/results");
  };

  return (
    <>
      <section className="container-page grid items-center gap-10 py-12 lg:grid-cols-[1.05fr_1fr] lg:py-20">
        <div>
          <h1 className="text-5xl leading-[1.05] sm:text-6xl lg:text-7xl">Seen it somewhere? Find it here.</h1>
          <p className="mt-6 max-w-lg text-lg leading-relaxed text-muted">
            Upload a photo of a kurta, saree, lehenga or dress. StyleFind compares it with every piece in the catalogue and
            shows the same item if we stock it, then the closest alternatives in colour, print and silhouette.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <button type="button" className="btn-primary" disabled={!s.file || s.loading} onClick={go}>
              {s.loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <ScanSearch className="size-4" aria-hidden />}
              {s.loading ? "Searching the catalogue" : "Find similar pieces"}
            </button>
            <Link to="/catalogue" className="btn-ghost">Browse the catalogue</Link>
          </div>
          <div className="mt-4 max-w-lg">
            {(localErr || s.error) && <ErrorNotice message={(localErr || s.error)!} />}
          </div>
        </div>
        <ImageDropzone previewUrl={s.previewUrl} onFile={s.setFile} onError={setLocalErr} tall />
      </section>

      <section className="container-page py-10" aria-labelledby="cat-h">
        <div className="flex items-end justify-between">
          <h2 id="cat-h" className="text-3xl sm:text-4xl">Shop by category</h2>
          <Link to="/catalogue" className="text-sm font-semibold text-indigo hover:underline">View all products</Link>
        </div>
        {cats.loading && <Spinner />}
        {cats.error && <ErrorNotice message={cats.error} onRetry={cats.reload} />}
        <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          {cats.data?.map((c) => (
            <Link key={c.slug} to={`/catalogue?category=${c.slug}`} className="group">
              <div className="aspect-[4/5] overflow-hidden rounded-xl bg-stone">
                {c.coverImageUrl && <img src={c.coverImageUrl} alt="" className="size-full object-contain transition-transform duration-500 group-hover:scale-105" />}
              </div>
              <p className="mt-2 text-sm font-semibold group-hover:text-indigo">{c.name}</p>
              <p className="text-xs text-muted">{c.productCount} pieces</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="mt-10 bg-stone py-16" aria-labelledby="how-h">
        <div className="container-page grid gap-10 lg:grid-cols-[1fr_2fr]">
          <div>
            <h2 id="how-h" className="text-3xl sm:text-4xl">How the search works</h2>
            <p className="mt-4 text-muted">Every search runs a real model on your photo. Nothing on the results page is pre-picked.</p>
            <Link to="/about" className="mt-4 inline-block text-sm font-semibold text-indigo hover:underline">Read the technical details</Link>
          </div>
          <ol className="grid gap-8 sm:grid-cols-2">
            {STEPS.map((st, i) => (
              <li key={st.title} className="flex gap-4">
                <span className="font-display text-4xl leading-none text-indigo/40">{i + 1}</span>
                <div>
                  <h3 className="font-sans text-base font-semibold">{st.title}</h3>
                  <p className="mt-1 text-sm leading-relaxed text-muted">{st.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="container-page py-16" aria-labelledby="new-h">
        <h2 id="new-h" className="text-3xl sm:text-4xl">Recently added</h2>
        {fresh.loading && <Spinner />}
        {fresh.error && <ErrorNotice message={fresh.error} onRetry={fresh.reload} />}
        <div className="mt-6 grid grid-cols-2 gap-x-5 gap-y-8 sm:grid-cols-3 lg:grid-cols-4">
          {fresh.data?.items.map((p) => <ProductCard key={p.id} p={p} />)}
        </div>
      </section>
    </>
  );
}
