import { Loader2, RefreshCw } from "lucide-react";
import { useState } from "react";
import { ErrorNotice, Spinner } from "../components/Feedback";
import { useAsync } from "../hooks/useAsync";
import { api, errorMessage } from "../services/api";

export function AdminDashboardPage() {
  const stats = useAsync(() => api.stats(), []);
  const models = useAsync(() => api.models(), []);
  const failures = useAsync(() => api.failures(), []);
  const [retrying, setRetrying] = useState<number | null>(null);
  const retry = async (id: number) => {
    setRetrying(id);
    try {
      const r = await api.reindex(id);
      setJob({ busy: false, msg: r.embedding.status === "generated" ? `Product ${id} indexed.` : null,
               err: r.embedding.status === "generated" ? null : `Product ${id}: ${r.embedding.error ?? "indexing failed"}` });
    } catch (e) {
      setJob({ busy: false, msg: null, err: errorMessage(e) });
    } finally {
      setRetrying(null);
      failures.reload(); stats.reload();
    }
  };
  const [job, setJob] = useState<{ busy: boolean; msg: string | null; err: string | null }>({ busy: false, msg: null, err: null });

  const run = async (opts: { model?: string; force?: boolean }) => {
    setJob({ busy: true, msg: null, err: null });
    try {
      const r = await api.generateEmbeddings(opts);
      setJob({ busy: false, msg: `${r.model_key}: embedded ${r.indexed} images in ${r.seconds}s.`, err: null });
      stats.reload();
      models.reload();
      failures.reload();
    } catch (e) {
      setJob({ busy: false, msg: null, err: errorMessage(e) });
    }
  };

  if (stats.loading) return <Spinner label="Loading statistics" />;
  if (stats.error || !stats.data) return <ErrorNotice message={stats.error ?? "No data"} onRetry={stats.reload} />;
  const s = stats.data;
  const tiles: [string, string | number][] = [
    ["Active products", s.activeProducts], ["Product images", s.images], ["Searches, last 7 days", s.searchesLast7Days],
    ["Average search time", s.avgSearchLatencyMs === null ? "No searches yet" : `${s.avgSearchLatencyMs} ms`],
  ];
  const maxCat = Math.max(1, ...s.byCategory.map((c) => c.count));

  return (
    <div className="space-y-8">
      <h1 className="text-3xl">Dashboard</h1>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {tiles.map(([k, v]) => (
          <div key={k} className="rounded-xl bg-white p-5">
            <p className="text-sm text-muted">{k}</p>
            <p className="mt-2 font-display text-3xl">{v}</p>
          </div>
        ))}
      </div>

      <div className="rounded-xl bg-white p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="font-sans text-lg font-semibold">Search index</h2>
            <p className="mt-1 text-sm text-muted">
              Search uses <strong>{s.primaryModel}</strong>.{" "}
              {s.missingPrimaryEmbeddings === 0 ? "Every active product is indexed." : `${s.missingPrimaryEmbeddings} products are not searchable yet.`}
              {(s.failedIndexing ?? 0) > 0 && <span className="ml-1 font-medium text-rani">{s.failedIndexing} failed.</span>}
            </p>
          </div>
          <button type="button" className="btn-primary" disabled={job.busy || s.missingPrimaryEmbeddings === 0} onClick={() => run({})}>
            {job.busy ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <RefreshCw className="size-4" aria-hidden />}
            Index missing products
          </button>
        </div>
        {job.msg && <p role="status" className="mt-4 rounded-lg bg-sage-soft p-3 text-sm text-sage">{job.msg}</p>}
        {job.err && <div className="mt-4"><ErrorNotice message={job.err} /></div>}
        <div className="mt-6 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-muted">
              <tr><th className="py-2 font-medium">Model</th><th className="font-medium">Dimensions</th><th className="font-medium">Text search</th><th className="font-medium">Indexed images</th><th /></tr>
            </thead>
            <tbody className="divide-y divide-stone-deep">
              {models.data?.map((m) => (
                <tr key={m.key}>
                  <td className="py-2.5"><span className="font-medium">{m.key}</span><span className="block text-xs text-muted">{m.name}</span></td>
                  <td>{m.dims}</td>
                  <td>{m.supports_text ? "Yes" : "No"}</td>
                  <td>{m.catalogue_embeddings} / {s.images}</td>
                  <td className="text-right">
                    <button type="button" className="text-xs font-semibold text-indigo hover:underline disabled:opacity-40" disabled={job.busy}
                      onClick={() => run({ model: m.key, force: m.catalogue_embeddings > 0 })}>
                      {m.catalogue_embeddings > 0 ? "Regenerate" : "Generate"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {models.error && <ErrorNotice message={models.error} onRetry={models.reload} />}
          <p className="mt-3 text-xs text-muted">Models other than {s.primaryModel} download their weights on first use (about 350 MB to 1.2 GB) and can take several minutes for a full catalogue on a laptop CPU.</p>
        </div>
      </div>

      {failures.data && failures.data.length > 0 && (
        <div className="rounded-xl bg-white p-6">
          <h2 className="font-sans text-lg font-semibold">Products that are not searchable</h2>
          <p className="mt-1 text-sm text-muted">These products have no embedding for the search model yet, usually because the AI service was offline or the image could not be read.</p>
          <ul className="mt-4 divide-y divide-stone-deep text-sm">
            {failures.data.map((f) => (
              <li key={f.id} className="flex flex-wrap items-center justify-between gap-3 py-2.5">
                <span className="min-w-0">
                  <span className="font-medium">{f.name}</span> <span className="text-xs text-muted">{f.productCode}</span>
                  <span className={`block text-xs ${f.status === "failed" ? "text-rani" : "text-muted"}`}>
                    {f.status === "failed" ? `Failed: ${f.error ?? "unknown error"}` : "Waiting to be indexed"}
                  </span>
                </span>
                <button type="button" className="btn-ghost !py-1.5 text-xs" disabled={retrying !== null} onClick={() => retry(f.id)}>
                  {retrying === f.id ? <Loader2 className="size-3.5 animate-spin" aria-hidden /> : <RefreshCw className="size-3.5" aria-hidden />}
                  Retry indexing
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl bg-white p-6">
          <h2 className="font-sans text-lg font-semibold">Products by category</h2>
          <ul className="mt-4 space-y-3">
            {s.byCategory.map((c) => (
              <li key={c.slug} className="text-sm">
                <div className="flex justify-between"><span>{c.name}</span><span className="text-muted">{c.count}</span></div>
                <div className="mt-1 h-2 rounded-full bg-stone"><div className="h-2 rounded-full bg-indigo" style={{ width: `${(c.count / maxCat) * 100}%` }} /></div>
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-xl bg-white p-6">
          <h2 className="font-sans text-lg font-semibold">Data sources</h2>
          <ul className="mt-4 divide-y divide-stone-deep text-sm">
            {s.sources.map((x) => <li key={x.source} className="flex justify-between gap-4 py-2"><span className="break-all">{x.source}</span><span className="text-muted">{x.count}</span></li>)}
          </ul>
          <p className="mt-4 text-sm text-muted">Total searches logged: {s.searches}. Only an image hash and the predicted attributes are stored, never the photo.</p>
        </div>
      </div>
    </div>
  );
}
