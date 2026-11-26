import { Eye, EyeOff, Pencil, Plus, RefreshCw, Trash2 } from "lucide-react";
import { useState } from "react";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { ErrorNotice, Spinner } from "../components/Feedback";
import { ProductForm } from "../components/ProductForm";
import { titleCase } from "../components/ProductCard";
import { useAsync } from "../hooks/useAsync";
import { api, errorMessage } from "../services/api";
import type { Product } from "../types";

type Mode = { kind: "list" } | { kind: "create" } | { kind: "edit"; product: Product };

export function ProductManagementPage() {
  const [page, setPage] = useState(1);
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<Mode>({ kind: "list" });
  const [notice, setNotice] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<Product | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionErr, setActionErr] = useState<string | null>(null);
  const list = useAsync(() => api.adminProducts({ page, limit: 20, q: q || undefined, sort: "newest" }), [page, q]);
  const cats = useAsync(() => api.categories(), []);

  const wrap = (fn: (form: FormData) => Promise<{ embedding?: { status: string; error?: string } }>, verb: string) => async (form: FormData) => {
    try {
      const r = await fn(form);
      const emb = r.embedding
        ? r.embedding.status === "generated" ? " and indexed for visual search" : `, but indexing is pending (${r.embedding.error})`
        : "";
      setNotice(`Product ${verb}${emb}.`);
      setMode({ kind: "list" });
      list.reload();
    } catch (e) {
      throw new Error(errorMessage(e));
    }
  };

  const remove = async (p: Product) => {
    setActionErr(null);
    setBusyId(p.id);
    try {
      await api.deleteProduct(p.id);
      setNotice(`"${p.name}" deleted.`);
      setConfirm(null);
      list.reload();
    } catch (e) {
      setActionErr(errorMessage(e));
    } finally {
      setBusyId(null);
    }
  };

  const toggleActive = async (p: Product) => {
    setActionErr(null);
    setBusyId(p.id);
    try {
      const form = new FormData();
      form.append("isActive", String(!p.isActive));
      await api.updateProduct(p.id, form);
      setNotice(`"${p.name}" is now ${p.isActive ? "hidden from the shop and search" : "visible"}.`);
      list.reload();
    } catch (e) {
      setActionErr(errorMessage(e));
    } finally {
      setBusyId(null);
    }
  };

  const reindex = async (p: Product) => {
    setActionErr(null);
    setBusyId(p.id);
    try {
      const r = await api.reindex(p.id);
      if (r.embedding.status === "generated") setNotice(`"${p.name}" re-indexed for visual search.`);
      else setActionErr(`Indexing "${p.name}" failed: ${r.embedding.error ?? "unknown error"}`);
      list.reload();
    } catch (e) {
      setActionErr(errorMessage(e));
    } finally {
      setBusyId(null);
    }
  };

  if (mode.kind !== "list") {
    return (
      <div>
        <h1 className="mb-6 text-3xl">{mode.kind === "create" ? "Add product" : "Edit product"}</h1>
        <div className="rounded-xl bg-white p-6">
          {cats.data ? (
            <ProductForm
              initial={mode.kind === "edit" ? mode.product : null}
              categories={cats.data}
              onCancel={() => setMode({ kind: "list" })}
              onSubmit={mode.kind === "create" ? wrap(api.createProduct, "added") : wrap((f) => api.updateProduct(mode.product.id, f), "saved")}
            />
          ) : <Spinner />}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-3xl">Products</h1>
        <button type="button" className="btn-primary" onClick={() => { setNotice(null); setMode({ kind: "create" }); }}>
          <Plus className="size-4" aria-hidden /> Add product
        </button>
      </div>
      <form className="mt-6 flex gap-2" onSubmit={(e) => { e.preventDefault(); setPage(1); setQ(new FormData(e.currentTarget).get("q") as string); }}>
        <label htmlFor="admin-q" className="sr-only">Search products</label>
        <input id="admin-q" name="q" className="field max-w-sm" placeholder="Search by name" defaultValue={q} />
        <button type="submit" className="btn-ghost">Search</button>
      </form>
      {notice && <p role="status" className="mt-4 rounded-lg bg-sage-soft p-3 text-sm text-sage">{notice}</p>}
      {actionErr && <div className="mt-4"><ErrorNotice message={actionErr} /></div>}
      <ConfirmDialog
        open={confirm !== null}
        title="Delete this product?"
        message={confirm ? `"${confirm.name}" and its embeddings will be removed permanently. Images uploaded through the admin panel are deleted; dataset images stay on disk.` : ""}
        confirmLabel="Delete product"
        busy={busyId !== null}
        onConfirm={() => confirm && remove(confirm)}
        onCancel={() => setConfirm(null)}
      />
      {list.loading && <Spinner />}
      {list.error && <div className="mt-4"><ErrorNotice message={list.error} onRetry={list.reload} /></div>}
      {list.data && (
        <div className="mt-4 overflow-x-auto rounded-xl bg-white">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-stone-deep text-muted">
              <tr><th className="p-3 font-medium">Product</th><th className="font-medium">Category</th><th className="font-medium">Colour</th><th className="font-medium">Status</th><th className="p-3" /></tr>
            </thead>
            <tbody className="divide-y divide-stone-deep">
              {list.data.items.map((p) => (
                <tr key={p.id}>
                  <td className="flex items-center gap-3 p-3">
                    {p.imageUrl && <img src={p.imageUrl} alt="" className="h-14 w-11 rounded bg-stone object-contain" />}
                    <span><span className="line-clamp-1 font-medium">{p.name}</span><span className="text-xs text-muted">{p.productCode}</span></span>
                  </td>
                  <td>{p.categoryName}</td>
                  <td>{titleCase(p.colour)}</td>
                  <td className="text-xs">
                    <span className={p.isActive ? "text-sage" : "text-muted"}>{p.isActive ? "Visible" : "Hidden"}</span>
                    {p.hasEmbedding === false && <span className="block text-rani">Not indexed</span>}
                  </td>
                  <td className="whitespace-nowrap p-3 text-right">
                    <span className="inline-flex gap-1">
                      <button type="button" aria-label={`Re-index ${p.name}`} title="Regenerate embedding" disabled={busyId !== null}
                        className="rounded p-2 text-muted hover:bg-stone hover:text-indigo disabled:opacity-40" onClick={() => reindex(p)}>
                        <RefreshCw className={`size-4 ${busyId === p.id ? "animate-spin" : ""}`} />
                      </button>
                      <button type="button" aria-label={p.isActive ? `Hide ${p.name}` : `Show ${p.name}`} title={p.isActive ? "Hide from shop" : "Show in shop"}
                        disabled={busyId !== null} className="rounded p-2 text-muted hover:bg-stone hover:text-indigo disabled:opacity-40" onClick={() => toggleActive(p)}>
                        {p.isActive ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                      </button>
                      <button type="button" aria-label={`Edit ${p.name}`} className="rounded p-2 text-muted hover:bg-stone hover:text-indigo" onClick={() => { setNotice(null); setMode({ kind: "edit", product: p }); }}>
                        <Pencil className="size-4" />
                      </button>
                      <button type="button" aria-label={`Delete ${p.name}`} className="rounded p-2 text-muted hover:bg-stone hover:text-danger" onClick={() => setConfirm(p)}>
                        <Trash2 className="size-4" />
                      </button>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex items-center justify-between p-3 text-sm text-muted">
            <span>{list.data.total} products</span>
            <span className="flex items-center gap-2">
              <button type="button" className="btn-ghost !py-1.5" disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous</button>
              Page {page} of {list.data.pages}
              <button type="button" className="btn-ghost !py-1.5" disabled={page >= list.data.pages} onClick={() => setPage(page + 1)}>Next</button>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
