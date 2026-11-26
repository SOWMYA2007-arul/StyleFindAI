import { Loader2 } from "lucide-react";
import { useState } from "react";
import type { Category, Product } from "../types";
import { ErrorNotice } from "./Feedback";
import { ImageDropzone } from "./ImageDropzone";

interface Props {
  initial?: Product | null;
  categories: Category[];
  onSubmit: (form: FormData) => Promise<void>;
  onCancel: () => void;
}

const clean = (v: string | undefined | null) => (v && v !== "unknown" ? v : "");

export function ProductForm({ initial, categories, onSubmit, onCancel }: Props) {
  const [f, setF] = useState({
    productCode: initial?.productCode ?? "",
    name: initial?.name ?? "",
    category: initial?.category ?? categories[0]?.slug ?? "",
    subcategory: initial?.subcategory ?? "",
    price: initial?.price?.toString() ?? "",
    description: initial?.description ?? "",
    colour: clean(initial?.colour),
    pattern: clean(initial?.pattern),
    sleeveType: clean(initial?.sleeveType),
    neckline: clean(initial?.neckline),
    isActive: initial?.isActive ?? true,
  });
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(initial?.imageUrl ?? null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (k: keyof typeof f) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) =>
    setF({ ...f, [k]: e.target.value });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!initial && !file) return setErr("Add a product photo.");
    const form = new FormData();
    for (const [k, v] of Object.entries(f)) {
      if (k === "productCode" && (initial || !v)) continue;
      form.append(k, String(v));
    }
    if (file) form.append("image", file);
    setBusy(true);
    setErr(null);
    try {
      await onSubmit(form);
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : String(e2));
    } finally {
      setBusy(false);
    }
  };

  const input = (k: keyof typeof f, label: string, props: React.InputHTMLAttributes<HTMLInputElement> = {}) => (
    <div>
      <label htmlFor={`pf-${k}`} className="label">{label}</label>
      <input id={`pf-${k}`} className="field" value={String(f[k])} onChange={set(k)} {...props} />
    </div>
  );

  return (
    <form onSubmit={submit} className="grid gap-6 md:grid-cols-[260px_1fr]">
      <div>
        <ImageDropzone previewUrl={preview} onError={setErr} onFile={(fl) => { setFile(fl); setPreview(fl ? URL.createObjectURL(fl) : initial?.imageUrl ?? null); }} />
        {initial && <p className="mt-2 text-xs text-muted">Uploading a new photo replaces the old one and re-indexes the product.</p>}
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {!initial && input("productCode", "Product code (optional)", { placeholder: "generated if empty", pattern: "[A-Za-z0-9_-]{1,40}" })}
        <div className={initial ? "sm:col-span-2" : ""}>{input("name", "Product name", { required: true, minLength: 2, maxLength: 200 })}</div>
        <div>
          <label htmlFor="pf-category" className="label">Category</label>
          <select id="pf-category" className="field" value={f.category} onChange={set("category")}>
            {categories.map((c) => <option key={c.slug} value={c.slug}>{c.name}</option>)}
          </select>
        </div>
        {input("subcategory", "Type", { placeholder: "e.g. Kurtas" })}
        {input("price", "Price (INR)", { type: "number", min: 0, step: "0.01" })}
        {input("colour", "Colour", { placeholder: "e.g. navy blue" })}
        {input("pattern", "Pattern", { placeholder: "e.g. floral" })}
        {input("sleeveType", "Sleeves", { placeholder: "e.g. full sleeves" })}
        {input("neckline", "Neckline", { placeholder: "e.g. v-neck" })}
        <div className="sm:col-span-2">
          <label htmlFor="pf-desc" className="label">Description</label>
          <textarea id="pf-desc" rows={3} className="field" value={f.description} onChange={set("description")} maxLength={2000} />
        </div>
        <label className="flex items-center gap-2 text-sm sm:col-span-2">
          <input type="checkbox" className="size-4 accent-indigo" checked={f.isActive} onChange={(e) => setF({ ...f, isActive: e.target.checked })} />
          Visible in the shop and in search
        </label>
        {err && <div className="sm:col-span-2"><ErrorNotice message={err} /></div>}
        <div className="flex gap-3 sm:col-span-2">
          <button type="submit" className="btn-primary" disabled={busy}>
            {busy && <Loader2 className="size-4 animate-spin" aria-hidden />} {initial ? "Save changes" : "Add product"}
          </button>
          <button type="button" className="btn-ghost" onClick={onCancel} disabled={busy}>Cancel</button>
        </div>
      </div>
    </form>
  );
}
