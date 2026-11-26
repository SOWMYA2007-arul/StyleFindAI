import { Loader2, ScanSearch } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router";
import { ErrorNotice } from "../components/Feedback";
import { ImageDropzone } from "../components/ImageDropzone";
import { useSearch } from "../context/SearchContext";

const EXAMPLES = ["in blue with full sleeves", "same print but sleeveless", "in maroon"];

export function VisualSearchPage() {
  const s = useSearch();
  const nav = useNavigate();
  const [localErr, setLocalErr] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (await s.runSearch()) nav("/search/results");
  };

  return (
    <section className="container-page py-12">
      <h1 className="text-4xl sm:text-5xl">Search by photo</h1>
      <p className="mt-3 max-w-2xl text-muted">
        Clear photos of a single garment work best. You can also describe a change, such as a different colour or sleeve
        length, and the search will look for that variation.
      </p>
      <form onSubmit={submit} className="mt-8 grid gap-8 lg:grid-cols-[1.2fr_1fr]">
        <ImageDropzone previewUrl={s.previewUrl} onFile={s.setFile} onError={setLocalErr} tall />
        <div className="flex flex-col gap-6">
          <div>
            <label htmlFor="mod" className="label">Describe a change (optional)</label>
            <textarea
              id="mod"
              rows={3}
              maxLength={300}
              className="field resize-none"
              placeholder="e.g. find a similar design in blue with full sleeves"
              value={s.text}
              onChange={(e) => s.setText(e.target.value)}
            />
            <div className="mt-2 flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button key={ex} type="button" onClick={() => s.setText(ex)} className="rounded-full bg-stone px-3 py-1 text-xs text-muted hover:bg-stone-deep hover:text-ink">
                  {ex}
                </button>
              ))}
            </div>
          </div>
          <label className="flex items-start gap-3 text-sm">
            <input type="checkbox" checked={s.crop} onChange={(e) => s.setCrop(e.target.checked)} className="mt-0.5 size-4 accent-indigo" />
            <span>
              <span className="font-medium">Focus on the garment</span>
              <span className="block text-muted">Tries to crop out the background and other objects first. If the detector isn't installed or finds nothing, the whole photo is used.</span>
            </span>
          </label>
          {(localErr || s.error) && <ErrorNotice message={(localErr || s.error)!} />}
          <button type="submit" className="btn-primary self-start" disabled={!s.file || s.loading}>
            {s.loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <ScanSearch className="size-4" aria-hidden />}
            {s.loading ? "Searching the catalogue" : "Find similar dresses"}
          </button>
          {s.loading && <p role="status" className="text-sm text-muted">Running the model on your photo. The first search after starting the server also loads the model, which can take a little longer.</p>}
        </div>
      </form>
    </section>
  );
}
