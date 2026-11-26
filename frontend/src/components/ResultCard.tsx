import { useState } from "react";
import { Link } from "react-router";
import type { SearchResult } from "../types";
import { formatPrice, titleCase } from "./ProductCard";

const MATCH_STYLE = {
  exact: { label: "Exact match", cls: "bg-rani text-white" },
  similar: { label: "Visually similar", cls: "bg-sage-soft text-sage" },
  related: { label: "Related style", cls: "bg-stone text-muted" },
} as const;

const BARS: { key: keyof SearchResult["scores"]; label: string }[] = [
  { key: "visual", label: "Look" },
  { key: "colour", label: "Colour" },
  { key: "pattern", label: "Pattern" },
  { key: "attribute", label: "Details" },
];

/** A search result with its score broken into the parts the ranking actually used. */
export function ResultCard({ r, index }: { r: SearchResult; index: number }) {
  const [open, setOpen] = useState(false);
  const m = MATCH_STYLE[r.matchType];
  const score = Math.round((r.scores.final ?? 0) * 100);
  return (
    <article className="rise group" style={{ animationDelay: `${Math.min(index, 12) * 35}ms` }}>
      <Link to={`/products/${r.id}`} className="relative block aspect-[3/4] overflow-hidden rounded-xl bg-stone">
        <img src={r.imageUrl} alt={r.name} loading="lazy" className="size-full object-contain transition-transform duration-500 group-hover:scale-[1.03]" />
        <span className={`absolute left-2.5 top-2.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${m.cls}`}>{m.label}</span>
      </Link>
      <div className="mt-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-muted">{r.categoryName}</p>
          <Link to={`/products/${r.id}`} className="line-clamp-2 text-sm font-medium leading-snug hover:text-indigo">
            {r.name}
          </Link>
          <p className="mt-0.5 text-xs text-muted">{[r.colour, r.pattern].filter((v) => v !== "unknown").map(titleCase).join(", ")}</p>
          {r.price !== null && <p className="mt-0.5 text-sm font-semibold">{formatPrice(r.price)}</p>}
        </div>
        <div className="text-right" title="Ranking score: how well this product matches your photo compared with the other results. It is not a probability.">
          <p className="font-display text-2xl leading-none">{score}</p>
          <p className="text-[11px] text-muted">match score</p>
        </div>
      </div>
      <dl className="mt-3 grid grid-cols-4 gap-2">
        {BARS.map((b) => {
          const v = r.scores[b.key];
          return (
            <div key={b.key}>
              <dt className="text-[11px] text-muted">{b.label}</dt>
              <dd className="mt-1 h-1.5 overflow-hidden rounded-full bg-stone-deep" aria-label={`${b.label} ${v === null ? "not available" : Math.round(v * 100)}`}>
                {v !== null && <div className="h-full rounded-full bg-indigo" style={{ width: `${Math.round(v * 100)}%` }} />}
              </dd>
            </div>
          );
        })}
      </dl>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="mt-2 text-xs font-semibold text-indigo hover:underline">
        {open ? "Hide reasons" : "Why this result?"}
      </button>
      {open && (
        <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-xs text-muted">
          {r.reasons.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      )}
    </article>
  );
}
