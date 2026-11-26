import { Link } from "react-router";
import type { Product } from "../types";

export const titleCase = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());
export const formatPrice = (p: number | null) =>
  p === null ? null : new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(p);

export function ProductCard({ p }: { p: Pick<Product, "id" | "name" | "imageUrl" | "categoryName" | "colour" | "pattern" | "price"> }) {
  return (
    <Link to={`/products/${p.id}`} className="group block">
      <div className="aspect-[3/4] overflow-hidden rounded-xl bg-stone">
        {p.imageUrl && (
          <img src={p.imageUrl} alt={p.name} loading="lazy" className="size-full object-contain transition-transform duration-500 group-hover:scale-[1.03]" />
        )}
      </div>
      <p className="mt-3 text-xs text-muted">{p.categoryName}</p>
      <p className="mt-0.5 line-clamp-2 text-sm font-medium leading-snug group-hover:text-indigo">{p.name}</p>
      <p className="mt-1 text-xs text-muted">
        {[p.colour, p.pattern].filter((v) => v && v !== "unknown").map(titleCase).join(", ") || " "}
        {p.price !== null && <span className="ml-2 font-semibold text-ink">{formatPrice(p.price)}</span>}
      </p>
    </Link>
  );
}
