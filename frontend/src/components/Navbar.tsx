import { Menu, ScanSearch, X } from "lucide-react";
import { useState } from "react";
import { Link, NavLink } from "react-router";

const links = [
  { to: "/catalogue", label: "Catalogue" },
  { to: "/search", label: "Visual search" },
  { to: "/about", label: "How it works" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);
  const cls = ({ isActive }: { isActive: boolean }) =>
    `text-sm font-medium transition-colors ${isActive ? "text-indigo" : "text-muted hover:text-ink"}`;
  return (
    <header className="sticky top-0 z-30 border-b border-stone-deep/70 bg-white/90 backdrop-blur">
      <nav className="container-page flex h-16 items-center justify-between" aria-label="Main">
        <Link to="/" className="flex items-baseline gap-1.5" onClick={() => setOpen(false)}>
          <span className="font-display text-2xl font-semibold tracking-tight">StyleFind</span>
          <span className="rounded bg-indigo px-1.5 py-0.5 text-[11px] font-semibold text-white">AI</span>
        </Link>
        <div className="hidden items-center gap-8 md:flex">
          {links.map((l) => (
            <NavLink key={l.to} to={l.to} className={cls}>
              {l.label}
            </NavLink>
          ))}
          <Link to="/search" className="btn-primary !py-2">
            <ScanSearch className="size-4" aria-hidden /> Search by photo
          </Link>
        </div>
        <button type="button" className="md:hidden" aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? <X /> : <Menu />}
        </button>
      </nav>
      {open && (
        <div className="container-page flex flex-col gap-4 border-t border-stone-deep pb-5 pt-4 md:hidden">
          {links.map((l) => (
            <NavLink key={l.to} to={l.to} className={cls} onClick={() => setOpen(false)}>
              {l.label}
            </NavLink>
          ))}
        </div>
      )}
    </header>
  );
}
