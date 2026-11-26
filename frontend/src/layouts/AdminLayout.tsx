import { BarChart3, LogOut, Package, Store } from "lucide-react";
import { Link, NavLink, Outlet, useNavigate } from "react-router";
import { useAuth } from "../context/AuthContext";

export function AdminLayout() {
  const { email, logout } = useAuth();
  const nav = useNavigate();
  const cls = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium ${isActive ? "bg-white text-indigo shadow-sm" : "text-muted hover:text-ink"}`;
  return (
    <div className="min-h-screen bg-stone">
      <header className="border-b border-stone-deep bg-white">
        <div className="container-page flex h-14 items-center justify-between">
          <Link to="/admin" className="font-display text-xl font-semibold">StyleFind <span className="font-sans text-sm font-medium text-muted">admin</span></Link>
          <div className="flex items-center gap-4 text-sm">
            <span className="hidden text-muted sm:inline">{email}</span>
            <Link to="/" className="inline-flex items-center gap-1 text-muted hover:text-ink"><Store className="size-4" aria-hidden /> Shop</Link>
            <button type="button" onClick={async () => { await logout(); nav("/admin/login"); }} className="inline-flex items-center gap-1 text-muted hover:text-danger">
              <LogOut className="size-4" aria-hidden /> Log out
            </button>
          </div>
        </div>
      </header>
      <div className="container-page grid gap-8 py-8 lg:grid-cols-[200px_1fr]">
        <nav className="flex gap-2 lg:flex-col" aria-label="Admin">
          <NavLink to="/admin" end className={cls}><BarChart3 className="size-4" aria-hidden /> Dashboard</NavLink>
          <NavLink to="/admin/products" className={cls}><Package className="size-4" aria-hidden /> Products</NavLink>
        </nav>
        <div className="min-w-0"><Outlet /></div>
      </div>
    </div>
  );
}
