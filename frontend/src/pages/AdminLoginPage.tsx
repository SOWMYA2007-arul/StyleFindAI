import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router";
import { ErrorNotice } from "../components/Feedback";
import { useAuth } from "../context/AuthContext";
import { errorMessage } from "../services/api";

export function AdminLoginPage() {
  const { login, email, ensureChecked } = useAuth();
  useEffect(ensureChecked, [ensureChecked]);
  const nav = useNavigate();
  const from = (useLocation().state as { from?: string } | null)?.from ?? "/admin";
  const [form, setForm] = useState({ email: "", password: "" });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (email) return <Navigate to={from} replace />;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await login(form.email, form.password);
      nav(from, { replace: true });
    } catch (e2) {
      setErr(errorMessage(e2));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-stone px-5">
      <form onSubmit={submit} className="w-full max-w-sm rounded-2xl bg-white p-8 shadow-sm">
        <Link to="/" className="font-display text-2xl font-semibold">StyleFind</Link>
        <h1 className="mt-6 font-sans text-xl font-semibold">Admin sign in</h1>
        <div className="mt-6 space-y-4">
          <div>
            <label htmlFor="email" className="label">Email</label>
            <input id="email" type="email" autoComplete="username" required className="field" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
          </div>
          <div>
            <label htmlFor="pw" className="label">Password</label>
            <input id="pw" type="password" autoComplete="current-password" required className="field" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
          </div>
          {err && <ErrorNotice message={err} />}
          <button type="submit" className="btn-primary w-full" disabled={busy}>
            {busy && <Loader2 className="size-4 animate-spin" aria-hidden />} Sign in
          </button>
        </div>
        <p className="mt-6 text-xs text-muted">Create an account with <code>npm run create-admin</code> in the backend folder.</p>
      </form>
    </div>
  );
}
