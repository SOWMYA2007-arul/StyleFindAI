import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "../services/api";

interface AuthState {
  email: string | null;
  checking: boolean;
  /** Ask the server once whether the session cookie is valid (called by admin pages only). */
  ensureChecked: () => void;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const Ctx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [email, setEmail] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);
  const [started, setStarted] = useState(false);

  // The session lives in an HttpOnly cookie, so only the server can tell whether we are signed in.
  // Shoppers never trigger this check; admin pages call ensureChecked().
  const ensureChecked = useCallback(() => setStarted(true), []);
  useEffect(() => {
    if (!started) return;
    api
      .session()
      .then((r) => setEmail(r.authenticated ? r.email ?? null : null))
      .catch(() => setEmail(null))
      .finally(() => setChecking(false));
  }, [started]);

  const login = useCallback(async (e: string, p: string) => {
    const r = await api.login(e, p);
    setEmail(r.email);
    setChecking(false);
    setStarted(true);
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout(); // revokes the session on the server, clears cookies
    } finally {
      setEmail(null);
    }
  }, []);

  return <Ctx.Provider value={{ email, checking, ensureChecked, login, logout }}>{children}</Ctx.Provider>;
}

export function useAuth() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useAuth must be used inside AuthProvider");
  return c;
}
