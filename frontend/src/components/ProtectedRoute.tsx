import { useEffect, type ReactNode } from "react";
import { Navigate, useLocation } from "react-router";
import { useAuth } from "../context/AuthContext";
import { Spinner } from "./Feedback";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { email, checking, ensureChecked } = useAuth();
  const loc = useLocation();
  useEffect(ensureChecked, [ensureChecked]);
  if (checking) return <Spinner label="Checking your session" />;
  if (!email) return <Navigate to="/admin/login" replace state={{ from: loc.pathname }} />;
  return <>{children}</>;
}
