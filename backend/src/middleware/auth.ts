import crypto from "node:crypto";
import type { CookieOptions, NextFunction, Request, Response } from "express";
import jwt from "jsonwebtoken";
import { query } from "../config/db.js";
import { env } from "../config/env.js";
import { ApiError } from "./errors.js";

/**
 * Admin authentication.
 *
 * Browser (website): the JWT lives in an HttpOnly cookie (`sf_session`) that JavaScript cannot read,
 * so an XSS bug cannot steal it. Because browsers send cookies automatically, every state-changing
 * request must also carry the CSRF token (double-submit: readable cookie `sf_csrf` copied into the
 * `X-CSRF-Token` header). SameSite=Strict blocks cross-site sending as a second layer.
 *
 * Scripts / tests: `Authorization: Bearer <token>` still works (login with ?mode=bearer). Bearer
 * requests are not sent automatically by browsers, so they need no CSRF token.
 *
 * Revocation: tokens carry the account's token_version; logout increments it, invalidating every
 * issued token for that admin immediately.
 */
export const SESSION_COOKIE = "sf_session";
export const CSRF_COOKIE = "sf_csrf";

export interface AdminClaims {
  sub: string;
  email: string;
  role: "admin";
  tv: number;
}

function parseDurationMs(v: string): number {
  const m = /^(\d+)\s*([smhd]?)$/.exec(v.trim());
  if (!m) return 8 * 3600_000;
  return Number(m[1]) * ({ s: 1000, m: 60_000, h: 3600_000, d: 86_400_000, "": 1000 } as Record<string, number>)[m[2]];
}

const secure = () => env.NODE_ENV === "production" || process.env.COOKIE_SECURE === "true";

export function signAdminToken(id: number, email: string, tokenVersion = 0): string {
  return jwt.sign({ email, role: "admin", tv: tokenVersion }, env.JWT_SECRET, {
    subject: String(id),
    expiresIn: env.JWT_EXPIRES_IN as jwt.SignOptions["expiresIn"],
    algorithm: "HS256",
  });
}

export function setSessionCookies(res: Response, token: string): string {
  const maxAge = parseDurationMs(env.JWT_EXPIRES_IN);
  const base: CookieOptions = { sameSite: "strict", secure: secure(), maxAge };
  const csrf = crypto.randomBytes(24).toString("base64url");
  res.cookie(SESSION_COOKIE, token, { ...base, httpOnly: true, path: "/api" });
  res.cookie(CSRF_COOKIE, csrf, { ...base, httpOnly: false, path: "/" });
  return csrf;
}

export function clearSessionCookies(res: Response) {
  res.clearCookie(SESSION_COOKIE, { path: "/api", sameSite: "strict", secure: secure() });
  res.clearCookie(CSRF_COOKIE, { path: "/", sameSite: "strict", secure: secure() });
}

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

function safeEqual(a: string, b: string) {
  const x = Buffer.from(a), y = Buffer.from(b);
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

/** Protects admin-only routes. */
export async function requireAdmin(req: Request, res: Response, next: NextFunction) {
  const header = req.headers.authorization ?? "";
  const bearer = header.startsWith("Bearer ") ? header.slice(7) : null;
  const cookieToken: string | undefined = req.cookies?.[SESSION_COOKIE];
  const token = bearer ?? cookieToken;
  if (!token) return next(new ApiError(401, "Admin login required"));
  let claims: AdminClaims;
  try {
    claims = jwt.verify(token, env.JWT_SECRET, { algorithms: ["HS256"] }) as AdminClaims;
    if (claims.role !== "admin") throw new Error("not admin");
  } catch {
    return next(new ApiError(401, "Session expired or invalid - please log in again"));
  }
  if (!bearer && !SAFE_METHODS.has(req.method)) {
    const sent = String(req.headers["x-csrf-token"] ?? "");
    const expected = String(req.cookies?.[CSRF_COOKIE] ?? "");
    if (!sent || !expected || !safeEqual(sent, expected)) return next(new ApiError(403, "Security check failed (CSRF). Reload the page and try again."));
  }
  try {
    const r = await query<{ token_version: number }>("SELECT token_version FROM admin_users WHERE id = $1", [Number(claims.sub)]);
    if (!r.rowCount || r.rows[0].token_version !== (claims.tv ?? 0))
      return next(new ApiError(401, "Session has been signed out - please log in again"));
  } catch (e) {
    return next(e);
  }
  res.locals.admin = claims;
  next();
}
