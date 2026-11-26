import type { Request, Response } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { env } from "../config/env.js";
import { query } from "../config/db.js";
import { ApiError } from "../middleware/errors.js";
import { clearSessionCookies, CSRF_COOKIE, setSessionCookies, signAdminToken } from "../middleware/auth.js";
import { aiIndex, aiModels } from "../services/aiClient.js";
import { catalogueStats, createProduct, deleteProduct, indexingFailures, reindexProduct, updateProduct } from "../services/adminService.js";
import { listProducts } from "../services/productService.js";
import { embeddingJobSchema, idParam, listQuerySchema, loginSchema, productInputSchema, productUpdateSchema } from "./validators.js";

// Constant-time-ish: always run bcrypt even for unknown emails so timing doesn't reveal valid accounts.
const DUMMY_HASH = bcrypt.hashSync("not-a-real-password", 10);

export async function login(req: Request, res: Response) {
  const { email, password } = loginSchema.parse(req.body);
  const r = await query<{ id: number; password_hash: string; token_version: number }>(
    "SELECT id, password_hash, token_version FROM admin_users WHERE email = $1", [email]);
  const ok = await bcrypt.compare(password, r.rows[0]?.password_hash ?? DUMMY_HASH);
  if (!r.rowCount || !ok) throw new ApiError(401, "Invalid email or password");
  const { id, token_version } = r.rows[0];
  await query("UPDATE admin_users SET last_login_at = now() WHERE id = $1", [id]);
  const token = signAdminToken(id, email, token_version);
  if (req.query.mode === "bearer") {
    // for scripts / API clients that cannot keep cookies; the website never uses this
    res.json({ token, email });
    return;
  }
  const csrfToken = setSessionCookies(res, token);
  res.json({ email, csrfToken });
}

export async function logout(_req: Request, res: Response) {
  // revoke every token issued to this admin (cookie and bearer)
  await query("UPDATE admin_users SET token_version = token_version + 1 WHERE id = $1", [Number(res.locals.admin.sub)]);
  clearSessionCookies(res);
  res.json({ ok: true });
}

/** Public: tells the admin UI whether a valid session cookie exists (always 200, never an error). */
export async function session(req: Request, res: Response) {
  const token: string | undefined = req.cookies?.sf_session;
  if (!token) {
    res.json({ authenticated: false });
    return;
  }
  try {
    const claims = jwt.verify(token, env.JWT_SECRET, { algorithms: ["HS256"] }) as { sub: string; email: string; tv?: number };
    const r = await query<{ token_version: number }>("SELECT token_version FROM admin_users WHERE id = $1", [Number(claims.sub)]);
    const ok = r.rowCount && r.rows[0].token_version === (claims.tv ?? 0);
    res.json(ok ? { authenticated: true, email: claims.email } : { authenticated: false });
  } catch {
    res.json({ authenticated: false });
  }
}

export async function me(req: Request, res: Response) {
  res.json({ email: res.locals.admin.email, csrfToken: req.cookies?.[CSRF_COOKIE] ?? null });
}

export async function stats(_req: Request, res: Response) {
  res.json(await catalogueStats());
}

export async function adminProducts(req: Request, res: Response) {
  const q = listQuerySchema.parse(req.query);
  res.json(await listProducts({
    category: q.category, q: q.q, sort: q.sort === "relevance" ? "newest" : q.sort, page: q.page, limit: q.limit,
    attrs: { colour: q.colour, pattern: q.pattern, sleeve_type: q.sleeve_type, neckline: q.neckline }, includeInactive: true,
  }));
}

export async function create(req: Request, res: Response) {
  const input = productInputSchema.parse(req.body);
  if (!req.file) throw new ApiError(400, "A product image is required");
  res.status(201).json(await createProduct({ ...input, price: input.price ?? null }, req.file.buffer));
}

export async function update(req: Request, res: Response) {
  const { id } = idParam.parse(req.params);
  const input = productUpdateSchema.parse(req.body ?? {});
  res.json(await updateProduct(id, input, req.file?.buffer));
}

export async function remove(req: Request, res: Response) {
  const { id } = idParam.parse(req.params);
  res.json(await deleteProduct(id));
}

export async function generateEmbeddings(req: Request, res: Response) {
  const job = embeddingJobSchema.parse(req.body ?? {});
  res.json(await aiIndex(job.productIds, { force: job.force, model: job.model }));
}

export async function reindex(req: Request, res: Response) {
  const { id } = idParam.parse(req.params);
  res.json(await reindexProduct(id));
}

export async function failures(_req: Request, res: Response) {
  res.json(await indexingFailures());
}

export async function models(_req: Request, res: Response) {
  res.json(await aiModels());
}
