import type { NextFunction, Request, Response } from "express";
import multer from "multer";
import { ZodError } from "zod";

export class ApiError extends Error {
  constructor(public status: number, message: string, public details?: unknown) {
    super(message);
  }
}

export function notFound(_req: Request, res: Response) {
  res.status(404).json({ error: "Not found" });
}

// Express recognises error handlers by their 4 arguments.
export function errorHandler(err: unknown, req: Request, res: Response, _next: NextFunction) {
  if (err instanceof ApiError) {
    res.status(err.status).json({ error: err.message, ...(err.details ? { details: err.details } : {}) });
    return;
  }
  if (err instanceof ZodError) {
    res.status(400).json({ error: "Invalid request", details: err.issues.map((i) => ({ field: i.path.join("."), message: i.message })) });
    return;
  }
  if (err instanceof multer.MulterError) {
    const msg = err.code === "LIMIT_FILE_SIZE" ? "Image is too large" : "Invalid upload";
    res.status(err.code === "LIMIT_FILE_SIZE" ? 413 : 400).json({ error: msg });
    return;
  }
  // Standard http-errors thrown by Express internals (e.g. static file 403/404, bad JSON 400).
  const status = (err as { status?: number; statusCode?: number })?.status ?? (err as { statusCode?: number })?.statusCode;
  if (typeof status === "number" && status >= 400 && status < 500) {
    res.status(status).json({ error: status === 404 ? "Not found" : status === 403 ? "Forbidden" : "Bad request" });
    return;
  }
  // Unknown error: log the details on the server, return a generic message (no paths / stack traces).
  console.error(`[error] ${req.method} ${req.originalUrl}`, err);
  res.status(500).json({ error: "Internal server error" });
}
