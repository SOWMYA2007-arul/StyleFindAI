import rateLimit from "express-rate-limit";

const common = { standardHeaders: "draft-8" as const, legacyHeaders: false, message: { error: "Too many requests, please slow down" } };

export const apiLimiter = rateLimit({ ...common, windowMs: 60_000, limit: 300 });
/** AI search is expensive (runs a neural network) - stricter limit. */
export const searchLimiter = rateLimit({ ...common, windowMs: 60_000, limit: 30 });
/** Slows down password guessing. */
export const loginLimiter = rateLimit({ ...common, windowMs: 15 * 60_000, limit: 10 });
