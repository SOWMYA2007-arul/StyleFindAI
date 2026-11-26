import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import { z } from "zod";

// backend/src/config -> project root is three levels up (works for src/ and dist/)
const here = path.dirname(fileURLToPath(import.meta.url));
export const PROJECT_ROOT = path.resolve(here, "..", "..", "..");
dotenv.config({ path: path.join(PROJECT_ROOT, ".env"), quiet: true });

const schema = z.object({
  NODE_ENV: z.string().default("development"),
  BACKEND_PORT: z.coerce.number().int().default(4000),
  POSTGRES_HOST: z.string().default("localhost"),
  POSTGRES_PORT: z.coerce.number().int().default(5432),
  POSTGRES_DB: z.string().default("stylefind"),
  POSTGRES_USER: z.string().default("postgres"),
  POSTGRES_PASSWORD: z.string().default("postgres"),
  AI_SERVICE_URL: z.string().url().default("http://127.0.0.1:8001"),
  AI_TIMEOUT_MS: z.coerce.number().int().default(60000),
  AI_INTERNAL_TOKEN: z.string().min(8),
  AI_PRIMARY_MODEL: z.string().default("fashion-clip"),
  JWT_SECRET: z.string().min(16, "JWT_SECRET must be at least 16 characters"),
  JWT_EXPIRES_IN: z.string().default("8h"),
  CORS_ORIGINS: z.string().default("http://localhost:5173"),
  MAX_UPLOAD_MB: z.coerce.number().positive().default(8),
  STORE_SEARCH_HISTORY: z
    .string()
    .default("true")
    .transform((v) => v.toLowerCase() === "true"),
  PRODUCTS_DIR: z.string().default("dataset/products"),
});

const parsed = schema.safeParse(process.env);
if (!parsed.success) {
  console.error("Invalid environment configuration:", parsed.error.issues.map((i) => `${i.path}: ${i.message}`));
  process.exit(1);
}
const e = parsed.data;
if (e.NODE_ENV === "production") {
  const weak = [e.JWT_SECRET, e.AI_INTERNAL_TOKEN].some((v) => /change_me|sandbox|secret$/i.test(v) || v.length < 24);
  if (weak) {
    console.error("Refusing to start in production with default/weak JWT_SECRET or AI_INTERNAL_TOKEN (use 24+ random characters).");
    process.exit(1);
  }
}

export const env = {
  ...e,
  corsOrigins: e.CORS_ORIGINS.split(",").map((s) => s.trim()).filter(Boolean),
  maxUploadBytes: Math.round(e.MAX_UPLOAD_MB * 1024 * 1024),
  productsDir: path.isAbsolute(e.PRODUCTS_DIR) ? e.PRODUCTS_DIR : path.join(PROJECT_ROOT, e.PRODUCTS_DIR),
};
