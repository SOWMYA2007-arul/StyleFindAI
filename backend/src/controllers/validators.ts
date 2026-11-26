import { z } from "zod";

const csv = z
  .union([z.string(), z.array(z.string())])
  .optional()
  .transform((v) => (v === undefined ? undefined : (Array.isArray(v) ? v : v.split(",")).map((s) => s.trim()).filter(Boolean).slice(0, 20)));

export const listQuerySchema = z.object({
  category: csv,
  colour: csv,
  pattern: csv,
  sleeve_type: csv,
  neckline: csv,
  q: z.string().trim().max(100).optional(),
  min_price: z.coerce.number().nonnegative().optional(),
  max_price: z.coerce.number().nonnegative().optional(),
  sort: z.enum(["relevance", "name", "newest", "price_asc", "price_desc"]).default("relevance"),
  page: z.coerce.number().int().min(1).max(10000).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(24),
});

export const idParam = z.object({ id: z.coerce.number().int().positive() });

const filterValues = z.array(z.string().max(60)).max(20);
export const searchBodySchema = z.object({
  text: z.string().trim().max(300).optional().transform((v) => v || undefined),
  model: z.enum(["fashion-clip", "openclip-b32", "dinov2-b14", "cv-baseline"]).optional(),
  top_k: z.coerce.number().int().min(1).max(50).default(20),
  crop: z.enum(["true", "false"]).optional().transform((v) => v === "true"),
  // optional: which garment to search for in a multi-garment photo ("upper", "lower", "dupatta", or an index)
  target: z.string().trim().max(40).regex(/^[A-Za-z0-9 _-]*$/).optional().transform((v) => v || undefined),
  filters: z
    .string()
    .optional()
    .transform((s, ctx) => {
      if (!s) return undefined;
      try {
        return JSON.parse(s);
      } catch {
        ctx.addIssue({ code: "custom", message: "filters must be JSON" });
        return z.NEVER;
      }
    })
    .pipe(
      z
        .object({ category: filterValues, colour: filterValues, pattern: filterValues, sleeve_type: filterValues, neckline: filterValues,
                  min_price: z.number().nonnegative().max(10_000_000), max_price: z.number().nonnegative().max(10_000_000) })
        .partial()
        .strict()
        .optional(),
    ),
});

const optionalText = (max: number) => z.string().trim().max(max).optional();
const bool = z.union([z.boolean(), z.enum(["true", "false"]).transform((v) => v === "true")]);

export const productInputSchema = z.object({
  productCode: z.string().trim().regex(/^[A-Za-z0-9_-]{1,40}$/, "letters, digits, - and _ only").optional(),
  name: z.string().trim().min(2).max(200),
  category: z.string().trim().min(1).max(60),
  subcategory: optionalText(60),
  price: z.union([z.literal("").transform(() => null), z.coerce.number().nonnegative().max(10_000_000)]).optional(),
  description: optionalText(2000),
  colour: optionalText(40),
  pattern: optionalText(40),
  sleeveType: optionalText(40),
  neckline: optionalText(40),
  isActive: bool.optional(),
});
export const productUpdateSchema = productInputSchema.partial();

export const loginSchema = z.object({ email: z.string().trim().toLowerCase().email().max(200), password: z.string().min(1).max(200) });
export const embeddingJobSchema = z.object({
  productIds: z.array(z.number().int().positive()).max(5000).default([]),
  force: z.boolean().default(false),
  model: z.enum(["fashion-clip", "openclip-b32", "dinov2-b14", "cv-baseline"]).optional(),
});
