/**
 * Integration tests: real PostgreSQL + the real Python AI service must be running
 * (uvicorn on AI_SERVICE_URL) with catalogue embeddings for AI_PRIMARY_MODEL.
 *   npm test
 */
import fs from "node:fs";
import path from "node:path";
import bcrypt from "bcryptjs";
import request from "supertest";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createApp } from "../src/app.js";
import { pool, query } from "../src/config/db.js";
import { env } from "../src/config/env.js";
import { sniffImageType } from "../src/middleware/upload.js";

const app = createApp();
const ADMIN = { email: "test-admin@stylefind.local", password: "test-password-123" };
let token = "";
let sample: { id: number; key: string; code: string };

beforeAll(async () => {
  await query(
    `INSERT INTO admin_users (email, password_hash) VALUES ($1,$2) ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash`,
    [ADMIN.email, await bcrypt.hash(ADMIN.password, 4)],
  );
  const r = await query(`SELECT p.id, p.product_code, pi.storage_key FROM products p JOIN product_images pi ON pi.product_id = p.id
                         WHERE p.source_dataset <> 'admin_upload' ORDER BY p.id LIMIT 1`);
  sample = { id: Number(r.rows[0].id), key: r.rows[0].storage_key, code: r.rows[0].product_code };
});

afterAll(async () => {
  await query("DELETE FROM admin_users WHERE email = $1", [ADMIN.email]);
  await pool.end();
});

const img = () => fs.readFileSync(path.join(env.productsDir, sample.key));

describe("upload validation (unit)", () => {
  it("detects real image signatures", () => {
    expect(sniffImageType(img())).toBe("jpeg");
    expect(sniffImageType(Buffer.from("GIF89a....."))).toBeNull();
    expect(sniffImageType(Buffer.from("<?php echo 1; ?>"))).toBeNull();
  });
});

describe("public API", () => {
  it("health reports database and AI service", async () => {
    const r = await request(app).get("/api/health");
    expect(r.status).toBe(200);
    expect(r.body.database).toBe("ok");
    expect(r.body.ai.status).toBe("ok");
  });

  it("lists products with pagination and filters", async () => {
    const r = await request(app).get("/api/products?limit=5&page=1");
    expect(r.status).toBe(200);
    expect(r.body.items).toHaveLength(5);
    expect(r.body.total).toBeGreaterThan(5);
    expect(r.body.items[0].imageUrl).toMatch(/^\/media\//);
    const f = await request(app).get("/api/products?category=sarees&limit=100");
    expect(f.body.items.every((p: { category: string }) => p.category === "sarees")).toBe(true);
    const bad = await request(app).get("/api/products?limit=5000");
    expect(bad.status).toBe(400);
  });

  it("serves product details, categories, filters and real image files", async () => {
    const p = await request(app).get(`/api/products/${sample.id}`);
    expect(p.status).toBe(200);
    expect(p.body.productCode).toBe(sample.code);
    const media = await request(app).get(p.body.imageUrl);
    expect(media.status).toBe(200);
    expect(media.headers["content-type"]).toMatch(/image/);
    expect((await request(app).get("/api/products/99999999")).status).toBe(404);
    expect((await request(app).get("/api/products/abc")).status).toBe(400);
    const cats = await request(app).get("/api/categories");
    // the six original categories are always there; imports (migration 004) may add more
    expect(cats.body.length).toBeGreaterThanOrEqual(6);
    expect(cats.body.map((c: { slug: string }) => c.slug)).toEqual(
      expect.arrayContaining(["kurtas-kurtis", "sarees", "lehengas", "western-dresses", "tops-shirts", "other-ethnic"]));
    const filters = await request(app).get("/api/filters");
    expect(Object.keys(filters.body.attributes)).toContain("colour");
  });

  it("blocks path traversal on /media", async () => {
    const r = await request(app).get("/media/..%2f..%2f.env");
    expect([400, 403, 404]).toContain(r.status);
    expect(r.text).not.toContain("POSTGRES_PASSWORD");
  });

  it("returns visually similar products for a product page", async () => {
    const r = await request(app).get(`/api/products/${sample.id}/similar`);
    expect(r.status).toBe(200);
    expect(r.body.results.length).toBeGreaterThan(0);
    expect(r.body.results.every((x: { id: number }) => x.id !== sample.id)).toBe(true);
  });
});

describe("POST /api/search/image (end-to-end through the AI service)", () => {
  it("finds the uploaded catalogue product as an exact match", async () => {
    const r = await request(app).post("/api/search/image").attach("image", img(), "photo.jpg").field("top_k", "12");
    expect(r.status).toBe(200);
    expect(r.body.count).toBe(12);
    expect(r.body.results[0].productCode).toBe(sample.code);
    expect(r.body.results[0].matchType).toBe("exact");
    expect(r.body.results[0].imageUrl).toMatch(/^\/media\//);
    expect(r.body.query.sha256).toBeUndefined();
    expect(r.body.searchId).toBeTruthy();
    expect(r.headers["server-timing"]).toMatch(/total;dur=/);
    expect(r.headers["x-request-id"]).toBeTruthy();
    expect(r.body.timingsMs.backend_to_ai_roundtrip_ms).toBeGreaterThan(0);
    const h = await query("SELECT result_count FROM search_history WHERE id = $1", [r.body.searchId]);
    expect(h.rows[0].result_count).toBe(12);
  });

  it("accepts text + filters", async () => {
    const r = await request(app)
      .post("/api/search/image")
      .attach("image", img(), "photo.jpg")
      .field("text", "same style in blue with full sleeves")
      .field("filters", JSON.stringify({ category: ["kurtas-kurtis", "tops-shirts"], min_price: 0, max_price: 100000 }));
    expect(r.status).toBe(200);
    expect(r.body.query.requested_attributes.colour).toBe("blue");
    expect(r.body.results.every((x: { category: string }) => ["kurtas-kurtis", "tops-shirts"].includes(x.category))).toBe(true);
  });

  it("rejects missing, fake and oversized files", async () => {
    expect((await request(app).post("/api/search/image")).status).toBe(400);
    const fake = await request(app).post("/api/search/image").attach("image", Buffer.from("not an image"), { filename: "x.jpg", contentType: "image/jpeg" });
    expect(fake.status).toBe(415);
    const gif = await request(app).post("/api/search/image").attach("image", Buffer.from("GIF89a"), { filename: "x.gif", contentType: "image/gif" });
    expect(gif.status).toBe(415);
    const big = await request(app).post("/api/search/image").attach("image", Buffer.alloc(env.maxUploadBytes + 10, 0xff), { filename: "big.jpg", contentType: "image/jpeg" });
    expect(big.status).toBe(413);
    const badFilters = await request(app).post("/api/search/image").attach("image", img(), "p.jpg").field("filters", "{nope");
    expect(badFilters.status).toBe(400);
  });
});

describe("admin", () => {
  it("rejects unauthenticated access and bad logins", async () => {
    expect((await request(app).get("/api/admin/stats")).status).toBe(401);
    expect((await request(app).delete(`/api/admin/products/${sample.id}`)).status).toBe(401);
    expect((await request(app).get("/api/admin/stats").set("Authorization", "Bearer forged.token.value")).status).toBe(401);
    const bad = await request(app).post("/api/auth/login").send({ email: ADMIN.email, password: "wrong-password" });
    expect(bad.status).toBe(401);
  });

  it("logs in and reads statistics", async () => {
    const r = await request(app).post("/api/auth/login?mode=bearer").send(ADMIN);
    expect(r.status).toBe(200);
    token = r.body.token;
    const s = await request(app).get("/api/admin/stats").set("Authorization", `Bearer ${token}`);
    expect(s.status).toBe(200);
    expect(s.body.products).toBeGreaterThan(0);
    expect(s.body.embeddings.length).toBeGreaterThan(0);
  });

  it("creates (with embedding), finds via AI search, updates and deletes a product", async () => {
    // Use a horizontally flipped catalogue image so it is a NEW image (sha256 differs) but a real garment.
    const { execFileSync } = await import("node:child_process");
    const tmp = path.join(env.productsDir, "..", "uploads", "test-flipped.png");
    const venv = path.join(path.dirname(env.productsDir), "..", "ai-service", ".venv");
    const candidates = [process.env.PYTHON, path.join(venv, "Scripts", "python.exe"), path.join(venv, "bin", "python")];
    const py = candidates.find((p) => p && fs.existsSync(p)) ?? (process.platform === "win32" ? "python" : "python3");
    execFileSync(py, ["-c",
      `from PIL import Image,ImageOps; ImageOps.mirror(Image.open(r"${path.join(env.productsDir, sample.key)}").convert("RGB")).save(r"${tmp}")`]);
    const buf = fs.readFileSync(tmp);
    fs.rmSync(tmp);

    const created = await request(app)
      .post("/api/admin/products")
      .set("Authorization", `Bearer ${token}`)
      .field("name", "Test Admin Kurta")
      .field("category", "kurtas-kurtis")
      .field("colour", "Red")
      .field("price", "1499")
      .attach("image", buf, "anything../../evil.png");
    expect(created.status).toBe(201);
    expect(created.body.embedding.status).toBe("generated");
    const id = created.body.product.id;
    expect(created.body.product.imageUrl).toMatch(/^\/media\/admin-uploads\/[0-9a-f-]{36}\.png$/);

    const dup = await request(app).post("/api/admin/products").set("Authorization", `Bearer ${token}`)
      .field("name", "Duplicate").field("category", "sarees").attach("image", buf, "x.png");
    expect(dup.status).toBe(409);

    const s = await request(app).post("/api/search/image").attach("image", buf, "q.png");
    expect(s.body.results[0].id).toBe(id);

    const upd = await request(app).put(`/api/admin/products/${id}`).set("Authorization", `Bearer ${token}`)
      .field("name", "Renamed Kurta").field("pattern", "Solid");
    expect(upd.status).toBe(200);
    expect(upd.body.product.name).toBe("Renamed Kurta");
    expect(upd.body.product.pattern).toBe("solid");

    const del = await request(app).delete(`/api/admin/products/${id}`).set("Authorization", `Bearer ${token}`);
    expect(del.status).toBe(200);
    expect((await request(app).get(`/api/products/${id}`)).status).toBe(404);
    expect(fs.existsSync(path.join(env.productsDir, created.body.product.imageUrl.replace("/media/", "")))).toBe(false);
  });

  it("browser login uses an HttpOnly cookie + CSRF token, and logout revokes the session", async () => {
    const agent = request.agent(app);
    const r = await agent.post("/api/auth/login").send(ADMIN);
    expect(r.status).toBe(200);
    expect(r.body.token).toBeUndefined(); // the JWT is never exposed to page JavaScript
    const cookies = ([] as string[]).concat(r.headers["set-cookie"] ?? []);
    const session = cookies.find((c) => c.startsWith("sf_session="))!;
    expect(session).toMatch(/HttpOnly/i);
    expect(session).toMatch(/SameSite=Strict/i);
    expect(cookies.find((c) => c.startsWith("sf_csrf="))).not.toMatch(/HttpOnly/i);
    const csrf = r.body.csrfToken as string;
    expect((await agent.get("/api/admin/stats")).status).toBe(200); // GET: cookie is enough
    const noCsrf = await agent.post("/api/admin/embeddings/generate").send({ productIds: [sample.id] });
    expect(noCsrf.status).toBe(403); // state change without the CSRF header is refused
    const withCsrf = await agent.post("/api/admin/embeddings/generate").set("X-CSRF-Token", csrf).send({ productIds: [sample.id] });
    expect(withCsrf.status).toBe(200);
    // a bearer token issued before logout must also stop working
    const bearer = (await request(app).post("/api/auth/login?mode=bearer").send(ADMIN)).body.token;
    expect((await agent.post("/api/auth/logout").set("X-CSRF-Token", csrf)).status).toBe(200);
    expect((await agent.get("/api/admin/stats")).status).toBe(401);
    expect((await request(app).get("/api/admin/stats").set("Authorization", `Bearer ${bearer}`)).status).toBe(401);
    token = (await request(app).post("/api/auth/login?mode=bearer").send(ADMIN)).body.token; // fresh token for later tests
  });

  it("rejects too-small, wrong-extension and truncated uploads", async () => {
    const auth = { Authorization: `Bearer ${token}` };
    const png1x1 = Buffer.from("89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000A49444154789C6300010000050001ABE3D8B60000000049454E44AE426082", "hex");
    const small = await request(app).post("/api/search/image").attach("image", png1x1, "tiny.png");
    expect(small.status).toBe(422);
    const ext = await request(app).post("/api/search/image").attach("image", img(), { filename: "photo.exe", contentType: "image/jpeg" });
    expect(ext.status).toBe(415);
    const truncated = await request(app).post("/api/admin/products").set(auth).field("name", "Truncated").field("category", "sarees")
      .attach("image", img().subarray(0, 20), "t.jpg");
    expect([415, 422]).toContain(truncated.status);
  });

  it("replaces an image safely: failed update leaves no file, successful update re-indexes", async () => {
    const auth = { Authorization: `Bearer ${token}` };
    const { execFileSync } = await import("node:child_process");
    const venv = path.join(path.dirname(env.productsDir), "..", "ai-service", ".venv");
    const py = [process.env.PYTHON, path.join(venv, "Scripts", "python.exe"), path.join(venv, "bin", "python")]
      .find((p) => p && fs.existsSync(p)) ?? (process.platform === "win32" ? "python" : "python3");
    const make = (angle: number) => {
      const out = path.join(env.productsDir, "..", "uploads", `t${angle}.png`);
      execFileSync(py, ["-c", `from PIL import Image; Image.open(r"${path.join(env.productsDir, sample.key)}").convert("RGB").rotate(${angle}, fillcolor=(250,250,250)).save(r"${out}")`]);
      const b = fs.readFileSync(out); fs.rmSync(out); return b;
    };
    const created = await request(app).post("/api/admin/products").set(auth).field("name", "Replace Test").field("category", "sarees")
      .attach("image", make(7), "a.png");
    expect(created.status).toBe(201);
    const id = created.body.product.id;
    const adminDir = path.join(env.productsDir, "admin-uploads");
    const before = new Set(fs.readdirSync(adminDir));
    const bad = await request(app).put(`/api/admin/products/${id}`).set(auth).field("category", "no-such-category").attach("image", make(11), "b.png");
    expect(bad.status).toBe(400);
    expect(new Set(fs.readdirSync(adminDir))).toEqual(before); // the new file was cleaned up
    const good = await request(app).put(`/api/admin/products/${id}`).set(auth).attach("image", make(13), "c.png");
    expect(good.status).toBe(200);
    expect(good.body.embedding.status).toBe("generated");
    expect(fs.existsSync(path.join(adminDir, created.body.product.imageUrl.split("/").pop()))).toBe(false); // old file removed
    const st = await query("SELECT index_status, (SELECT count(*) FROM image_embeddings e WHERE e.product_image_id = pi.id) AS n FROM product_images pi WHERE product_id = $1", [id]);
    expect(st.rows[0].index_status).toBe("indexed");
    expect(Number(st.rows[0].n)).toBeGreaterThan(0);
    const re = await request(app).post(`/api/admin/products/${id}/reindex`).set(auth);
    expect(re.body.embedding.status).toBe("generated");
    expect((await request(app).get("/api/admin/indexing-failures").set(auth)).status).toBe(200);
    const details = await request(app).get(`/api/products/${id}`);
    expect(details.body.images.length).toBeGreaterThan(0);
    await request(app).delete(`/api/admin/products/${id}`).set(auth);
  });

  it("validates product input", async () => {
    const r = await request(app).post("/api/admin/products").set("Authorization", `Bearer ${token}`)
      .field("name", "X").field("category", "sarees").attach("image", img(), "a.jpg");
    expect(r.status).toBe(400);
    const r2 = await request(app).post("/api/admin/products").set("Authorization", `Bearer ${token}`)
      .field("name", "Valid name").field("category", "nope").attach("image", img(), "a.jpg");
    expect([400, 409]).toContain(r2.status);
  });
});
