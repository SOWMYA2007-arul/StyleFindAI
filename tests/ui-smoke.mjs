/**
 * Browser end-to-end smoke test of the real website (no mocks).
 * Requires: AI service (8001), backend (4000), frontend dev server (5173) running and an admin account.
 *
 *   cd tests && npm install && npx playwright install chromium
 *   node ui-smoke.mjs --query ../dataset/evaluation/queries/<some file>.jpg --email admin@... --password ...
 *
 * Screenshots go to tests/screenshots/.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = path.dirname(fileURLToPath(import.meta.url));
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i > -1 ? process.argv[i + 1] : d; };
const BASE = arg("base", "http://127.0.0.1:5173");
const QUERY = path.resolve(arg("query"));
const NEW_PRODUCT_IMAGE = arg("new-image");
const EMAIL = arg("email"), PASSWORD = arg("password");
const SHOTS = path.join(here, "screenshots");
fs.mkdirSync(SHOTS, { recursive: true });

const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || undefined });
const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
const errors = [];
// Ignore web-font loading (offline machines) and the deliberate wrong-password 401.
const ignored = (url) => /fonts\.(googleapis|gstatic)\.com/.test(url) || url.endsWith("/api/auth/login");
page.on("console", (m) => m.type() === "error" && !ignored(m.location().url ?? "") && errors.push(`${m.text()} (${m.location().url})`));
page.on("pageerror", (e) => errors.push(e.message));
const check = (cond, msg) => { if (!cond) throw new Error(`FAILED: ${msg}`); console.log(`ok   ${msg}`); };
const shot = (n, full = true) => page.screenshot({ path: path.join(SHOTS, `${n}.png`), fullPage: full });

// 1. Home -> upload -> search -> results
await page.goto(BASE);
await page.getByRole("heading", { level: 1 }).waitFor();
await shot("01-home");
await page.setInputFiles('input[type="file"]', QUERY);
await page.getByAltText("Your uploaded photo").waitFor();
check(await page.getByRole("button", { name: "Find similar pieces" }).isEnabled(), "search button enabled after upload");
await page.getByRole("button", { name: "Find similar pieces" }).click();
await page.waitForURL("**/search/results", { timeout: 120_000 });
await page.getByRole("heading", { name: "Your matches" }).waitFor();
const cards = page.locator("article");
await cards.first().waitFor();
const n = await cards.count();
check(n >= 10 && n <= 20, `results page shows 10-20 products (got ${n})`);
const imgsLoaded = await page.$$eval("article img", (els) => els.filter((e) => e.complete && e.naturalWidth > 0).length);
await page.waitForTimeout(800);
check(imgsLoaded > 0, "result images load from /media");
await shot("02-results");
await cards.first().getByRole("button", { name: "Why this result?" }).click();
check(await cards.first().locator("li").count() > 0, "recommendation reasons expand");
const firstName = await cards.first().locator("a").nth(1).innerText();

// server-side filters on results: choose one category, apply -> the AI search re-runs with the filter
await page.locator("details summary").filter({ hasText: "Filter the catalogue search" }).click();
const filterForm = page.getByRole("form", { name: "Filter results" });
const firstCat = filterForm.locator("fieldset").first().locator("label").first();
const catName = (await firstCat.locator("span").first().innerText()).trim();
await firstCat.locator("input").check();
const searchResp = page.waitForResponse((r) => r.url().includes("/api/search/image") && r.request().method() === "POST");
await filterForm.getByRole("button", { name: "Apply filters" }).click();
const resp = await searchResp;
check(resp.status() === 200, "filters re-run the real search on the server");
const applied = (await resp.json()).filters ?? {};
check(Array.isArray(applied.category) && applied.category.length === 1, `server applied filter ${JSON.stringify(applied)}`);
await page.waitForTimeout(300);
const catsShown = await page.$$eval("article", (els) => els.map((e) => e.querySelector("p")?.textContent?.trim()));
check(catsShown.length > 0 && catsShown.every((c) => c === catName), `all ${catsShown.length} results are "${catName}"`);
await filterForm.getByRole("button", { name: "Clear filters" }).click();
await page.waitForResponse((r) => r.url().includes("/api/search/image"));
await page.getByLabel("Sort by").selectOption("colour");
check(await cards.count() > 0, "sort by colour works");

// 2. Product details + similar
await page.getByLabel("Sort by").selectOption("final");
await cards.first().locator("a").nth(1).click();
await page.waitForURL("**/products/*");
await page.getByRole("heading", { level: 1, name: firstName }).waitFor();
check((await page.getByRole("heading", { level: 1 }).innerText()) === firstName, "product page opens for clicked result");
await page.getByRole("heading", { name: "Visually similar" }).waitFor();
await page.locator("article").first().waitFor({ timeout: 30_000 });
check(await page.locator("article").count() > 0, "product page shows visually similar products");
await shot("03-product");

// 3. Catalogue with filters + pagination
await page.goto(`${BASE}/catalogue`);
await page.getByText(/\d+ products/).waitFor();
const total = Number((await page.getByText(/\d+ products/).innerText()).split(" ")[0]);
await page.getByRole("checkbox", { name: /Sarees/ }).click();
await page.waitForURL(/category=sarees/);
await page.waitForFunction(() => [...document.querySelectorAll("label")].some((l) => l.textContent.startsWith("Sarees") && l.querySelector("input").checked));
await page.waitForTimeout(500);
const sarees = Number((await page.getByText(/\d+ products/).innerText()).split(" ")[0]);
check(sarees > 0 && sarees < total, `catalogue category filter (${total} -> ${sarees})`);
await page.getByRole("button", { name: "Clear all filters" }).click();
await page.getByRole("button", { name: "Next" }).click();
await page.waitForURL(/page=2/);
check(true, "catalogue pagination");
await shot("04-catalogue");

// 4. Invalid file on the search page
await page.goto(`${BASE}/search`);
const bad = path.join(SHOTS, "not-an-image.txt");
fs.writeFileSync(bad, "hello");
await page.setInputFiles('input[type="file"]', { name: "fake.jpg", mimeType: "image/gif", buffer: Buffer.from("GIF89a") });
check(await page.getByRole("alert").isVisible(), "invalid file type shows an error");

// 5. Admin
if (EMAIL && PASSWORD) {
  await page.goto(`${BASE}/admin`);
  await page.waitForURL("**/admin/login");
  await page.getByLabel("Email").fill(EMAIL);
  await page.getByLabel("Password").fill("wrong password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByRole("alert").waitFor();
  check(true, "wrong password rejected");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByRole("heading", { name: "Dashboard" }).waitFor();
  await page.getByText("Search index").waitFor();
  await page.locator("table tbody tr").first().waitFor();
  check(true, "admin dashboard loads stats");
  await shot("05-admin-dashboard");
  await page.getByRole("link", { name: "Products" }).click();
  await page.locator("table tbody tr").first().waitFor();
  await shot("06-admin-products", false);
  if (NEW_PRODUCT_IMAGE) {
    await page.getByRole("button", { name: "Add product" }).click();
    await page.setInputFiles('input[type="file"]', path.resolve(NEW_PRODUCT_IMAGE));
    await page.getByLabel("Product name").fill("UI Test Kurta");
    await page.getByLabel("Colour").fill("teal");
    await page.getByRole("button", { name: "Add product" }).click();
    await page.getByRole("status").filter({ hasText: "Product added" }).waitFor({ timeout: 60_000 });
    const msg = await page.getByRole("status").filter({ hasText: "Product added" }).innerText();
    check(msg.includes("indexed for visual search"), `admin add product: "${msg}"`);
    await page.getByLabel("Search products").fill("UI Test Kurta");
    await page.getByRole("button", { name: "Search" }).click();
    await page.getByRole("button", { name: "Delete UI Test Kurta" }).click();
    await page.getByRole("dialog").waitFor();
    await page.getByRole("button", { name: "Delete product" }).click();
    await page.getByRole("status").filter({ hasText: "deleted" }).waitFor();
    check(true, "admin delete product (confirmation dialog)");
  }
}

if (EMAIL && PASSWORD) {
  // logout must revoke the session: the admin area redirects to login afterwards
  await page.getByRole("button", { name: "Log out" }).click();
  await page.waitForURL("**/admin/login");
  await page.goto(`${BASE}/admin`);
  await page.waitForURL("**/admin/login");
  check(true, "logout revokes the session");
}

// 6. Mobile layout
await page.setViewportSize({ width: 390, height: 844 });
await page.goto(BASE);
await page.getByRole("heading", { level: 1 }).waitFor();
await shot("07-home-mobile", false);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check(overflow <= 1, `no horizontal scrolling on a 390px phone screen (overflow ${overflow}px)`);

check(errors.length === 0, `no browser console errors${errors.length ? ": " + errors.join(" | ") : ""}`);
await browser.close();
console.log("UI smoke test passed");
