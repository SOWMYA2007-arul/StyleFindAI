import { Router } from "express";
import * as admin from "../controllers/adminController.js";
import * as pub from "../controllers/publicController.js";
import { requireAdmin } from "../middleware/auth.js";
import { loginLimiter, searchLimiter } from "../middleware/rateLimit.js";
import { imageUpload, requireImage } from "../middleware/upload.js";

export const apiRouter = Router();

// ---------- public ----------
apiRouter.get("/health", pub.health);
apiRouter.post("/search/image", searchLimiter, imageUpload.single("image"), requireImage(true), pub.searchImage);
apiRouter.get("/products", pub.products);
apiRouter.get("/products/:id", pub.product);
apiRouter.get("/products/:id/similar", pub.similar);
apiRouter.get("/categories", pub.categories);
apiRouter.get("/filters", pub.facets);

// ---------- admin ----------
apiRouter.post("/auth/login", loginLimiter, admin.login);
apiRouter.get("/auth/session", admin.session);
apiRouter.get("/auth/me", requireAdmin, admin.me);
apiRouter.post("/auth/logout", requireAdmin, admin.logout);
apiRouter.get("/admin/stats", requireAdmin, admin.stats);
apiRouter.get("/admin/models", requireAdmin, admin.models);
apiRouter.get("/admin/products", requireAdmin, admin.adminProducts);
apiRouter.post("/admin/products", requireAdmin, imageUpload.single("image"), requireImage(true), admin.create);
apiRouter.put("/admin/products/:id", requireAdmin, imageUpload.single("image"), requireImage(false), admin.update);
apiRouter.delete("/admin/products/:id", requireAdmin, admin.remove);
apiRouter.post("/admin/embeddings/generate", requireAdmin, admin.generateEmbeddings);
apiRouter.post("/admin/products/:id/reindex", requireAdmin, admin.reindex);
apiRouter.get("/admin/indexing-failures", requireAdmin, admin.failures);
