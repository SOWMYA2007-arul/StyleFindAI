import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import cookieParser from "cookie-parser";
import cors from "cors";
import express from "express";
import helmet from "helmet";
import morgan from "morgan";
import { env, PROJECT_ROOT } from "./config/env.js";
import { errorHandler, notFound } from "./middleware/errors.js";
import { apiLimiter } from "./middleware/rateLimit.js";
import { apiRouter } from "./routes/index.js";

export function createApp() {
  const app = express();
  app.disable("x-powered-by");
  app.set("trust proxy", "loopback");
  app.use(helmet({ crossOriginResourcePolicy: { policy: "cross-origin" } }));
  // Request id: returned to the client and forwarded to the AI service so logs can be correlated.
  app.use((req, res, next) => {
    const incoming = String(req.headers["x-request-id"] ?? "");
    const id = /^[A-Za-z0-9-]{8,64}$/.test(incoming) ? incoming : crypto.randomUUID();
    res.locals.requestId = id;
    res.setHeader("X-Request-Id", id);
    next();
  });
  app.use(cookieParser());
  app.use(
    cors({
      origin: (origin, cb) => cb(null, !origin || env.corsOrigins.includes(origin)),
      methods: ["GET", "POST", "PUT", "DELETE"],
      allowedHeaders: ["Content-Type", "Authorization", "X-CSRF-Token", "X-Request-Id"],
      exposedHeaders: ["X-Request-Id", "Server-Timing"],
      credentials: true,
    }),
  );
  if (env.NODE_ENV !== "test") {
    morgan.token("rid", (_req, res) => String((res as express.Response).locals?.requestId ?? "-"));
    // method, url, status, time, request id - never bodies, cookies or Authorization headers
    app.use(morgan(":method :url :status :response-time ms rid=:rid"));
  }
  app.use(express.json({ limit: "100kb" }));
  app.use(express.urlencoded({ extended: false, limit: "100kb" }));

  // Catalogue images (only this folder is exposed; dotfiles and directory listings are refused).
  app.use("/media", express.static(env.productsDir, { dotfiles: "deny", index: false, fallthrough: false, maxAge: "7d" }));

  app.use("/api", apiLimiter, apiRouter);
  app.get("/api-docs", (_req, res) => res.type("text/markdown").sendFile(path.join(PROJECT_ROOT, "docs", "API.md")));

  // Production: serve the built React app from frontend/dist (npm run build in frontend/).
  const spa = path.join(PROJECT_ROOT, "frontend", "dist");
  if (fs.existsSync(path.join(spa, "index.html"))) {
    app.use(express.static(spa, { index: false }));
    app.get(/^\/(?!api|media).*/, (_req, res) => res.sendFile(path.join(spa, "index.html")));
  }

  app.use(notFound);
  app.use(errorHandler);
  return app;
}
