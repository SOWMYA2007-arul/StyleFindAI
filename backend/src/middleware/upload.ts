import path from "node:path";
import type { NextFunction, Request, Response } from "express";
import { imageSize } from "image-size";
import multer from "multer";
import { env } from "../config/env.js";
import { ApiError } from "./errors.js";

const ALLOWED_MIME = new Set(["image/jpeg", "image/png", "image/webp"]);
const ALLOWED_EXT = new Set([".jpg", ".jpeg", ".png", ".webp", ""]); // "" = pasted/blob uploads without a name
export const MIN_SIDE = 48; // Kaggle "small" catalogue thumbnails are 60x80
export const MAX_SIDE = 8000;
export const MAX_PIXELS = 40_000_000; // decompression-bomb guard

/**
 * Files stay in memory (never written to disk with a user-supplied name), so there are no temporary
 * files to clean up. Size, count and field limits are enforced while streaming.
 */
export const imageUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: env.maxUploadBytes, files: 1, fields: 20, fieldSize: 64 * 1024, parts: 25 },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname || "").toLowerCase();
    if (!ALLOWED_MIME.has(file.mimetype) || !ALLOWED_EXT.has(ext))
      return cb(new ApiError(415, "Only JPG, JPEG, PNG and WEBP images are allowed"));
    cb(null, true);
  },
});

/** Check the real file signature ("magic bytes") - the browser-sent MIME type and extension can be faked. */
export function sniffImageType(buf: Buffer): "jpeg" | "png" | "webp" | null {
  if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return "jpeg";
  if (buf.length >= 8 && buf.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return "png";
  if (buf.length >= 12 && buf.toString("ascii", 0, 4) === "RIFF" && buf.toString("ascii", 8, 12) === "WEBP") return "webp";
  return null;
}

export interface ImageCheck {
  type: "jpeg" | "png" | "webp";
  width: number;
  height: number;
}

/** Signature + header dimensions + pixel budget. Throws ApiError with a user-facing message. */
export function inspectImage(buf: Buffer): ImageCheck {
  const type = sniffImageType(buf);
  if (!type) throw new ApiError(415, "The file is not a valid JPG, PNG or WEBP image");
  let dims: { width?: number; height?: number; type?: string };
  try {
    dims = imageSize(new Uint8Array(buf));
  } catch {
    throw new ApiError(415, "The image is corrupt or truncated");
  }
  const { width = 0, height = 0 } = dims;
  if (!width || !height) throw new ApiError(415, "The image is corrupt or truncated");
  if (Math.min(width, height) < MIN_SIDE) throw new ApiError(422, `Image is too small (${width}x${height}); minimum side is ${MIN_SIDE}px`);
  if (Math.max(width, height) > MAX_SIDE) throw new ApiError(422, `Image is too large (${width}x${height}); maximum side is ${MAX_SIDE}px`);
  if (width * height > MAX_PIXELS) throw new ApiError(422, "Image has too many pixels");
  return { type, width, height };
}

export function requireImage(required = true) {
  return (req: Request, _res: Response, next: NextFunction) => {
    if (!req.file) return required ? next(new ApiError(400, "Please upload an image in the 'image' field")) : next();
    try {
      inspectImage(req.file.buffer);
      next();
    } catch (e) {
      next(e);
    }
  };
}
