"""Image (and text) encoders that turn pictures into embedding vectors.

An *embedding* is a list of numbers (512 or 768 here) produced by a neural network such that
visually/semantically similar images end up close together. We L2-normalise every vector so
cosine similarity == dot product, which is exactly what pgvector's `<=>` operator measures.

IMPORTANT: vectors from different models live in different "spaces" and must never be
compared with each other. Every vector is therefore stored with (model_key, model_version).
"""
from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable

import numpy as np
from PIL import Image

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("encoders")


class ModelUnavailableError(RuntimeError):
    """The model could not be loaded (missing weights, failed download, wrong dimensions...)."""


def l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class Encoder(ABC):
    key: str
    version: str
    name: str
    dims: int
    supports_text: bool = False

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return an (N, dims) float32 array of L2-normalised embeddings."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.key} has no text encoder")


# --------------------------------------------------------------------------- deep models
def _torch():
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    return torch


def _features_tensor(out):
    """transformers v4 returns a tensor, v5 may return a ModelOutput; handle both."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "image_embeds"):
        return out.image_embeds
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class HFClipEncoder(Encoder):
    """CLIP-architecture models on the Hugging Face hub (FashionCLIP is CLIP ViT-B/32
    fine-tuned on ~800k fashion product image/caption pairs)."""

    supports_text = True

    def __init__(self, key: str, repo_id: str, version: str, name: str, dims: int = 512,
                 model=None, processor=None, revision: str | None = None):
        self.key, self.version, self.name, self.dims = key, version, name, dims
        torch = _torch()
        self.device = get_settings().ai_device
        if model is None:
            from transformers import CLIPModel, CLIPProcessor

            cache = str(get_settings().model_cache_path)
            log.info("Loading %s from %s (first run downloads ~600 MB)", key, repo_id)
            kw = {"cache_dir": cache, **({"revision": revision} if revision else {})}
            model = CLIPModel.from_pretrained(repo_id, **kw)
            processor = CLIPProcessor.from_pretrained(repo_id, **kw)
        self.model = model.to(self.device).eval()
        self.processor = processor
        self._torch = torch

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        # The processor resizes the short side to 224, centre-crops 224x224 and applies
        # CLIP's mean/std normalisation - the exact preprocessing the model was trained with.
        proc = getattr(self.processor, "image_processor", self.processor)
        inputs = proc(images=[im.convert("RGB") for im in images], return_tensors="pt")
        with torch.inference_mode():
            feats = _features_tensor(self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device)))
        return l2_normalise(feats.float().cpu().numpy())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            raise NotImplementedError("Processor has no tokenizer")
        inputs = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        with torch.inference_mode():
            feats = _features_tensor(self.model.get_text_features(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device)))
        return l2_normalise(feats.float().cpu().numpy())


class OpenClipEncoder(Encoder):
    supports_text = True

    def __init__(self, key: str, arch: str, pretrained: str, version: str, name: str, dims: int = 512):
        import open_clip

        self.key, self.version, self.name, self.dims = key, version, name, dims
        self._torch = _torch()
        self.device = get_settings().ai_device
        log.info("Loading OpenCLIP %s/%s", arch, pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, cache_dir=str(get_settings().model_cache_path))
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(arch)

    def embed_images(self, images):
        torch = self._torch
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with torch.inference_mode():
            feats = self.model.encode_image(batch)
        return l2_normalise(feats.float().cpu().numpy())

    def embed_texts(self, texts):
        torch = self._torch
        with torch.inference_mode():
            feats = self.model.encode_text(self.tokenizer(texts).to(self.device))
        return l2_normalise(feats.float().cpu().numpy())


class DinoV2Encoder(Encoder):
    """Self-supervised ViT. Strong at fine visual detail (prints, embroidery) but has
    no text side, so it cannot do image+text search."""

    def __init__(self, key: str, repo_id: str, version: str, name: str, dims: int = 768, revision: str | None = None):
        from transformers import AutoImageProcessor, AutoModel

        self.key, self.version, self.name, self.dims = key, version, name, dims
        self._torch = _torch()
        self.device = get_settings().ai_device
        cache = str(get_settings().model_cache_path)
        log.info("Loading DINOv2 from %s", repo_id)
        kw = {"cache_dir": cache, **({"revision": revision} if revision else {})}
        self.processor = AutoImageProcessor.from_pretrained(repo_id, **kw)
        self.model = AutoModel.from_pretrained(repo_id, **kw).to(self.device).eval()

    def embed_images(self, images):
        torch = self._torch
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        with torch.inference_mode():
            out = self.model(pixel_values=inputs["pixel_values"].to(self.device))
        cls = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return l2_normalise(cls.float().cpu().numpy())


# --------------------------------------------------------------------------- classical baseline
class CVBaselineEncoder(Encoder):
    """Hand-crafted features (no neural network, no download).

    512 dims = HSV colour histogram of the garment (128) + HOG shape descriptor (324)
             + LBP texture histograms (58) + 2 global stats.
    It is a genuine (if weak) visual retrieval method: useful as a sanity baseline, for
    offline testing, and to show how much the deep models actually add.
    """

    # v2: uses the improved foreground mask (fg-v2) -> different vectors than v1, stored separately
    key, version, name, dims = "cv-baseline", "cv-v2", "Classical CV features (HSV hist + LBP + HOG)", 512

    def _one(self, img: Image.Image) -> np.ndarray:
        import cv2
        from skimage.feature import hog, local_binary_pattern

        from app.services.foreground import crop_to_mask, foreground_mask

        rgb, mask = foreground_mask(img)
        rgb_c, mask_c = crop_to_mask(rgb, mask)
        hsv = cv2.cvtColor(rgb_c, cv2.COLOR_RGB2HSV)
        m8 = mask_c.astype(np.uint8)
        hist = cv2.calcHist([hsv], [0, 1, 2], m8, [16, 4, 2], [0, 180, 0, 256, 0, 256]).flatten()
        hist = np.sqrt(hist / max(1.0, hist.sum()))                          # 128

        gray = cv2.cvtColor(rgb_c, cv2.COLOR_RGB2GRAY)
        gray64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        h = hog(gray64, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                feature_vector=True).astype(np.float32)                        # 324
        h = h / max(1e-6, np.linalg.norm(h))

        lbps = []
        for p, r in ((8, 1), (16, 2)):
            lbp = local_binary_pattern(gray, p, r, method="uniform")
            if p == 8:  # 4 spatial quadrants x 10 bins
                hh, ww = lbp.shape
                for qy in (slice(0, hh // 2), slice(hh // 2, hh)):
                    for qx in (slice(0, ww // 2), slice(ww // 2, ww)):
                        q = lbp[qy, qx][mask_c[qy, qx]]
                        hq, _ = np.histogram(q, bins=10, range=(0, 10))
                        lbps.append(np.sqrt(hq / max(1, hq.sum())))
            else:
                hq, _ = np.histogram(lbp[mask_c], bins=18, range=(0, 18))
                lbps.append(np.sqrt(hq / max(1, hq.sum())))
        lbpv = np.concatenate(lbps).astype(np.float32)                        # 58
        lbpv = lbpv / max(1e-6, np.linalg.norm(lbpv))

        extra = np.array([mask.mean(), rgb_c.shape[0] / max(1, rgb_c.shape[1]) / 3.0], np.float32)  # 2
        vec = np.concatenate([1.0 * hist / max(1e-6, np.linalg.norm(hist)), 0.8 * h, 0.6 * lbpv, 0.3 * extra])
        assert vec.shape[0] == 512, vec.shape
        return vec

    def embed_images(self, images):
        return l2_normalise(np.stack([self._one(im) for im in images]))


# --------------------------------------------------------------------------- registry
def _pinned(base: str, revision: str) -> str:
    """model_version string. Unpinned keeps the original value so existing embeddings stay valid;
    pinning a revision creates a NEW version (its embeddings are stored separately)."""
    return f"{base}@{revision}" if revision else base


_S = get_settings()
MODEL_SPECS: dict[str, dict] = {
    "fashion-clip": {"name": "FashionCLIP (CLIP ViT-B/32 fine-tuned on fashion)", "dims": 512, "text": True,
                     "approx_ram_mb": 1500, "download_mb": 600,
                     "version": _pinned("hf:patrickjohncyh/fashion-clip", _S.fashion_clip_revision)},
    "openclip-b32": {"name": "OpenCLIP ViT-B/32 (LAION-2B)", "dims": 512, "text": True,
                     "approx_ram_mb": 1500, "download_mb": 600,
                     "version": "openclip:ViT-B-32/laion2b_s34b_b79k"},
    "dinov2-b14": {"name": "DINOv2 ViT-B/14 (CLS token)", "dims": 768, "text": False,
                   "approx_ram_mb": 1000, "download_mb": 350,
                   "version": _pinned("hf:facebook/dinov2-base", _S.dinov2_revision)},
    "cv-baseline": {"name": CVBaselineEncoder.name, "dims": 512, "text": False, "approx_ram_mb": 50,
                    "download_mb": 0, "version": "cv-v2"},
}
MODEL_SPECS["fashion-clip"]["factory"] = lambda: HFClipEncoder(
    "fashion-clip", "patrickjohncyh/fashion-clip", MODEL_SPECS["fashion-clip"]["version"],
    MODEL_SPECS["fashion-clip"]["name"], revision=_S.fashion_clip_revision or None)
MODEL_SPECS["openclip-b32"]["factory"] = lambda: OpenClipEncoder(
    "openclip-b32", "ViT-B-32", "laion2b_s34b_b79k", MODEL_SPECS["openclip-b32"]["version"],
    MODEL_SPECS["openclip-b32"]["name"])
MODEL_SPECS["dinov2-b14"]["factory"] = lambda: DinoV2Encoder(
    "dinov2-b14", "facebook/dinov2-base", MODEL_SPECS["dinov2-b14"]["version"], MODEL_SPECS["dinov2-b14"]["name"],
    revision=_S.dinov2_revision or None)
MODEL_SPECS["cv-baseline"]["factory"] = CVBaselineEncoder


class EncoderRegistry:
    """Loads encoders on first use and keeps at most `max_loaded` in RAM (LRU)."""

    def __init__(self, max_loaded: int = 1):
        self.max_loaded = max(1, max_loaded)
        self._loaded: OrderedDict[str, Encoder] = OrderedDict()
        self._lock = threading.Lock()
        self._overrides: dict[str, Callable[[], Encoder]] = {}
        self._failures: dict[str, tuple[float, str]] = {}

    def register_override(self, key: str, factory: Callable[[], Encoder]) -> None:
        """Used by tests to inject a model without network access."""
        self._overrides[key] = factory
        self._loaded.pop(key, None)
        self._failures.pop(key, None)

    def get(self, key: str) -> Encoder:
        if key not in MODEL_SPECS:
            raise KeyError(f"Unknown model '{key}'. Choose from {list(MODEL_SPECS)}")
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
            failed_at, reason = self._failures.get(key, (0.0, ""))
            if time.time() - failed_at < 60:
                raise ModelUnavailableError(reason)
            while len(self._loaded) >= self.max_loaded:
                old, _ = self._loaded.popitem(last=False)
                log.info("Unloading %s to free memory", old)
                import gc

                gc.collect()
            factory = self._overrides.get(key) or MODEL_SPECS[key]["factory"]
            try:
                enc = factory()
                probe = enc.embed_images([Image.new("RGB", (64, 64), (128, 128, 128))])
            except ModelUnavailableError:
                raise
            except Exception as e:  # noqa: BLE001 - download failure, missing package, corrupt weights...
                msg = (f"Model '{key}' could not be loaded ({type(e).__name__}). Check the internet connection for the "
                       f"first download (~{MODEL_SPECS[key].get('download_mb', '?')} MB) and free RAM "
                       f"(~{MODEL_SPECS[key].get('approx_ram_mb', '?')} MB), or use another model.")
                log.error("%s Details: %s", msg, e)
                self._failures[key] = (time.time(), msg)
                raise ModelUnavailableError(msg) from e
            expected = MODEL_SPECS[key]["dims"]
            if probe.shape != (1, expected) or enc.dims != expected or enc.version != MODEL_SPECS[key]["version"]:
                msg = (f"Model '{key}' produced {probe.shape[-1]}-d vectors / version '{enc.version}', expected "
                       f"{expected}-d / '{MODEL_SPECS[key]['version']}'. Refusing to mix embedding spaces.")
                self._failures[key] = (time.time(), msg)
                raise ModelUnavailableError(msg)
            self._failures.pop(key, None)
            self._loaded[key] = enc
            return enc

    def loaded_keys(self) -> list[str]:
        return list(self._loaded)


_registry: EncoderRegistry | None = None


def get_registry() -> EncoderRegistry:
    global _registry
    if _registry is None:
        _registry = EncoderRegistry(get_settings().ai_max_loaded_models)
    return _registry
