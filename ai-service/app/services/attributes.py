"""Predict attributes of the QUERY image (never written into the catalogue tables).

Three sources, from cheapest to heaviest:
  1. pixels       -> colour palette / dominant colour name (always available)
  2. zero-shot    -> CLIP-style models compare the image with text prompts such as
                     "a photo of a striped garment"; we keep a value only if its probability is
                     high enough, otherwise report 'unknown'
  3. VLM (opt.)   -> Qwen2.5-VL answers a structured JSON question (off by default; heavy)
If the encoder has no text side (DINOv2 / cv-baseline) zero-shot is skipped and the category is
estimated from the categories of the nearest catalogue neighbours (retrieval voting).
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
from PIL import Image

from app.config import get_settings
from app.models.encoders import Encoder
from app.services.colour import extract_palette, is_multicolour
from app.services.foreground import estimate_foreground
from app.services.pattern import classify_pattern, pattern_descriptor
from app.utils.logging import get_logger

log = get_logger("attributes")

ZERO_SHOT: dict[str, dict[str, str]] = {
    "category": {
        "kurtas-kurtis": "a kurta or kurti, an Indian tunic", "sarees": "a saree, a draped Indian sari",
        "lehengas": "a lehenga choli, a long flared Indian skirt with blouse",
        "western-dresses": "a western dress or jumpsuit", "tops-shirts": "a women's top, t-shirt or shirt",
        "other-ethnic": "salwar, churidar or dupatta ethnic wear",
    },
    "pattern": {
        "solid": "a plain solid colour garment", "striped": "a striped garment", "checked": "a checked plaid garment",
        "floral": "a garment with a floral print", "polka dot": "a polka dot garment",
        "geometric": "a garment with a geometric print", "abstract": "a garment with an abstract print",
        "paisley": "a garment with a paisley print", "ethnic motifs": "a garment with traditional Indian block print motifs",
        "embroidered": "an embroidered garment", "textured": "a garment with a textured woven fabric",
    },
    "sleeve_type": {
        "sleeveless": "a sleeveless garment", "short sleeves": "a garment with short sleeves",
        "three-quarter": "a garment with three-quarter sleeves", "full sleeves": "a garment with long full sleeves",
    },
    "neckline": {
        "round neck": "a garment with a round neckline", "v-neck": "a garment with a v-neck",
        "mandarin collar": "a garment with a mandarin collar", "boat neck": "a garment with a boat neckline",
        "shirt collar": "a garment with a shirt collar", "square neck": "a garment with a square neckline",
    },
    "length": {
        "cropped": "a cropped short top", "hip length": "a hip length garment", "knee length": "a knee length garment",
        "ankle length": "an ankle length floor length garment",
    },
    "silhouette": {
        "straight": "a straight cut garment", "a-line": "an a-line flared garment", "anarkali": "an anarkali flared kurta",
        "fitted": "a fitted bodycon garment", "draped": "a draped garment", "flared": "a garment with a wide flared skirt",
    },
    "fabric_appearance": {
        "cotton": "a garment made of matte cotton fabric", "silk": "a garment made of shiny silk fabric",
        "georgette": "a garment made of sheer georgette or chiffon", "denim": "a garment made of denim",
        "velvet": "a garment made of velvet", "knit": "a garment made of knitted jersey fabric",
    },
    "embroidery": {
        "yes": "a garment with visible embroidery or zari work", "no": "a garment without embroidery",
    },
    "border": {
        "contrast border": "a garment with a wide contrasting decorative border", "no border": "a garment with no border",
    },
}
# A zero-shot value is kept only above these probabilities; otherwise the attribute is "unknown".
MIN_PROB = {"category": 0.45, "pattern": 0.40, "sleeve_type": 0.45, "neckline": 0.45, "length": 0.45,
            "silhouette": 0.40, "fabric_appearance": 0.45, "embroidery": 0.70, "border": 0.70}
STRUCTURAL_PATTERNS = {"solid", "striped", "checked", "polka dot"}


def merge_pattern(texture: dict, zero: dict) -> dict:
    """Combine the texture classifier (good at solid/stripes/checks/dots) with zero-shot CLIP
    (good at floral/paisley/embroidered...). Agreement raises confidence; disagreement is
    resolved in favour of the source that is strong for that kind of pattern."""
    tv, tc = texture.get("value", "unknown"), float(texture.get("confidence") or 0)
    zv, zc = zero.get("value", "unknown"), float(zero.get("confidence") or 0)
    if tv == "unknown" and zv == "unknown":
        return {"value": "unknown", "confidence": max(tc, zc), "source": "texture+zero_shot"}
    if tv == zv:
        return {"value": tv, "confidence": round(min(0.99, max(tc, zc) + 0.1), 3), "source": "texture+zero_shot"}
    if zv == "unknown":
        return texture
    if tv == "unknown":
        return zero
    if tv in STRUCTURAL_PATTERNS and tc >= 0.7 and zc < 0.6:
        return dict(texture, note=f"zero-shot suggested {zv}")
    return dict(zero, note=f"texture suggested {tv}")
_prompt_cache: dict[tuple, np.ndarray] = {}


def _prompt_matrix(enc: Encoder, attr: str) -> np.ndarray:
    key = (enc.key, enc.version, attr)
    if key not in _prompt_cache:
        texts = [f"a photo of {p}" for p in ZERO_SHOT[attr].values()]
        _prompt_cache[key] = enc.embed_texts(texts)
    return _prompt_cache[key]


def zero_shot(enc: Encoder, image_emb: np.ndarray) -> dict[str, dict]:
    out = {}
    for attr, options in ZERO_SHOT.items():
        logits = 100.0 * _prompt_matrix(enc, attr) @ image_emb   # CLIP's learned temperature is ~100
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        i = int(np.argmax(probs))
        value = list(options)[i] if probs[i] >= MIN_PROB[attr] else "unknown"
        out[attr] = {"value": value, "confidence": round(float(probs[i]), 3), "source": f"zero_shot:{enc.key}"}
    return out


def neighbour_vote(neighbour_categories: list[str], weights: list[float]) -> dict:
    if not neighbour_categories:
        return {"value": "unknown", "confidence": 0.0, "source": "neighbour_vote"}
    c = Counter()
    for cat, w in zip(neighbour_categories, weights):
        c[cat] += max(0.0, w)
    total = sum(c.values()) or 1.0
    cat, score = c.most_common(1)[0]
    conf = score / total
    return {"value": cat if conf >= 0.5 else "unknown", "confidence": round(conf, 3), "source": "neighbour_vote"}


def pixel_attributes(img: Image.Image) -> dict:
    """Attributes measured directly from pixels (no neural network)."""
    fg = estimate_foreground(img, version="v2")  # same mask version as the stored catalogue features
    palette = extract_palette(img, fg=fg)
    pvec, plabel = pattern_descriptor(img, fg=fg)
    pattern = classify_pattern(img, fg=fg)
    primary = palette[0] if palette else None
    secondary = palette[1] if len(palette) > 1 and palette[1]["weight"] >= 0.12 else None
    return {
        "palette": palette, "pattern_vector": pvec,
        "colour": {"value": primary["name"] if primary else "unknown",
                   "confidence": primary["weight"] if primary else 0.0, "source": "pixels"},
        "secondary_colour": {"value": secondary["name"] if secondary else "unknown",
                             "confidence": secondary["weight"] if secondary else 0.0, "source": "pixels"},
        "multicolour": {"value": "yes" if is_multicolour(palette) else "no", "confidence": None, "source": "pixels"},
        "texture": {"value": plabel, "confidence": None, "source": "pixels"},
        "pattern": {k: v for k, v in pattern.items() if k in ("value", "confidence", "source")},
    }


# ---------------------------------------------------------------- optional VLM (Qwen2.5-VL)
_vlm = None
VLM_PROMPT = ("Look at the clothing item in the image. Reply with ONLY a JSON object with keys: "
              "category, primary_colour, secondary_colour, pattern, sleeve_type, neckline, dress_style, "
              "fabric_appearance, embroidery, design_details. Use \"unknown\" when you cannot tell.")


def vlm_attributes(img: Image.Image) -> dict:
    """Structured attributes from Qwen2.5-VL. Only runs when ENABLE_VLM_ATTRIBUTES=true."""
    global _vlm
    s = get_settings()
    if not s.enable_vlm_attributes:
        return {}
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if _vlm is None:
            log.info("Loading VLM %s (large download, slow on CPU)", s.vlm_model)
            # bfloat16 halves memory (3B params: ~7 GB instead of ~12 GB in float32).
            dtype = torch.bfloat16 if s.ai_device == "cpu" else torch.float16
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    s.vlm_model, dtype=dtype, cache_dir=str(s.model_cache_path))
            except TypeError:  # older transformers versions call it torch_dtype
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    s.vlm_model, torch_dtype=dtype, cache_dir=str(s.model_cache_path))
            model = model.to(s.ai_device).eval()
            proc = AutoProcessor.from_pretrained(s.vlm_model, cache_dir=str(s.model_cache_path),
                                                 min_pixels=128 * 28 * 28, max_pixels=384 * 28 * 28)
            _vlm = (model, proc)
        model, proc = _vlm
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VLM_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(s.ai_device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        answer = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", answer, re.S)
        data = json.loads(m.group(0)) if m else {}
        return {k: {"value": str(v).lower() or "unknown", "confidence": None, "source": "vlm"} for k, v in data.items()}
    except Exception as e:  # noqa: BLE001 - VLM is optional; never break search
        log.warning("VLM attribute extraction failed: %s", e)
        return {}
