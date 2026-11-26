"""Generate e-commerce style dress product photos with Stable Diffusion (for a StyleFind demo catalogue).

Run it where there is an NVIDIA GPU - Google Colab's free T4 is enough (notebooks/generate_dresses_colab.ipynb
does everything for you). It also runs on a CPU, but slowly (see docs/GENERATED_DRESSES.md).

Output = the same layout as the Clothing Dataset, so the existing importer takes it unchanged:

    <out>/images.csv          image,sender_id,label,kids,dress_type,colour,pattern,sleeve_type,neckline,length,
                              fabric,presentation,prompt,seed,model
    <out>/images_original/    dress-0000.jpg ...

Every prompt is built from a balanced plan (dress type x colour x pattern x sleeves x neckline x length x fabric x
presentation), so the set covers many different designs. No people are generated (garment only: ghost mannequin,
flat lay, hanger, headless dress form). The script is RESUMABLE: re-run the same command after a Colab disconnect
and it continues where it stopped.

    python generate_dress_images.py --out generated-dresses --count 540 --preset fast --qc
    python generate_dress_images.py --out generated-dresses --plan-only          # just print / save the prompts

Presets
  fast     stabilityai/sdxl-turbo, 512x512, 4 steps   (~1 s per image on a T4; Stability AI Community License)
  quality  stabilityai/stable-diffusion-xl-base-1.0, 1024x1024, 25 steps + negative prompt
           (~25 s per image on a T4; CreativeML Open RAIL++-M)

These are AI-GENERATED images. The importer records them as such (source + description); do not present them as
photos of real products.

This file is standalone (no StyleFind imports) so it can be copied to Colab on its own.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------- design vocabulary
# values on the left are StyleFind's own attribute vocabulary; right = words used in the prompt
DRESS_TYPES = {
    "a-line": "A-line dress", "wrap": "wrap dress", "shift": "shift dress", "sheath": "sheath dress",
    "fit-and-flare": "fit and flare dress", "maxi": "flowing maxi dress", "shirt dress": "button-down shirt dress",
    "slip": "slip dress", "tiered": "tiered ruffle dress", "bodycon": "bodycon dress", "kaftan": "kaftan dress",
    "anarkali": "anarkali gown with a flared skirt", "smocked": "smocked sundress", "peplum": "peplum dress",
    "sweater": "knitted sweater dress", "skater": "skater dress",
}
COLOURS = {  # names from app/services/colour.py so the importer's colour check can compare them
    "black": "black", "white": "white", "off white": "off-white", "beige": "beige", "brown": "brown",
    "maroon": "maroon", "burgundy": "burgundy", "red": "red", "coral": "coral", "pink": "pastel pink",
    "rose": "rose pink", "peach": "peach", "orange": "orange", "mustard": "mustard yellow", "yellow": "yellow",
    "olive": "olive green", "green": "green", "bottle green": "bottle green", "mint": "mint green", "teal": "teal",
    "sky blue": "sky blue", "blue": "blue", "royal blue": "royal blue", "navy blue": "navy blue",
    "lavender": "lavender", "purple": "purple", "magenta": "magenta", "grey": "grey", "mauve": "mauve",
}
PATTERNS = {  # {c} = colour, {k} = contrast colour for two-colour patterns
    "solid": "plain {c}", "striped": "{c} and {k} striped", "checked": "{c} and {k} gingham check",
    "floral": "{c} floral print", "polka dot": "{c} polka dot", "geometric": "{c} geometric print",
    "abstract": "{c} abstract print", "paisley": "{c} paisley print", "ethnic motifs": "{c} block print",
    "embroidered": "{c} embroidered", "textured": "{c} textured",
}
SLEEVES = {"sleeveless": "sleeveless", "short sleeves": "short-sleeved", "three-quarter": "three-quarter sleeve",
           "full sleeves": "long-sleeved"}
NECKLINES = {"round neck": "round neck", "v-neck": "v-neck", "square neck": "square neck",
             "boat neck": "boat neck", "shirt collar": "shirt collar", "mandarin collar": "mandarin collar"}
LENGTHS = {"knee length": "knee-length", "midi": "midi", "ankle length": "ankle-length"}
LIGHT = {"white", "off white", "beige", "peach", "pink", "mint", "sky blue", "lavender", "yellow"}
FABRICS = ["cotton", "linen", "rayon", "georgette", "chiffon", "crepe", "satin", "denim", "jersey knit", "velvet"]
PRESENTATIONS = {  # weight, template. Important words first: SDXL reads only the first 77 CLIP tokens.
    "ghost mannequin": (0.50, "e-commerce product photo of {d}, invisible ghost mannequin, front view, full dress visible, "
                              "pure white background, studio lighting"),
    "flat lay": (0.25, "e-commerce flat lay photo of {d}, laid flat, top-down view, full dress visible, "
                       "pure white background"),
    "hanger": (0.15, "product photo of {d} on a wooden hanger, plain light grey wall, front view, full dress visible, "
                     "soft daylight"),
    "dress form": (0.10, "product photo of {d} on a headless white dress form, plain white background, front view, "
                         "full dress visible"),
}
QUALITY_SUFFIX = ", photorealistic, sharp fabric detail"
NEGATIVE = ("person, woman, man, model, face, head, hands, arms, legs, skin, text, watermark, logo, label, price tag, "
            "blurry, lowres, deformed, cropped, cut off, multiple dresses, collage, cartoon, illustration, 3d render")
PRESETS = {
    "fast": {"model": "stabilityai/sdxl-turbo", "size": 512, "steps": 4, "guidance": 0.0, "negative": False},
    "quality": {"model": "stabilityai/stable-diffusion-xl-base-1.0", "size": 1024, "steps": 25, "guidance": 6.0,
                "negative": True},
}
FIELDS = ["image", "sender_id", "label", "kids", "dress_type", "colour", "pattern", "sleeve_type", "neckline",
          "length", "fabric", "presentation", "prompt", "seed", "model"]
# combinations that look wrong or that SD draws badly
INCOMPATIBLE = {("sweater", "satin"), ("sweater", "chiffon"), ("sweater", "georgette"), ("slip", "full sleeves"),
                ("slip", "shirt collar"), ("slip", "denim"), ("shirt dress", "boat neck"), ("bodycon", "chiffon"),
                ("kaftan", "bodycon"), ("sweater", "denim"), ("velvet", "striped"), ("velvet", "checked")}


def _balanced(values: list, n: int, rng: random.Random) -> list:
    """n items using every value equally often (shuffled in rounds) - avoids clumps of one colour."""
    out = []
    while len(out) < n:
        v = list(values)
        rng.shuffle(v)
        out += v
    return out[:n]


def _weighted(weights: dict, n: int, rng: random.Random) -> list:
    out = []
    for k, (w, _) in weights.items():
        out += [k] * round(w * n)
    out = (out + list(weights) * n)[:n]
    rng.shuffle(out)
    return out


def _ok(spec: dict) -> bool:
    vals = set(spec.values())
    return not any(a in vals and b in vals for a, b in INCOMPATIBLE)


def build_plan(count: int, seed: int = 42) -> list[dict]:
    """Deterministic list of `count` design specs + prompts."""
    rng = random.Random(seed)
    cols = {"dress_type": _balanced(list(DRESS_TYPES), count, rng), "colour": _balanced(list(COLOURS), count, rng),
            "pattern": _balanced(list(PATTERNS), count, rng), "sleeve_type": _balanced(list(SLEEVES), count, rng),
            "neckline": _balanced(list(NECKLINES), count, rng), "length": _balanced(list(LENGTHS), count, rng),
            "fabric": _balanced(FABRICS, count, rng), "presentation": _weighted(PRESENTATIONS, count, rng)}
    plan = []
    for i in range(count):
        spec = {k: v[i] for k, v in cols.items()}
        tries = 0
        while not _ok(spec) and tries < 50:  # swap the clashing attribute with another row's value
            k = rng.choice(["fabric", "sleeve_type", "neckline", "pattern"])
            spec[k] = rng.choice(list(SLEEVES) if k == "sleeve_type" else list(NECKLINES) if k == "neckline"
                                 else list(PATTERNS) if k == "pattern" else FABRICS)
            tries += 1
        if spec["dress_type"] == "maxi":
            spec["length"] = "ankle length"
        if spec["dress_type"] == "shirt dress":
            spec["neckline"] = "shirt collar"
        if spec["dress_type"] == "sweater":
            spec["fabric"] = "wool knit"
        plan.append({**spec, "seed": seed * 100_000 + i, "image": f"dress-{i:04d}"})
    for p in plan:
        p["prompt"] = make_prompt(p)
    return plan


def make_prompt(spec: dict) -> str:
    colour = COLOURS[spec["colour"]]
    contrast = "navy blue" if spec["colour"] in LIGHT else "white"
    look = PATTERNS[spec["pattern"]].format(c=colour, k=contrast)
    words = f"{SLEEVES[spec['sleeve_type']]} {look} {spec['fabric']} {LENGTHS[spec['length']]} {DRESS_TYPES[spec['dress_type']]}"
    words = words.replace("ankle-length flowing maxi", "flowing maxi")
    article = "an" if words[0].lower() in "aeiou" else "a"
    d = f"{article} {words} with a {NECKLINES[spec['neckline']]}"
    return PRESENTATIONS[spec["presentation"]][1].format(d=d) + QUALITY_SUFFIX


# --------------------------------------------------------------------- generation
def load_pipeline(preset: dict, device: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if device == "cuda" else torch.float32
    kw = {"torch_dtype": dtype}
    if device == "cuda":
        kw["variant"] = "fp16"
    pipe = AutoPipelineForText2Image.from_pretrained(preset["model"], **kw).to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        try:
            pipe.enable_vae_slicing()
        except Exception:  # noqa: BLE001
            pass
    return pipe


def generate_one(pipe, spec: dict, preset: dict, device: str, attempt: int = 0):
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(spec["seed"]) + 10_000 * attempt)
    kw = {"prompt": spec["prompt"], "num_inference_steps": preset["steps"], "guidance_scale": preset["guidance"],
          "height": preset["size"], "width": preset["size"], "generator": g}
    if preset["negative"]:
        kw["negative_prompt"] = NEGATIVE
    return pipe(**kw).images[0]


class Checker:
    """Quality control with CLIP ViT-B/32 (zero-shot): keep images that look like a dress product photo, reject
    people, other garments, text and broken outputs; plus a plain-background check for white-background shots."""

    LABELS = ["a product photo of a dress", "a photo of a person or model wearing clothes",
              "a photo of a shirt, trousers or a skirt", "text, a logo or a label", "a blurry, broken or empty image"]

    def __init__(self, device: str, strict_white: bool = False, no_people: bool = False):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch, self.device = torch, device
        self.strict_white, self.no_people = strict_white, no_people
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, img, presentation: str) -> tuple[bool, str]:
        import numpy as np

        with self.torch.no_grad():
            inp = self.proc(text=self.LABELS, images=img, return_tensors="pt", padding=True).to(self.device)
            probs = self.model(**inp).logits_per_image.softmax(-1)[0].cpu().numpy()
        best = int(probs.argmax())
        if self.no_people and probs[1] >= 0.3:
            return False, f"shows a person ({probs[1]:.2f})"
        # a dress worn by a (generated) model is still a dress photo unless --no-people is set
        dress_like = probs[0] + (0 if self.no_people else probs[1])
        if best not in (0, 1) or dress_like < 0.5:
            return False, f"looks like '{self.LABELS[best]}' ({probs[best]:.2f})"
        a = np.asarray(img.convert("L"), dtype=np.float32)
        border = np.concatenate([a[:8].ravel(), a[-8:].ravel(), a[:, :8].ravel(), a[:, -8:].ravel()])
        if self.strict_white and presentation in ("ghost mannequin", "flat lay", "dress form") \
                and (border.mean() < 215 or border.std() > 25):
            return False, f"background not plain white (mean {border.mean():.0f}, std {border.std():.0f})"
        # Default: only dark or cluttered backgrounds are rejected. v1 demanded pure white and threw away 60 %
        # of good SDXL-Turbo images (it likes off-white / beige studio walls).
        if border.mean() < 100 or border.std() > 55:
            return False, f"background too dark or cluttered (mean {border.mean():.0f}, std {border.std():.0f})"
        return True, f"dress {probs[0]:.2f}"


def read_done(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r["image"]: r for r in csv.DictReader(f)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=540, help="images to generate (8 %% spare so 500 survive the checks)")
    ap.add_argument("--preset", choices=list(PRESETS), default="fast")
    ap.add_argument("--model", default=None, help="override the preset's model id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qc", action="store_true", help="CLIP quality check; regenerate rejects (up to --retries times)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--strict-white", action="store_true", help="v1 rule: white-background shots must be pure white")
    ap.add_argument("--no-people", action="store_true", help="also reject images that show a (generated) person")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--plan-only", action="store_true", help="write plan.json with the prompts and stop")
    args = ap.parse_args(argv)

    preset = dict(PRESETS[args.preset])
    if args.model:
        preset["model"] = args.model
    out, img_dir = args.out, args.out / "images_original"
    img_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.count, args.seed)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    if args.plan_only:
        for p in plan[:5]:
            print(p["image"], "|", p["prompt"])
        print(f"... {len(plan)} prompts written to {out / 'plan.json'}")
        return

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: no GPU - this will be slow (fast preset: roughly 20-60 s per image on a laptop CPU).")
    csv_path = out / "images.csv"
    done = read_done(csv_path)
    missing = [k for k in done if not (img_dir / f"{k}.jpg").exists()]
    if missing:  # a row whose image never got written (e.g. disconnect): drop it, it will be regenerated
        done = {k: v for k, v in done.items() if k not in missing}
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(done.values())
    todo = [p for p in plan if p["image"] not in done or not (img_dir / f"{p['image']}.jpg").exists()]
    print(f"{len(done)} already done, {len(todo)} to generate with {preset['model']} on {device}")
    if not todo:
        return
    pipe = load_pipeline(preset, device)
    checker = Checker(device, args.strict_white, args.no_people) if args.qc else None
    rejects = []
    new_file = not csv_path.exists()
    t0 = time.time()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for k, spec in enumerate(todo):
            img, verdict = None, "not checked"
            for attempt in range(args.retries + 1 if checker else 1):
                img = generate_one(pipe, spec, preset, device, attempt)
                if checker is None:
                    break
                ok, verdict = checker(img, spec["presentation"])
                if ok:
                    break
                img = None
            if img is None:
                rejects.append({"image": spec["image"], "reason": verdict, "prompt": spec["prompt"]})
                continue
            img.convert("RGB").save(img_dir / f"{spec['image']}.jpg", quality=93)
            w.writerow({**spec, "sender_id": "generated", "label": "Dress", "kids": "False", "model": preset["model"]})
            f.flush()
            if (k + 1) % 10 == 0 or k == len(todo) - 1:
                rate = (time.time() - t0) / (k + 1)
                print(f"  {k + 1}/{len(todo)}  {rate:.1f} s/image  ~{rate * (len(todo) - k - 1) / 60:.0f} min left",
                      flush=True)
    (out / "rejected.json").write_text(json.dumps(rejects, indent=1))
    total = len(read_done(csv_path))
    print(f"done: {total} images in {out} ({len(rejects)} rejected by the quality check, see rejected.json)")


if __name__ == "__main__":
    sys.exit(main())
