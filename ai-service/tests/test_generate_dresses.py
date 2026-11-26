"""Tests for scripts/generate_dress_images.py.

Stable Diffusion itself cannot run in CI (no GPU, no model download), so a stand-in pipeline draws simple
garment pictures. That checks everything around the model: the plan, prompt limits, resuming, quality-check
handling and that the output is accepted by the importer. Image realism is NOT tested here.
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

import generate_dress_images as G
import make_segmentation_fixture as F

AI_DIR = Path(__file__).resolve().parents[1]


class FakePipe:
    """Mimics diffusers' text-to-image call signature; returns a drawn garment on white."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, num_inference_steps, guidance_scale, height, width, generator, negative_prompt=None):
        self.calls += 1
        img, _, _ = F.scene("product_only", random.Random(self.calls))
        return type("Out", (), {"images": [img.resize((width, height))]})()


@pytest.fixture
def fake_models(monkeypatch):
    pipe = FakePipe()
    monkeypatch.setattr(G, "load_pipeline", lambda preset, device: pipe)
    monkeypatch.setitem(sys.modules, "torch", type(sys)("torch"))  # generator only needs torch for seeds/cuda check
    sys.modules["torch"].cuda = type("cuda", (), {"is_available": staticmethod(lambda: False)})
    sys.modules["torch"].Generator = lambda device="cpu": type("Gen", (), {"manual_seed": lambda self, s: self})()
    return pipe


def test_plan_is_deterministic_balanced_and_unique():
    a, b = G.build_plan(540, 42), G.build_plan(540, 42)
    assert a == b
    assert len({p["image"] for p in a}) == 540
    designs = {tuple(p[k] for k in ("dress_type", "colour", "pattern", "sleeve_type", "neckline", "length", "fabric")) for p in a}
    assert len(designs) == 540
    for key, vocab in (("colour", G.COLOURS), ("pattern", G.PATTERNS), ("dress_type", G.DRESS_TYPES)):
        c = Counter(p[key] for p in a)
        assert set(c) == set(vocab)
        assert max(c.values()) - min(c.values()) <= max(12, 0.3 * max(c.values()))
    assert G.build_plan(540, 7) != a


def test_no_clashing_designs_and_vocabulary_matches_project():
    plan = G.build_plan(540)
    assert all(G._ok({k: p[k] for k in ("dress_type", "pattern", "sleeve_type", "neckline", "fabric")}) for p in plan)
    assert all(p["fabric"] == "wool knit" for p in plan if p["dress_type"] == "sweater")
    from app.services.attributes import ZERO_SHOT
    from app.services.colour import COLOUR_REFERENCE

    assert set(G.PATTERNS) <= set(ZERO_SHOT["pattern"])
    assert set(G.SLEEVES) <= set(ZERO_SHOT["sleeve_type"])
    assert set(G.COLOURS) <= set(COLOUR_REFERENCE)


def test_prompts_fit_in_77_clip_tokens_and_ask_for_no_people():
    open_clip = pytest.importorskip("open_clip")
    tok = open_clip.get_tokenizer("ViT-B-32")
    for p in G.build_plan(540):
        n = int((tok([p["prompt"]])[0] != 0).sum())
        assert n <= 77, (n, p["prompt"])
        assert "background" in p["prompt"] or "wall" in p["prompt"]
        for word in ("woman", "model wearing", "girl"):
            assert word not in p["prompt"]
        assert " a a" not in p["prompt"] and "with sleeveless" not in p["prompt"]
    assert "person" in G.NEGATIVE and "face" in G.NEGATIVE


def test_generation_writes_importer_layout_and_resumes(tmp_path, fake_models):
    out = tmp_path / "gen"
    G.main(["--out", str(out), "--count", "12", "--device", "cpu"])
    rows = list(csv.DictReader(open(out / "images.csv", encoding="utf-8")))
    assert len(rows) == 12 and fake_models.calls == 12
    assert {"image", "sender_id", "label", "kids", "prompt", "colour", "pattern"} <= set(rows[0])
    assert all(r["label"] == "Dress" for r in rows)
    assert all((out / "images_original" / f"{r['image']}.jpg").exists() for r in rows)
    # simulate a disconnect: one image lost -> only that one is regenerated, no duplicate rows
    (out / "images_original" / "dress-0005.jpg").unlink()
    G.main(["--out", str(out), "--count", "12", "--device", "cpu"])
    rows = list(csv.DictReader(open(out / "images.csv", encoding="utf-8")))
    assert fake_models.calls == 13 and len(rows) == 12 and len({r["image"] for r in rows}) == 12
    # nothing left to do
    G.main(["--out", str(out), "--count", "12", "--device", "cpu"])
    assert fake_models.calls == 13


def test_quality_check_retries_then_records_rejects(tmp_path, fake_models, monkeypatch):
    verdicts = iter([(False, "looks like a person"), (True, "ok")] + [(True, "ok")] * 3 +
                    [(False, "text"), (False, "text"), (False, "text")] + [(True, "ok")] * 10)
    monkeypatch.setattr(G, "Checker", lambda device, *a: (lambda img, presentation: next(verdicts)))
    out = tmp_path / "gen"
    G.main(["--out", str(out), "--count", "6", "--device", "cpu", "--qc", "--retries", "2"])
    rows = list(csv.DictReader(open(out / "images.csv", encoding="utf-8")))
    rejected = json.loads((out / "rejected.json").read_text())
    assert len(rows) == 5 and len(rejected) == 1 and rejected[0]["reason"] == "text"
    assert fake_models.calls == 2 + 3 + 3 + 1   # retry after a reject, give up after retries


def test_importer_accepts_generated_folder(tmp_path, fake_models):
    out = tmp_path / "gen"
    G.main(["--out", str(out), "--count", "16", "--device", "cpu"])
    rep_dir = tmp_path / "rep"
    r = subprocess.run([sys.executable, "scripts/import_clothing_dataset.py", "--raw", str(out), "--count", "10",
                        "--pattern-model", "none", "--dry-run", "--report-dir", str(rep_dir), "--code-prefix", "ai",
                        "--source", "AI-generated dresses (StyleFind demo)"],
                       cwd=AI_DIR, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((rep_dir / "report.json").read_text())
    assert rep["source"] == "AI-generated dresses (StyleFind demo)"
    assert rep["per_category"] == {"western-dresses": rep["selected"]} and rep["selected"] >= 8
    assert rep["generated_colour_agreement"] is not None


def test_generated_names_use_prompt_but_trust_measured_colour():
    import import_clothing_dataset as I

    meta = {"colour": "navy blue", "pattern": "polka dot", "length": "midi", "dress_type": "wrap"}
    assert I.generated_name({"meta": meta, "colour": "navy blue"}) == "Navy blue polka dot midi wrap dress"
    assert I.generated_name({"meta": meta, "colour": "yellow"}).startswith("Yellow ")   # image disagrees with prompt
    meta = {"colour": "pink", "pattern": "solid", "length": "ankle length", "dress_type": "maxi"}
    assert I.generated_name({"meta": meta, "colour": "pink"}) == "Pink maxi dress"


def test_background_rule_accepts_studio_walls_rejects_dark_and_clutter():
    """Checker's background rule (CLIP part replaced by a fixed 'dress' verdict)."""
    import numpy as np

    class FakeProbs:
        def __init__(self, p):
            self.p = p

    chk = G.Checker.__new__(G.Checker)
    chk.strict_white, chk.no_people = False, False

    def verdict(img, presentation="ghost mannequin"):
        # call only the pixel part: emulate CLIP saying "dress" with 0.9
        import types
        chk.torch = types.SimpleNamespace(no_grad=lambda: __import__("contextlib").nullcontext())
        chk.proc = lambda **k: types.SimpleNamespace(to=lambda d: {})
        chk.device = "cpu"
        chk.model = lambda **k: types.SimpleNamespace(logits_per_image=types.SimpleNamespace(
            softmax=lambda d: [types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: np.array([.9, .04, .03, .02, .01])))]))
        return chk(img, presentation)

    beige = Image.new("RGB", (512, 512), (205, 190, 170))              # studio wall like the accepted hanger shots
    dark = Image.new("RGB", (512, 512), (40, 40, 45))
    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 255, (512, 512), dtype=np.uint8)).convert("RGB")
    assert verdict(beige)[0] is True
    assert verdict(dark)[0] is False and verdict(noisy)[0] is False
    chk.strict_white = True
    assert verdict(beige)[0] is False                                   # the old v1 rule
