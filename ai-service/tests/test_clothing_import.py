"""Tests for scripts/import_clothing_dataset.py (the Clothing Dataset importer).

Uses small generated images in the dataset's own layout (images.csv + images_original + images_compressed),
including the problems a real download can have: a missing file, a corrupt file, a duplicate photo,
a tiny image, labels that must be excluded. Only --dry-run is exercised here (no files copied, DB untouched).
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import import_clothing_dataset as I
import make_segmentation_fixture as F

AI_DIR = Path(__file__).resolve().parents[1]


def test_quotas_sum_and_respect_availability():
    avail = {"T-Shirt": 1011, "Pants": 692, "Dress": 357, "Top": 43, "Blouse": 23}
    q = I.quotas(avail, 500, 10)
    assert sum(q.values()) == 500
    assert all(q[k] <= avail[k] for k in avail)
    assert q["T-Shirt"] > q["Dress"] > q["Blouse"] >= 10          # sqrt-proportional, rare types still present
    small = I.quotas({"A": 3, "B": 5}, 500, 10)
    assert small == {"A": 3, "B": 5}                                 # cannot invent images


def test_pick_diverse_covers_every_pattern_group():
    rng = random.Random(0)
    cands = []
    for pat, n in (("solid", 40), ("striped", 3), ("checked", 2), ("polka dot", 1)):
        for i in range(n):
            v = np.random.default_rng(len(cands)).random(10).astype(np.float32)
            cands.append({"pattern": pat, "vec": v, "seg_fallback": False, "id": f"{pat}{i}"})
    picked = I.pick_diverse(cands, 8, rng)
    assert len(picked) == 8
    assert {c["pattern"] for c in picked} == {"solid", "striped", "checked", "polka dot"}


def test_pick_diverse_prefers_different_colours():
    rng = random.Random(0)
    same = [{"pattern": "solid", "vec": np.zeros(6, np.float32) + 0.001 * i, "seg_fallback": False} for i in range(10)]
    odd = {"pattern": "solid", "vec": np.ones(6, np.float32), "seg_fallback": False}
    picked = I.pick_diverse(same + [odd], 2, rng)
    assert any(c is odd for c in picked)


def test_layout_detection_handles_windows_folder_names(tmp_path):
    root = tmp_path / "archive (2)"
    (root / "images_original").mkdir(parents=True)
    (root / "images_compressed").mkdir()
    (root / "images.csv").write_text("image,sender_id,label,kids\n")
    csv_path, orig, comp = I.find_layout(tmp_path)  # images.csv found one level down
    assert csv_path == root / "images.csv" and orig.name == "images_original" and comp.name == "images_compressed"


def _fixture(root: Path) -> dict:
    """~40 images in the real layout; returns expectations."""
    (root / "images_original").mkdir(parents=True)
    (root / "images_compressed").mkdir()
    rng = random.Random(5)
    rows, k = [], 0
    labels = ["T-Shirt", "Dress", "Pants", "Skirt"]
    for cat in ("product_only", "patterned", "white_on_white", "gradient_bg", "shadow"):
        for j in range(8):
            img, _, _ = F.scene(cat, rng)
            iid = f"img-{k:03d}"
            big = img.resize((img.width * 2, img.height * 2))
            big.save(root / "images_original" / f"{iid}.jpg", quality=92)
            img.save(root / "images_compressed" / f"{iid}.jpg", quality=85)
            rows.append([iid, str(k % 7), labels[k % 4], "False"])
            k += 1
    # problems a real download can have
    (root / "images_original" / "img-000.jpg").write_bytes(b"not a jpeg")     # corrupt original
    (root / "images_compressed" / "img-000.jpg").unlink()
    rows.append(["missing-file", "1", "Dress", "False"])                       # listed, no file
    Image.open(root / "images_original" / "img-001.jpg").save(root / "images_original" / "dup.jpg")
    Image.open(root / "images_compressed" / "img-001.jpg").save(root / "images_compressed" / "dup.jpg")
    rows.append(["dup", "1", rows[1][2], "False"])                                # same photo, new id
    Image.new("RGB", (120, 90), "red").save(root / "images_original" / "tiny.jpg")
    rows.append(["tiny", "2", "T-Shirt", "False"])                               # too small
    for lab in ("Not sure", "Skip", "Other", "Shoes", "Hat"):
        Image.new("RGB", (400, 500), "blue").save(root / "images_original" / f"x-{lab}.jpg")
        rows.append([f"x-{lab}", "3", lab, "False"])
    with open(root / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "sender_id", "label", "kids"])
        w.writerows(rows)
    return {"valid": 39}


def test_dry_run_end_to_end(tmp_path):
    raw = tmp_path / "clothing-dataset"
    exp = _fixture(raw)
    out = tmp_path / "report"
    r = subprocess.run([sys.executable, "scripts/import_clothing_dataset.py", "--raw", str(raw), "--count", "20",
                        "--min-per-type", "2", "--pattern-model", "none", "--dry-run", "--report-dir", str(out)],
                       cwd=AI_DIR, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((out / "report.json").read_text())
    assert rep["selected"] == 20
    assert rep["rejected"]["image file missing"] == 1
    assert rep["rejected"]["corrupt / unreadable"] == 1
    assert rep["rejected"].get("near-duplicate photo", 0) + rep["rejected"].get("byte-identical duplicate", 0) >= 1
    assert rep["rejected"][f"too small (< {I.MIN_SIDE} px)"] == 1
    assert rep["excluded"]["label Not sure"] == 1 and rep["excluded"]["label Shoes (opt-in)"] == 1
    assert set(rep["per_type"]) == {"T-Shirt", "Dress", "Pants", "Skirt"}
    assert (out / "preview.jpg").exists() and (out / "report.md").exists()
    assert "dry run" in r.stdout
    assert exp["valid"] >= rep["selected"]


def test_product_names_are_honest():
    c = {"colour": "navy blue", "pattern": "striped", "label": "T-Shirt", "kids": True}
    assert I.product_name(c) == "Navy blue striped T-shirt (kids)"
    c = {"colour": "unknown", "pattern": "unknown", "label": "Outwear", "kids": False}
    assert I.product_name(c) == "Jacket"
