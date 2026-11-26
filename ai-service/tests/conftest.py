import io
import sys
from pathlib import Path

import pytest
from PIL import Image

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR / "scripts"))

from app.config import PROJECT_ROOT  # noqa: E402


def image_bytes(fmt="JPEG", size=(200, 300), colour=(200, 30, 40), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, colour if mode == "RGB" else colour + (0,)).save(buf, fmt)
    return buf.getvalue()


def _db_ready() -> bool:
    try:
        from app.config import get_settings
        from app.services.db import connection

        with connection() as conn:
            n = conn.execute("SELECT count(*) AS n FROM image_embeddings WHERE model_key=%s",
                             (get_settings().ai_primary_model,)).fetchone()["n"]
        return n > 0
    except Exception:  # noqa: BLE001
        return False


requires_db = pytest.mark.skipif(not _db_ready(), reason="needs PostgreSQL with catalogue embeddings")
CATALOGUE = PROJECT_ROOT / "dataset" / "metadata" / "catalogue.csv"
QUERIES = PROJECT_ROOT / "dataset" / "evaluation" / "queries.csv"
