"""Central configuration loaded from the project-root .env file.

Every tunable value (DB credentials, model choice, ranking weights, feature flags)
comes from environment variables so nothing is hard-coded to one laptop.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ai-service/app/config.py -> project root is two levels above ai-service/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "stylefind"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    ai_service_host: str = "127.0.0.1"
    ai_service_port: int = 8001
    ai_primary_model: str = "fashion-clip"
    ai_device: str = "cpu"
    ai_batch_size: int = 16
    ai_max_loaded_models: int = 1
    ai_internal_token: str = "change_me_internal_token"
    hf_home: str = ".model-cache"

    colour_request_candidates: int = 300   # candidates examined when the shopper types a colour
    rank_w_visual: float = 0.65
    rank_w_colour: float = 0.15
    rank_w_pattern: float = 0.10
    rank_w_attribute: float = 0.10
    search_candidates: int = 50
    # hnsw.ef_search: 0 = automatic (max(100, 2 x candidates)); otherwise a fixed value 1-1000.
    hnsw_ef_search: int = 0
    # pgvector >= 0.8: keep scanning the HNSW graph until enough rows pass the filters.
    hnsw_iterative_scan: bool = True
    # If a filtered approximate search still returns fewer rows than requested, re-run it exactly.
    filtered_exact_fallback: bool = True
    # Query multi-crop: off | max_score | mean_embedding (see docs/EVALUATION.md before enabling)
    search_multicrop: str = "off"
    # Hugging Face revision (commit hash / tag) per model; recorded in model_version when set.
    fashion_clip_revision: str = ""
    dinov2_revision: str = ""

    # Garment isolation for searches with crop=true:
    #   auto = segmentation pipeline with the models enabled below (default), foreground = pipeline with
    #   weight-free methods only, yolo = pipeline + YOLO garment detector, off = full image
    garment_isolation: str = "auto"
    isolation_padding: float = 0.06          # keep a margin so sleeves / hems / borders survive the crop
    isolation_default: str = "all_garments"  # crop scope when no target is given: all_garments | main_garment
    detection_min_confidence: float = 0.35
    enable_garment_detection: bool = False
    # ---- garment segmentation pipeline (used by crop=true and /ai/segment) ----
    # auto = choose the cheapest adequate path per photo; light = weight-free methods only; off = full image
    segmentation_strategy: str = "auto"
    garment_detector: str = "none"            # none | yolo   (clothing weights in YOLO_WEIGHTS)
    yolo_imgsz: int = 640
    detection_min_box_area: float = 0.02      # share of the image; smaller boxes are ignored
    detection_max_instances: int = 10
    person_segmenter: str = "none"            # none | yolo   (COCO yolo*-seg 'person' masks, PERSON_YOLO_WEIGHTS)
    person_yolo_weights: str = "yolo11n-seg.pt"
    human_parser: str = "none"                # none | segformer
    human_parser_model: str = "mattmdjaga/segformer_b2_clothes"
    human_parser_size: int = 512
    pose_estimator: str = "none"              # none | mediapipe
    pose_model_path: str = ""
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    face_cascade_path: str = ""
    segmentation_timeout_s: float = 20.0      # per optional model call; on timeout the next fallback is used
    min_segmentation_confidence: float = 0.4  # below this the pipeline falls back (never returns a silent bad mask)
    boundary_refinement: str = "guided"       # guided | none
    segmentation_cache_size: int = 64
    embedding_cache_size: int = 256
    yolo_weights: str = "yolo11n.pt"
    enable_sam2: bool = False
    sam2_checkpoint: str = ""
    enable_vlm_attributes: bool = False
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    products_dir: str = "dataset/products"
    max_upload_mb: int = 8

    @property
    def database_url(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} dbname={self.postgres_db} "
            f"user={self.postgres_user} password={self.postgres_password}"
        )

    @property
    def products_path(self) -> Path:
        p = Path(self.products_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def model_cache_path(self) -> Path:
        p = Path(self.hf_home)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
