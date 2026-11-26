from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.models.encoders import MODEL_SPECS

ALLOWED_FILTER_KEYS = {"category", "colour", "pattern", "sleeve_type", "neckline"}


class SearchFilters(BaseModel):
    category: list[str] | None = None
    colour: list[str] | None = None
    pattern: list[str] | None = None
    sleeve_type: list[str] | None = None
    neckline: list[str] | None = None
    min_price: float | None = Field(default=None, ge=0, le=10_000_000)
    max_price: float | None = Field(default=None, ge=0, le=10_000_000)

    @field_validator("category", "colour", "pattern", "sleeve_type", "neckline")
    @classmethod
    def _limit(cls, v):
        if v is not None:
            if len(v) > 20:
                raise ValueError("too many filter values")
            v = [str(x)[:60] for x in v if str(x).strip()]
        return v or None


class IndexRequest(BaseModel):
    product_ids: list[int] = Field(default_factory=list, max_length=5000)
    model: str | None = None
    force: bool = False

    @field_validator("model")
    @classmethod
    def _model(cls, v):
        if v is not None and v not in MODEL_SPECS:
            raise ValueError(f"unknown model; choose from {list(MODEL_SPECS)}")
        return v


class ErrorBody(BaseModel):
    error: str
    detail: str | None = None
