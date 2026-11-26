"""StyleFind AI - Python AI service (FastAPI).

Run (PowerShell, inside ai-service with the venv active):
    uvicorn app.main:app --host 127.0.0.1 --port 8001
Docs: http://127.0.0.1:8001/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.models.encoders import ModelUnavailableError
from app.services.db import close_pool
from app.utils.logging import get_logger, setup_logging

setup_logging()
log = get_logger("main")
os.environ.setdefault("HF_HOME", str(get_settings().model_cache_path))


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    log.info("AI service starting (primary model=%s, device=%s)", s.ai_primary_model, s.ai_device)
    yield
    close_pool()


app = FastAPI(title="StyleFind AI - AI service", version="1.0.0", lifespan=lifespan)
# Only the Node backend should call this service; allow its origin for local debugging tools.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4000", "http://127.0.0.1:4000"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    fields = [".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid request", "fields": fields})


@app.exception_handler(ModelUnavailableError)
async def model_unavailable(_: Request, exc: ModelUnavailableError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    # Log full details server-side; never leak paths / stack traces to clients.
    log.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "internal error in AI service"})
