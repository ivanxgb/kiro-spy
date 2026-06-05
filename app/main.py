"""P1 — FastAPI service for pixels-only image geolocation.

Endpoints:
  GET  /health  -> backend status
  POST /locate  -> top-K {lat, lon, confidence} for an uploaded image (EXIF stripped)
  GET  /        -> Leaflet web UI (static)

NOTE: no authentication. Fine for local/internal use; add auth + rate limiting
before exposing this on a network (tracked for the SaaS/prod phase).
"""
from __future__ import annotations
import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

from app.geo.backends import get_backend

app = FastAPI(title="kiro-spy — Image Geolocation Engine", version="0.1.0")
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/health")
def health():
    b = get_backend()
    return {"status": "ok", "backend": b.name, "real_model": b.real}


@app.post("/locate")
async def locate(file: UploadFile = File(...), top_k: int = 5):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="upload an image file")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    # Strip metadata (prove pixels-only): re-encode pixels without EXIF/GPS.
    from PIL import Image, UnidentifiedImageError
    try:
        buf = io.BytesIO()
        Image.open(io.BytesIO(raw)).convert("RGB").save(buf, format="JPEG")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="not a readable image")
    backend = get_backend()
    preds = backend.predict(buf.getvalue(), top_k=max(1, min(top_k, 10)))
    return {
        "backend": backend.name,
        "real_model": backend.real,
        "note": None if backend.real
        else "STUB backend: placeholder predictions, not real geolocation. Install requirements-ml.txt + set GEO_BACKEND=geoclip for the real model.",
        "predictions": preds,
    }


# Serve the UI at "/" (declared last so /health and /locate take precedence).
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
