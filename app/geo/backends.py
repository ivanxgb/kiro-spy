"""P1 — Pluggable geolocation backends.

A backend implements ``predict(image_bytes, top_k) -> list[{lat, lon, confidence}]``.

- ``StubBackend``: deterministic, ML-free placeholder so the whole service (API +
  UI + eval wiring) is runnable and testable WITHOUT a GPU or multi-GB downloads.
  It does NOT perform real geolocation — it maps an image's average color +
  content hash to entries in a small city gallery. Development/integration only.
- ``GeoCLIPBackend``: the real image->GPS model. Heavy deps are lazily imported
  (``pip install -r requirements-ml.txt``) so the app boots without torch.

Select via env ``GEO_BACKEND`` = auto (default) | stub | geoclip.
"""
from __future__ import annotations
import io
import os
import math
import hashlib
import tempfile
from abc import ABC, abstractmethod
from typing import List, Dict

Prediction = Dict[str, float]

# Small reference gallery of world cities (name, lat, lon) for the stub backend.
_CITY_GALLERY = [
    ("Paris", 48.8566, 2.3522), ("London", 51.5074, -0.1278),
    ("New York", 40.7128, -74.0060), ("Tokyo", 35.6762, 139.6503),
    ("Sydney", -33.8688, 151.2093), ("Cairo", 30.0444, 31.2357),
    ("Rio de Janeiro", -22.9068, -43.1729), ("Moscow", 55.7558, 37.6173),
    ("Cape Town", -33.9249, 18.4241), ("Mumbai", 19.0760, 72.8777),
    ("Mexico City", 19.4326, -99.1332), ("Reykjavik", 64.1466, -21.9426),
]


class Geolocator(ABC):
    name: str = "base"
    real: bool = False

    @abstractmethod
    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[Prediction]:
        ...


class StubBackend(Geolocator):
    """Deterministic, ML-free placeholder. NOT real geolocation."""
    name = "stub"
    real = False

    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[Prediction]:
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((16, 16))
            px = list(img.getdata())
            r = sum(p[0] for p in px) / len(px)
            g = sum(p[1] for p in px) / len(px)
            b = sum(p[2] for p in px) / len(px)
        except Exception:
            r = g = b = 0.0
        h = int(hashlib.sha256(image_bytes).hexdigest(), 16)
        scored = []
        for i, (_, lat, lon) in enumerate(_CITY_GALLERY):
            s = abs(math.sin(r / 255 * 3 + i)
                    + math.sin(g / 255 * 5 + i * 1.7)
                    + math.sin(b / 255 * 7 + i * 2.3))
            s += ((h >> (i * 4)) & 0xF) / 30.0
            scored.append((s, lat, lon))
        scored.sort(reverse=True)
        top = scored[: max(1, top_k)]
        total = sum(s for s, _, _ in top) or 1.0
        return [{"lat": round(lat, 4), "lon": round(lon, 4), "confidence": round(s / total, 4)}
                for s, lat, lon in top]


class GeoCLIPBackend(Geolocator):
    """Real image->GPS model (GeoCLIP). Lazily loaded."""
    name = "geoclip"
    real = True
    _model = None

    def _load(self):
        if GeoCLIPBackend._model is None:
            from geoclip import GeoCLIP  # heavy import, deferred
            GeoCLIPBackend._model = GeoCLIP()
        return GeoCLIPBackend._model

    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[Prediction]:
        from PIL import Image
        model = self._load()
        path = tempfile.mktemp(suffix=".jpg")
        try:
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(path, format="JPEG")
            gps, probs = model.predict(path, top_k=top_k)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return [{"lat": float(a), "lon": float(b), "confidence": float(c)}
                for (a, b), c in zip(gps, probs)]


class RetrievalBackend(Geolocator):
    """Grounded geolocation: encode -> FAISS k-NN over a geotagged gallery -> fuse.

    The decision comes from real geotagged evidence in the gallery, NOT from a
    language model guessing. Returns the fused estimate plus the nearest
    neighbors (the supporting evidence) as ranked candidates.
    """
    name = "retrieval"
    real = True

    def __init__(self, gallery_path: str | None = None):
        from app.geo.encoders import get_encoder
        from app.geo.retrieval import GalleryIndex
        self.encoder = get_encoder()
        path = gallery_path or os.getenv("GALLERY_PATH", "data/gallery")
        self.gallery = GalleryIndex.load(path)

    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[Prediction]:
        emb = self.encoder.encode(image_bytes)[None, :]
        res = self.gallery.locate(emb, k=max(top_k, 10))
        neigh = res["neighbors"]
        top_sim = neigh[0]["similarity"] if neigh else 0.0
        # #1 = fused estimate (the engine's decision); rest = supporting evidence.
        out = [{"lat": res["lat"], "lon": res["lon"], "confidence": round(float(top_sim), 4)}]
        for n in neigh[: max(0, top_k - 1)]:
            out.append({"lat": n["lat"], "lon": n["lon"], "confidence": round(float(n["similarity"]), 4)})
        return out


_backend: Geolocator | None = None


def get_backend() -> Geolocator:
    """Singleton backend chosen by the GEO_BACKEND env var (auto|stub|geoclip)."""
    global _backend
    if _backend is not None:
        return _backend
    choice = os.getenv("GEO_BACKEND", "auto").lower()
    if choice == "stub":
        _backend = StubBackend()
    elif choice == "geoclip":
        _backend = GeoCLIPBackend()
    elif choice == "retrieval":
        _backend = RetrievalBackend()
    else:  # auto: use the real model if geoclip is importable, else the stub
        try:
            import geoclip  # noqa: F401
            _backend = GeoCLIPBackend()
        except Exception:
            _backend = StubBackend()
    return _backend
