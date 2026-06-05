"""Image encoders for retrieval-based geolocation. Pluggable like the backends.

- ``ClipEncoder``: real image embeddings from a CLIP/StreetCLIP model (transformers,
  lazily imported). This is the production encoder.
- ``HashEncoder``: deterministic, ML-free embedding (a fixed random projection of a
  small color grid) so the retrieval pipeline is runnable/testable WITHOUT torch or
  a GPU. Same image -> same vector; visually similar images -> similar vectors.

Select via env ``ENCODER`` = auto (default) | clip | hash.
CLIP model id via env ``CLIP_MODEL`` (default ``geolocal/StreetCLIP``).
"""
from __future__ import annotations
import io
import os
from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def encode(self, image_bytes: bytes) -> np.ndarray:
        """Return a 1-D float32 embedding of shape (dim,)."""
        ...


class HashEncoder(Encoder):
    """Deterministic ML-free embedding for testing the retrieval wiring."""
    name = "hash"

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._feat = 8 * 8 * 3  # 8x8 RGB color grid
        # Fixed projection (same for every image) so vectors are comparable.
        self._proj = np.random.default_rng(0).standard_normal((self._feat, dim)).astype("float32")

    def encode(self, image_bytes: bytes) -> np.ndarray:
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((8, 8))
            v = np.asarray(img, dtype="float32").reshape(-1) / 255.0
        except Exception:
            v = np.zeros(self._feat, dtype="float32")
        return (v @ self._proj).astype("float32")


class ClipEncoder(Encoder):
    """Real CLIP/StreetCLIP image embeddings. Heavy deps loaded lazily."""
    name = "clip"
    _model = None
    _proc = None

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("CLIP_MODEL", "geolocal/StreetCLIP")

    def _load(self):
        if ClipEncoder._model is None:
            from transformers import CLIPModel, CLIPProcessor
            ClipEncoder._model = CLIPModel.from_pretrained(self.model_id).eval()
            ClipEncoder._proc = CLIPProcessor.from_pretrained(self.model_id)
        return ClipEncoder._model, ClipEncoder._proc

    def encode(self, image_bytes: bytes) -> np.ndarray:
        import torch
        from PIL import Image
        model, proc = self._load()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
        v = feat[0].cpu().numpy().astype("float32")
        self.dim = int(v.shape[0])
        return v


_encoder: Encoder | None = None


def get_encoder() -> Encoder:
    """Singleton encoder chosen by env ENCODER (auto|clip|hash)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    choice = os.getenv("ENCODER", "auto").lower()
    if choice == "hash":
        _encoder = HashEncoder()
    elif choice == "clip":
        _encoder = ClipEncoder()
    else:  # auto: real CLIP if transformers is importable, else hash
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            _encoder = ClipEncoder()
        except Exception:
            _encoder = HashEncoder()
    return _encoder
