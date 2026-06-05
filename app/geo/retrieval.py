"""P2 — Retrieval-based geolocation over a geotagged embedding gallery.

Pipeline: encode images -> L2-normalized embeddings -> FAISS inner-product
(= cosine) index of a geotagged gallery -> k-NN -> similarity-weighted *spherical*
fusion of neighbor coordinates into a single (lat, lon).

Encoder-agnostic: feed embeddings from CLIP / DINOv2 / StreetCLIP. Coordinate
fusion averages unit vectors on the sphere (correct across the antimeridian/poles),
not raw lat/lon. Tested with synthetic embeddings (no GPU needed).
"""
from __future__ import annotations
import math
import numpy as np
import faiss

Coord = tuple[float, float]


def normalize(x) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    if x.ndim == 1:
        x = x[None, :]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def _latlon_to_xyz(coords: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=1)


def _xyz_to_latlon(v: np.ndarray) -> Coord:
    x, y, z = v
    return (round(math.degrees(math.atan2(z, math.hypot(x, y))), 6),
            round(math.degrees(math.atan2(y, x)), 6))


class GalleryIndex:
    """A FAISS cosine index of geotagged embeddings + their coordinates."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.coords = np.empty((0, 2), dtype="float32")

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, embeddings, coords) -> None:
        emb = normalize(embeddings)
        if emb.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {emb.shape[1]}")
        self.index.add(emb)
        self.coords = np.vstack([self.coords, np.asarray(coords, dtype="float32")])

    def search(self, query, k: int = 10):
        sims, idx = self.index.search(normalize(query), min(k, len(self)))
        return sims[0], idx[0]

    def locate(self, query, k: int = 10, temperature: float = 0.1) -> dict:
        sims, idx = self.search(query, k)
        w = np.exp(sims / temperature)
        w /= w.sum()
        mean = (w[:, None] * _latlon_to_xyz(self.coords[idx])).sum(axis=0)
        lat, lon = _xyz_to_latlon(mean)
        neighbors = [{"lat": float(self.coords[i][0]), "lon": float(self.coords[i][1]),
                      "similarity": round(float(s), 4)} for s, i in zip(sims, idx)]
        return {"lat": lat, "lon": lon, "neighbors": neighbors}

    def save(self, prefix: str) -> None:
        faiss.write_index(self.index, prefix + ".index")
        np.save(prefix + ".coords.npy", self.coords)

    @classmethod
    def load(cls, prefix: str) -> "GalleryIndex":
        idx = faiss.read_index(prefix + ".index")
        obj = cls(idx.d)
        obj.index = idx
        obj.coords = np.load(prefix + ".coords.npy")
        return obj


if __name__ == "__main__":
    from app.geo.metrics import haversine_km
    rng = np.random.default_rng(0)
    dim = 32
    cities = {"Paris": (48.85, 2.35), "Tokyo": (35.68, 139.65), "Sydney": (-33.87, 151.21)}
    centers = {n: normalize(rng.normal(size=dim))[0] for n in cities}

    gi = GalleryIndex(dim)
    for name, (lat, lon) in cities.items():
        embs = centers[name] + 0.02 * rng.normal(size=(50, dim))
        coords = np.array([[lat, lon]] * 50) + 0.01 * rng.normal(size=(50, 2))
        gi.add(embs, coords)

    # Query close to the Paris cluster -> fusion should land near Paris.
    q = centers["Paris"] + 0.02 * rng.normal(size=dim)
    res = gi.locate(q, k=10)
    err = haversine_km((res["lat"], res["lon"]), cities["Paris"])
    assert err < 50, (err, res)

    # Round-trip save/load.
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "gal")
    gi.save(p)
    assert len(GalleryIndex.load(p)) == len(gi)

    print(f"gallery size = {len(gi)} | fused = {res['lat']:.3f},{res['lon']:.3f} "
          f"| err = {err:.2f} km | top sim = {res['neighbors'][0]['similarity']}")
    print("OK: retrieval self-test passed (query matched Paris; save/load ok)")
