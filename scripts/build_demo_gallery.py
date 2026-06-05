#!/usr/bin/env python3
"""Build a runnable DEMO retrieval gallery — no OSV-5M, no GPU.

Generates color-coded synthetic images for a few cities, embeds them with the
active (hash) encoder, and saves a FAISS gallery. Lets you exercise the retrieval
backend end-to-end. For REAL grounding use scripts/encode_osv5m.py.

    python3 scripts/build_demo_gallery.py [out_prefix]   # default: data/gallery
"""
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.geo.encoders import HashEncoder  # noqa: E402
from app.geo.retrieval import GalleryIndex  # noqa: E402

# (name, lat, lon, RGB) — each city gets a distinct dominant color.
CITIES = [
    ("Paris", 48.8566, 2.3522, (60, 90, 200)),
    ("Tokyo", 35.6762, 139.6503, (200, 60, 60)),
    ("Cairo", 30.0444, 31.2357, (220, 200, 90)),
    ("Sydney", -33.8688, 151.2093, (60, 180, 120)),
]


def _img_bytes(color, rng, jitter=25):
    arr = np.clip(np.array(color) + rng.integers(-jitter, jitter + 1, 3), 0, 255).astype("uint8")
    im = Image.new("RGB", (32, 32), tuple(int(x) for x in arr))
    b = io.BytesIO()
    im.save(b, format="JPEG")
    return b.getvalue()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "data/gallery"
    enc = HashEncoder()
    rng = np.random.default_rng(0)
    embs, coords = [], []
    for _, lat, lon, color in CITIES:
        for _ in range(30):
            embs.append(enc.encode(_img_bytes(color, rng)))
            coords.append((lat, lon))
    gi = GalleryIndex(enc.dim)
    gi.add(np.vstack(embs), np.array(coords))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    gi.save(out)
    print(f"demo gallery: {len(gi)} vectors, dim={gi.dim} -> {out}.index")


if __name__ == "__main__":
    main()
