#!/usr/bin/env python3
"""Build a FAISS gallery index for retrieval-based geolocation.

Real use (on a GPU box): encode your geotagged corpus (e.g. OSV-5M) with a CLIP/
DINOv2 image encoder, save embeddings.npy (N, dim) + coords.npy (N, 2), then:

    python3 scripts/build_gallery.py --embeddings embeddings.npy --coords coords.npy --out data/gallery

Demo (no data needed) — builds a synthetic gallery so the pipeline is runnable:

    python3 scripts/build_gallery.py --demo --out /tmp/gallery
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.geo.retrieval import GalleryIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", help="path to (N, dim) .npy")
    ap.add_argument("--coords", help="path to (N, 2) lat/lon .npy")
    ap.add_argument("--demo", action="store_true", help="build a synthetic gallery")
    ap.add_argument("--out", required=True, help="output index prefix")
    args = ap.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        dim, per = 32, 200
        cities = [(48.85, 2.35), (35.68, 139.65), (-33.87, 151.21), (40.71, -74.0)]
        embs, coords = [], []
        for i, (lat, lon) in enumerate(cities):
            c = rng.normal(size=dim)
            embs.append(c + 0.02 * rng.normal(size=(per, dim)))
            coords.append(np.array([[lat, lon]] * per) + 0.01 * rng.normal(size=(per, 2)))
        embeddings = np.vstack(embs).astype("float32")
        coords = np.vstack(coords).astype("float32")
    else:
        if not (args.embeddings and args.coords):
            ap.error("provide --embeddings and --coords, or use --demo")
        embeddings = np.load(args.embeddings)
        coords = np.load(args.coords)

    gi = GalleryIndex(embeddings.shape[1])
    gi.add(embeddings, coords)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gi.save(args.out)
    print(f"built gallery: {len(gi)} vectors, dim={gi.dim} -> {args.out}.index / {args.out}.coords.npy")


if __name__ == "__main__":
    main()
