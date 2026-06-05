#!/usr/bin/env python3
"""Encode an OSV-5M slice -> embeddings.npy + coords.npy for the retrieval gallery.

Needs requirements-ml.txt + bandwidth (GPU recommended). OSV-5M ships train.csv
(columns include: id, latitude, longitude, ...) and image zips under images/train.

  python3 scripts/encode_osv5m.py --csv train.csv --images images/train --limit 50000 --out data/osv
  python3 scripts/build_gallery.py --embeddings data/osv.embeddings.npy --coords data/osv.coords.npy --out data/gallery
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.geo.encoders import ClipEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="OSV-5M train.csv")
    ap.add_argument("--images", required=True, help="dir with <id>.jpg images")
    ap.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    enc = ClipEncoder()
    embs, coords = [], []
    with open(args.csv, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if args.limit and i >= args.limit:
                break
            path = os.path.join(args.images, f"{row['id']}.jpg")
            if not os.path.exists(path):
                continue
            with open(path, "rb") as im:
                embs.append(enc.encode(im.read()))
            coords.append((float(row["latitude"]), float(row["longitude"])))
            if len(coords) % 1000 == 0:
                print(f"  encoded {len(coords)}…")

    if not coords:
        raise SystemExit("no images encoded — check --images path and ids")
    np.save(args.out + ".embeddings.npy", np.vstack(embs).astype("float32"))
    np.save(args.out + ".coords.npy", np.array(coords, dtype="float32"))
    print(f"encoded {len(coords)} images -> {args.out}.embeddings.npy / {args.out}.coords.npy")


if __name__ == "__main__":
    main()
