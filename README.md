# kiro-spy

Research toolkit and phased build guide for image geolocation from pixels.

`kiro-spy` explores how modern systems estimate where a photo was taken without
using EXIF/GPS metadata. It combines a written roadmap with runnable FastAPI
prototypes for:

- metadata stripping
- image-to-location model backends
- retrieval over geotagged image galleries
- haversine evaluation
- FAISS-based nearest-neighbor search
- browser-based map visualization

The project is intentionally structured as a learning and research repo rather
than a surveillance product.

## What Is Inside

- A browsable roadmap in [index.html](./index.html).
- A FastAPI demo service with `/health` and `/locate`.
- A metadata-stripping upload path that re-encodes images before inference.
- A lightweight stub backend so the app runs without GPU or heavy model downloads.
- Optional GeoCLIP backend for real image-to-GPS predictions.
- A retrieval backend that uses CLIP/StreetCLIP-style embeddings plus FAISS.
- Synthetic and demo gallery builders for testing the retrieval pipeline.
- Haversine metrics for distance/error reporting.

## Why This Exists

Image geolocation is a useful research area for:

- visual place recognition
- open-street-map and Mapillary-style retrieval
- robustness benchmarks for multimodal models
- geographic dataset quality
- safety analysis around location inference

It is also sensitive. The repo avoids hidden data collection and keeps the
implementation explicit so researchers can inspect how each signal is produced.

## Status

Implemented:

- P1 FastAPI service and browser UI
- pluggable backend interface
- EXIF-stripping upload path
- deterministic no-ML stub backend
- optional GeoCLIP backend
- P2 FAISS retrieval module
- hash and CLIP encoders
- demo gallery build scripts

Not production ready:

- no authentication
- no rate limiting
- no abuse controls
- no privacy-preserving deployment wrapper
- real-world accuracy depends entirely on model choice and gallery coverage

Do not expose the demo service to the public internet without adding those
controls.

## Architecture

```text
image upload
  |
  | strip metadata by re-encoding pixels
  v
backend selector
  |
  | stub | geoclip | retrieval
  v
top-k coordinates
  |
  v
Leaflet map UI + JSON response
```

Retrieval mode:

```text
image -> encoder -> normalized embedding -> FAISS gallery -> nearest neighbors
      -> spherical coordinate fusion -> fused estimate + evidence
```

## Quick Start

Requirements:

- Python 3.11+
- pip

```bash
git clone https://github.com/ivanxgb/kiro-spy.git
cd kiro-spy

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

GEO_BACKEND=stub uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

The default lightweight path is useful for testing API/UI wiring. It is not real
geolocation.

## Real Model Backend

Install optional heavy dependencies:

```bash
pip install -r requirements-ml.txt
GEO_BACKEND=geoclip uvicorn app.main:app --reload
```

The GeoCLIP backend lazily loads the model on first prediction.

## Retrieval Backend

Build or provide a geotagged gallery, then run:

```bash
GEO_BACKEND=retrieval \
ENCODER=hash \
GALLERY_PATH=data/gallery \
uvicorn app.main:app --reload
```

For real embeddings:

```bash
pip install -r requirements-ml.txt

GEO_BACKEND=retrieval \
ENCODER=clip \
CLIP_MODEL=geolocal/StreetCLIP \
GALLERY_PATH=data/gallery \
uvicorn app.main:app --reload
```

The retrieval backend returns a fused estimate first, followed by nearest
neighbors as supporting evidence.

## API

```http
GET /health
```

Returns backend status.

```http
POST /locate?top_k=5
Content-Type: multipart/form-data
```

Upload an image file under the `file` field.

Example:

```bash
curl -F "file=@photo.jpg" "http://127.0.0.1:8000/locate?top_k=5"
```

Response:

```json
{
  "backend": "stub",
  "real_model": false,
  "note": "STUB backend: placeholder predictions, not real geolocation...",
  "predictions": [
    { "lat": 48.8566, "lon": 2.3522, "confidence": 0.42 }
  ]
}
```

## Responsible Use

This project is for research, education, benchmarking, and defensive analysis.

Do not use it to identify private people, infer sensitive locations, stalk,
harass, dox, or bypass privacy expectations. If you deploy a real model, add
authentication, rate limiting, logging, user consent boundaries, and clear data
retention rules.

## Roadmap

The roadmap in [index.html](./index.html) covers:

- visual place recognition and retrieval
- geocell classification
- GeoCLIP-style image/GPS alignment
- multimodal reasoning over retrieved candidates
- dataset and evaluation strategy
- production controls that should exist before any public service

## License

MIT. See [LICENSE](./LICENSE).
