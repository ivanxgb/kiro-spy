"""P0 — Evaluation harness for image geolocation.

Metrics standard in the literature: great-circle (haversine) distance error,
percentage of predictions within distance radii, and median error.
Pure-Python, no heavy deps, so it runs anywhere.
"""
from __future__ import annotations
import math
from typing import Iterable, Sequence, Tuple

Coord = Tuple[float, float]  # (lat, lon) in degrees

# Standard evaluation radii (km): street, city, region, country, continent.
THRESHOLDS_KM = {"street": 1, "city": 25, "region": 200, "country": 750, "continent": 2500}

EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: Coord, b: Coord) -> float:
    """Great-circle distance between two (lat, lon) points, in kilometers."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def evaluate(preds: Iterable[Coord], gts: Iterable[Coord]) -> dict:
    """Compute median km error and accuracy@radius for paired predictions/ground-truth."""
    dists = [haversine_km(p, g) for p, g in zip(preds, gts)]
    if not dists:
        raise ValueError("no predictions to evaluate")
    n = len(dists)
    acc = {k: round(100 * sum(d <= v for d in dists) / n, 2) for k, v in THRESHOLDS_KM.items()}
    return {"n": n, "median_km": round(median(dists), 2), "acc_pct": acc}


if __name__ == "__main__":
    # Self-test: known landmark distances (no test framework needed).
    paris, london = (48.8566, 2.3522), (51.5074, -0.1278)
    d = haversine_km(paris, london)
    assert 340 < d < 350, f"Paris-London should be ~344km, got {d:.1f}"
    assert haversine_km(paris, paris) == 0.0

    # Perfect predictions -> 0 km median, 100% at every tier.
    r = evaluate([paris, london], [paris, london])
    assert r["median_km"] == 0.0 and r["acc_pct"]["street"] == 100.0, r

    # One perfect, one ~344km off -> 50% within city (25km), 100% within country (750km).
    r2 = evaluate([paris, paris], [paris, london])
    assert r2["acc_pct"]["city"] == 50.0 and r2["acc_pct"]["country"] == 100.0, r2

    print(f"Paris->London = {d:.1f} km")
    print("evaluate(perfect):", r)
    print("evaluate(half-off):", r2)
    print("OK: metrics self-test passed")
