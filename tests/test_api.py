"""Tests for FastAPI endpoints.

Creates fake analysis JSON files before importing the app,
then tests all endpoints via TestClient with lifespan support.
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Create fake analysis data files ────────────────────────
test_results = Path("results")
test_results.mkdir(exist_ok=True)

fake_index = {
    "features": {
        "0": {
            "feature_id": 0, "label": "test_feature", "top_tokens": ["hello", "world"],
            "frequency": 0.05, "max_activation": 10.0,
        },
        "1": {
            "feature_id": 1, "label": "months", "top_tokens": ["january", "february"],
            "frequency": 0.03, "max_activation": 8.0,
        },
    },
    "search_terms": {
        "hello": [0],
        "world": [0],
        "january": [1],
        "february": [1],
    },
}
with open("results/feature_index.json", "w") as f:
    json.dump(fake_index, f)

fake_top_acts = {
    "features": {
        "0": {
            "feature_id": 0,
            "top_contexts": [
                {"tokens": ["hello", " ", "world"], "activation": 5.0},
                {"tokens": ["hi", " ", "there"], "activation": 3.0},
            ],
            "max_activation": 10.0,
            "mean_activation": 4.0,
            "frequency": 0.05,
            "histogram": [{"bin_center": 1.0, "count": 100}],
        },
        "1": {
            "feature_id": 1,
            "top_contexts": [
                {"tokens": ["january", " ", "1"], "activation": 4.0},
            ],
            "max_activation": 8.0,
            "mean_activation": 4.0,
            "frequency": 0.03,
            "histogram": [{"bin_center": 1.0, "count": 50}],
        },
    },
    "metadata": {"n_features": 2, "total_tokens": 10000},
}
with open("results/top_activations.json", "w") as f:
    json.dump(fake_top_acts, f)

fake_geom = {
    "d_sae": 6144,
    "d_model": 768,
    "geometry": {
        "similar_pairs": [
            {"feature_a": 0, "feature_b": 1, "cosine_similarity": 0.45},
        ],
    },
}
with open("results/geometry_analysis.json", "w") as f:
    json.dump(fake_geom, f)

# ── Create minimal api config ──────────────────────────────
fake_api_config = {
    "model_path": "models/nonexistent.pt",
    "top_activations_path": "results/top_activations.json",
    "feature_index_path": "results/feature_index.json",
    "geometry_path": "results/geometry_analysis.json",
    "log_level": "info",
}
with open("configs/api.yaml", "w") as f:
    json.dump(fake_api_config, f)

# ── Import app after files exist ───────────────────────────
from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_list_features(client):
    resp = client.get("/api/features")
    assert resp.status_code == 200
    data = resp.json()
    assert "features" in data
    assert "total" in data
    assert data["total"] == 2


def test_list_features_pagination(client):
    resp = client.get("/api/features?page=1&page_size=1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["features"]) == 1


def test_list_features_sort(client):
    resp = client.get("/api/features?sort_by=max_activation")
    assert resp.status_code == 200
    data = resp.json()
    if len(data["features"]) >= 2:
        assert data["features"][0]["max_activation"] >= data["features"][1]["max_activation"]


def test_get_feature(client):
    resp = client.get("/api/features/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature_id"] == 0
    assert "label" in data
    assert "top_contexts" in data
    assert len(data["top_contexts"]) > 0


def test_get_feature_not_found(client):
    resp = client.get("/api/features/99999")
    assert resp.status_code == 404


def test_search_features(client):
    resp = client.get("/api/search?q=hello")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) > 0


def test_search_features_no_results(client):
    resp = client.get("/api/search?q=zzzznonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 0


def test_similar_features(client):
    resp = client.get("/api/features/0/similar")
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature_id"] == 0
    assert "similar_features" in data


def test_stats(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "d_model" in data
    assert "d_sae" in data
    assert data["status"] == "model_not_loaded"


def test_prompt_no_model(client):
    resp = client.post("/api/prompt", json={"prompt": "hello world"})
    assert resp.status_code == 503
