"""
FastAPI backend for Mini SAE Feature Dashboard.

Loads trained SAE + GPT-2 at startup and serves feature analysis
data to the Next.js frontend.

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.analysis.auto_label import keyword_search
from src.analysis.prompt_activations import get_prompt_activations
from src.model import ActivationExtractor
from src.sae.model import SparseAutoencoder

# ──────────────────────────────────────────────
# Global state (loaded at startup)
# ──────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.sae: Optional[SparseAutoencoder] = None
        self.extractor: Optional[ActivationExtractor] = None
        self.top_activations: Dict[str, Any] = {}
        self.feature_index: Dict[str, Any] = {}
        self.geometry: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.device: str = "cpu"
        self.ready: bool = False
        self.config: Dict[str, Any] = {}


state = AppState()
logger = logging.getLogger("api")

# Simple in-memory response cache (TTL = 60 seconds)
_response_cache: Dict[str, tuple] = {}  # key -> (timestamp, response)

def _cached(key: str, ttl: int = 60):
    """Return cached response if fresh, else None."""
    entry = _response_cache.get(key)
    if entry and time.time() - entry[0] < ttl:
        return entry[1]
    return None

def _set_cache(key: str, response):
    _response_cache[key] = (time.time(), response)


def load_artifacts(config: Dict[str, Any]) -> None:
    """Load all required models and data files at startup."""
    state.config = config
    state.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {state.device}")

    # --- Load analysis files ---
    top_acts_path = config.get("top_activations_path", "results/top_activations.json")
    if Path(top_acts_path).exists():
        with open(top_acts_path) as f:
            state.top_activations = json.load(f)
        logger.info(f"Loaded top activations from {top_acts_path}")
    else:
        logger.warning(f"Top activations not found at {top_acts_path}")

    index_path = config.get("feature_index_path", "results/feature_index.json")
    if Path(index_path).exists():
        with open(index_path) as f:
            state.feature_index = json.load(f)
        logger.info(f"Loaded feature index from {index_path}")
    else:
        logger.warning(f"Feature index not found at {index_path}")

    geom_path = config.get("geometry_path", "results/geometry_analysis.json")
    if Path(geom_path).exists():
        with open(geom_path) as f:
            state.geometry = json.load(f)
        logger.info(f"Loaded geometry from {geom_path}")
    else:
        logger.warning(f"Geometry not found at {geom_path}")

    # --- Load SAE model ---
    model_path = config.get("model_path", "models/sae_medium/sae_final.pt")
    if Path(model_path).exists():
        logger.info(f"Loading SAE from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        sae_config = checkpoint.get("config", {})
        d_model = sae_config.get("d_model", 768)
        d_sae = sae_config.get("d_sae", 6144)

        state.sae = SparseAutoencoder(d_model=d_model, d_sae=d_sae)
        state.sae.load_state_dict(checkpoint["model_state_dict"])
        state.sae.to(state.device)
        state.sae.eval()

        # Build stats from checkpoint
        state.stats = {
            "d_model": d_model,
            "d_sae": d_sae,
            "lambda_l1": config.get("lambda_l1", sae_config.get("lambda_l1", 8e-5)),
            "training_steps": checkpoint.get("step", 0),
            "l0_sparsity": checkpoint.get("l0", 0.0),
            "explained_variance": checkpoint.get("explained_variance", 0.0),
            "dead_features": sae_config.get("dead_features", 0),
            "corpus_size": sae_config.get("corpus_size", "N/A"),
            "model_name": config.get("model_name", "gpt2"),
        }
        logger.info(f"SAE loaded: d_model={d_model}, d_sae={d_sae}")
    else:
        logger.warning(f"SAE model not found at {model_path}")
        state.stats = {
            "d_model": 768,
            "d_sae": 6144,
            "status": "model_not_loaded",
            "message": "Train an SAE first or update model_path in config",
        }

    # --- Load GPT-2 for live analysis ---
    if state.sae is not None:
        try:
            state.extractor = ActivationExtractor(
                model_name=config.get("model_name", "gpt2"),
                layer=config.get("layer", 8),
                hook_point=config.get("hook_point", "residual_post"),
                device=state.device,
            )
            logger.info("GPT-2 activation extractor loaded")
        except Exception as e:
            logger.warning(f"Failed to load activation extractor: {e}")

    state.ready = True
    logger.info("API ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load artifacts on startup, clean up on shutdown."""
    # Load config
    config_path = os.environ.get("API_CONFIG", "configs/api.yaml")
    if Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Env overrides
    config.setdefault("model_path", os.environ.get("SAE_MODEL_PATH", "models/sae_medium/sae_final.pt"))
    config.setdefault("top_activations_path", os.environ.get("TOP_ACTIVATIONS_PATH", "results/top_activations.json"))
    config.setdefault("feature_index_path", os.environ.get("FEATURE_INDEX_PATH", "results/feature_index.json"))
    config.setdefault("geometry_path", os.environ.get("GEOMETRY_PATH", "results/geometry_analysis.json"))
    config.setdefault("port", int(os.environ.get("API_PORT", "8000")))
    config.setdefault("host", os.environ.get("API_HOST", "0.0.0.0"))

    load_artifacts(config)
    yield

    # Cleanup
    if state.extractor is not None:
        state.extractor.remove_hooks()


app = FastAPI(
    title="Mini SAE Dashboard API",
    description="Backend for the Mini SAE Feature Dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request logging middleware ---
@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(f"{request.method} {request.url.path} — {response.status_code} ({elapsed:.3f}s)")
    return response


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok" if state.ready else "loading",
        "model_loaded": state.sae is not None,
        "gpt2_loaded": state.extractor is not None,
        "device": state.device,
    })


@app.get("/api/features")
async def list_features(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("frequency", pattern="^(frequency|max_activation|feature_id)$"),
):
    if not state.feature_index:
        return JSONResponse({"features": [], "total": 0, "page": page, "page_size": page_size})

    cache_key = f"features:{page}:{page_size}:{sort_by}"
    cached = _cached(cache_key)
    if cached:
        return JSONResponse(cached)

    features = list(state.feature_index.get("features", {}).values())

    reverse = sort_by != "feature_id"
    features.sort(key=lambda f: f.get(sort_by, 0), reverse=reverse)

    start = (page - 1) * page_size
    end = start + page_size
    page_features = features[start:end]

    result = {
        "features": page_features,
        "total": len(features),
        "page": page,
        "page_size": page_size,
    }
    _set_cache(cache_key, result)
    return JSONResponse(result)


@app.get("/api/features/{feature_id}")
async def get_feature(feature_id: int):
    fid = str(feature_id)

    # Basic info from feature index
    feature_info = state.feature_index.get("features", {}).get(fid, {})
    if not feature_info and fid not in state.top_activations.get("features", {}):
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")

    # Full data from top_activations
    act_data = state.top_activations.get("features", {}).get(fid, {})

    # Similar features from geometry
    similar = []
    geom_similar = state.geometry.get("geometry", {}).get("similar_pairs", [])
    for pair in geom_similar:
        if pair.get("feature_a") == feature_id:
            similar.append({
                "feature_id": pair["feature_b"],
                "similarity": pair["cosine_similarity"],
            })
        elif pair.get("feature_b") == feature_id:
            similar.append({
                "feature_id": pair["feature_a"],
                "similarity": pair["cosine_similarity"],
            })
    similar.sort(key=lambda x: -x["similarity"])

    result = {
        "feature_id": feature_id,
        "label": feature_info.get("label", f"feature_{feature_id}"),
        "frequency": feature_info.get("frequency", act_data.get("frequency", 0)),
        "max_activation": feature_info.get("max_activation", act_data.get("max_activation", 0)),
        "decoder_norm": 1.0,  # decoder columns are unit-normed
        "top_contexts": act_data.get("top_contexts", [])[:50],
        "histogram": act_data.get("histogram", []),
        "similar_features": similar[:10],
    }

    return JSONResponse(result)


@app.post("/api/prompt")
async def analyze_prompt(body: dict):
    prompt = body.get("prompt", "")
    top_k_features = body.get("top_k_features", 10)

    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    if state.sae is None or state.extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = get_prompt_activations(prompt, state.extractor, state.sae)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Map to expected response format
    tokens = result["tokens"]
    top_features = result["top_features"][:top_k_features]

    # Build per-token top-K
    per_token = []
    for pos, token in enumerate(tokens):
        pos_features = result["per_token_features"][pos][:top_k_features]
        per_token.append({
            "token": token,
            "activations": pos_features,
        })

    return JSONResponse({
        "tokens": per_token,
        "top_features": top_features,
        "full_matrix": result.get("full_matrix", []),
        "active_feature_ids": result.get("active_feature_ids", []),
    })


@app.get("/api/search")
async def search_features(q: str = Query("", min_length=1)):
    if not state.feature_index:
        return JSONResponse({"results": []})

    cache_key = f"search:{q}"
    cached = _cached(cache_key, ttl=30)
    if cached:
        return JSONResponse(cached)

    results = keyword_search(q, state.feature_index, max_results=20)
    response = {"results": results}
    _set_cache(cache_key, response)
    return JSONResponse(response)


@app.get("/api/features/{feature_id}/similar")
async def get_similar_features(feature_id: int):
    similar = []
    geom_similar = state.geometry.get("geometry", {}).get("similar_pairs", [])
    for pair in geom_similar:
        if pair.get("feature_a") == feature_id:
            similar.append({
                "feature_id": pair["feature_b"],
                "cosine_similarity": pair["cosine_similarity"],
            })
        elif pair.get("feature_b") == feature_id:
            similar.append({
                "feature_id": pair["feature_a"],
                "cosine_similarity": pair["cosine_similarity"],
            })

    similar.sort(key=lambda x: -x["cosine_similarity"])
    return JSONResponse({"feature_id": feature_id, "similar_features": similar[:10]})


@app.get("/api/stats")
async def get_stats():
    return JSONResponse(state.stats)
