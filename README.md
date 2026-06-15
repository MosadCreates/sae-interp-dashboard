# Mini SAE Trainer

[![CI](https://github.com/MosadCreates/novasight/actions/workflows/ci.yml/badge.svg)](https://github.com/MosadCreates/novasight/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)

Train Sparse Autoencoders on GPT-2 Small residual stream activations and explore learned features through an interactive dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Results](#results)
- [Feature Case Studies](#feature-case-studies)
- [Configuration](#configuration)
- [Key Design Decisions](#key-design-decisions)
- [Deployment](#deployment)
- [References](#references)

---

## Overview

Sparse Autoencoders (SAEs) decompose neural network activations into interpretable features. This project provides a complete, from-scratch implementation of the SAE training pipeline — activation collection, training, analysis, and visualisation — with **no SAELens dependency**.

Built for single-GPU environments (8GB VRAM) using GPT-2 Small (124M) and the TinyStories corpus for fast iteration.

### The Superposition Hypothesis

Neural networks represent more features than they have neurons by encoding features as *directions* in activation space, allowing them to overlap across the same neurons (Elman 1990; Anthropic 2022). A single neuron can activate for multiple unrelated concepts — GPT-2 has a neuron that fires for football, Texas, and country music simultaneously (the "Dallas Cowboys neuron"). SAEs solve this by learning an overcomplete dictionary of feature directions and reconstructing each activation from a sparse combination of them.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│  TinyStories ──► GPT-2 Small ──► HDF5 Store                  │
│                    (layer 8)      (50K tokens)               │
│                                       │                      │
│                                       ▼                      │
│  SparseAutoencoder ◄── HDF5ActivationDataset                 │
│   (d_sae=6144)        (RAM-cached chunks)                    │
│       │                                                      │
│       ▼                                                      │
│  Feature Analysis Pipeline                                   │
│   (top_activations · auto_label · keyword_search · geometry) │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI Backend (:8000)                         │
│  7 endpoints: features, search, prompt, similar...           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│           Next.js Dashboard (:3000)                          │
│  Feature Explorer · Detail · Prompt · Search                 │
│  Dark theme · Recharts · Zustand · SWR                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.11** (3.12+ has PyTorch issues on some platforms)
- **Node.js 20+**
- **RAM**: 8 GB min, 16 GB recommended
- **GPU** (optional): 8 GB+ VRAM; everything runs on CPU if unavailable
- **Disk**: ~5 GB free
- **Git**

### Option 0: Skip to the Dashboard (Pre-trained Artifacts Included)

The repo includes a trained SAE and analysis data. Start here if you want to explore the dashboard immediately:

```bash
# Setup
python -m venv .venv
source .venv/bin/activate              # Linux
.\.venv\Scripts\Activate.ps1           # Windows
pip install -r requirements.txt
cd dashboard && npm install && cd ..

# Terminal 1 — API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Dashboard
cd dashboard && npm run dev
```

Open http://localhost:3000. The API loads `models/case_study/sae_final.pt` (500 steps, 94.3% EV) automatically.

### Option A: Automated Setup (Full Pipeline)

```bash
# Linux / macOS
chmod +x setup.sh && ./setup.sh

# Windows PowerShell
.\setup.ps1
```

Creates venv, installs deps, pre-downloads GPT-2 + TinyStories, creates `.env`.

### Option B: Manual Setup

```bash
git clone https://github.com/your-username/mini-sae-trainer.git
cd mini-sae-trainer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cd dashboard && npm install && cd ..
python -m pytest tests/ -v              # verify everything works
```

### Pipeline Stages

| # | Stage | Command | Time | Output |
|---|---|---|---|---|
| 1 | Collect activations | `python src/data/collect_activations.py --config configs/collect.yaml` | ~2 min CPU | `data/activations.h5` (50K × 768) |
| 2 | Train SAE | `python src/sae/train.py --config configs/train_small.yaml` | ~2 min CPU | `models/case_study/sae_final.pt` |
| 3 | Analyse features | `python src/analysis/pipeline.py --config configs/analyze.yaml` | ~1 min CPU | `results/*.json` |
| 4 | Launch dashboard | `uvicorn src.api.main:app` + `cd dashboard && npm run dev` | — | http://localhost:3000 |

Training configs compared:

| Config | d_sae | Steps | Time (CPU) | Time (T4 GPU) | Explained Variance |
|---|---|---|---|---|---|
| `train_small.yaml` | 128 | 500 | ~2 min | ~10 sec | ~94% |
| `train_medium.yaml` | 6144 | 200K | ~10 hr | ~1 hr | ~85% |
| `train_large.yaml` | 24576 | 500K | ~2 days | ~4 hr | ~80% |

### Dashboard Pages

| Route | Description |
|---|---|
| `/` | Feature Explorer — sortable, paginated table of all features |
| `/features/123` | Feature Detail — top contexts, histogram, similar features, test prompt |
| `/prompt` | Prompt Analyser — enter text, see per-token feature heatmap |
| `/search?q=punctuation` | Search — keyword search with 300 ms debounce |

### Docker

```bash
docker compose up --build
# API on :8000, Dashboard on :3000
```

Mounts `./models` and `./data` as volumes. Requires pre-trained artifacts.

### Troubleshooting

| Problem | Solution |
|---|---|
| `No module named torch` | Activate the venv: `source .venv/bin/activate` |
| `CUDA out of memory` | Reduce `batch_size` in config YAML (try 512) |
| Training < 1 it/s | Enable `cache_in_ram: true` in train config |
| `wandb: Network error` | Set `WANDB_MODE=disabled` in `.env` |
| `h5py: Unable to open file` | Run activation collection first |
| API returns "model not loaded" | Train an SAE first, or use `models/case_study/sae_final.pt` |
| Dashboard loads forever | Start the API on port 8000 |

---

## Project Structure

```
mini-sae-trainer/
├── configs/              # YAML hyperparameter configs
├── dashboard/            # Next.js 14 frontend
│   ├── app/              # 4 routes (page.tsx, features/[id], prompt, search)
│   └── components/       # TokenHeatmap, FeatureCard, ActivationHistogram, LoadingSkeleton
├── docs/
│   ├── theory.md         # Full SAE theory with LaTeX
│   └── feature_case_studies.md  # 9 real feature analyses
├── notebooks/
│   └── 03_feature_case_studies.ipynb
├── src/
│   ├── api/main.py       # FastAPI (7 endpoints)
│   ├── analysis/         # pipeline, top_activations, prompt_activations, auto_label, geometry
│   ├── data/             # collect_activations, hdf5_dataset, corpus
│   ├── sae/              # model, train, loss, resampling
│   └── utils/            # config loader, seeds, logging
├── tests/test_api.py     # 11 pytest tests
├── Dockerfile.api
├── Dockerfile.dashboard
└── docker-compose.yml
```

---

## Results

Trained on TinyStories (50K tokens, d_sae=128, 500 steps):

```
Loss:         6.60
L2:           1.88
L1:           0.00047
L0 Norm:      7.84 active features / token
Explained Variance:  94.3%
```

---

## Feature Case Studies

9 interpretable features found in the trained SAE:

| Feature | Label | Monosemanticity | Concept |
|---|---|---|---|
| 40 | `"time"` | High | Narrative time marker / story opening |
| 95 | `","` | High | Dialogue attribution comma |
| 64 | `"end"` | High | Story ending / temporal conclusion |
| 120 | `"mouse"` | Concept-level high | Animal character introduction |
| 87 | `"down"` | Low-Medium | Directional descriptions |
| 53 | `"said"` | High | Narration verb |
| 98 | `"."` | High | Sentence-ending period |
| 71 | `","` | Medium | Adjective list comma |
| 47 | `"mer"` | Low | Character name suffix |

See [docs/feature_case_studies.md](docs/feature_case_studies.md) for full analysis with activating contexts.

---

## Configuration

All hyperparameters live in `configs/*.yaml`. Defaults:

```yaml
d_model: 768
d_sae: 6144
lambda_l1: 8e-5
lr: 4e-4
batch_size: 1024
n_steps: 200000
lr_warmup_steps: 1000
dead_feature_threshold: 10_000_000
resample_frequency: 5000
```

---

## Key Design Decisions

- **No SAELens dependency** — SAE written from scratch in raw PyTorch
- **HDF5 + RAM caching** — chunked gzip storage + `cache_in_ram=True` gives ~120 it/s on CPU vs ~0.09 it/s without
- **Dual Makefiles** — `Makefile` (GNU/Linux) + `Makefile.ps1` (Windows)
- **Static export fallback** — dashboard works as a static Next.js site if API is unavailable

---

## Deployment

### Hugging Face Spaces

1. Create a Space with Docker runtime
2. Push the repo, set `docker-compose.yml` as the Space Dockerfile
3. Dashboard at `https://your-username-mini-sae-trainer.hf.space`

### Manual Docker

```bash
docker compose up --build -d
```

### Live Demo

**https://mini-sae-trainer.vercel.app** — runs on precomputed static JSON (no API required).

---

## References

- Elman, J. L. (1990). *Finding Structure in Time*. Cognitive Science.
- Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic.
- Templeton, A., et al. (2024). *Scaling Monosemanticity*. Anthropic.
- Gao, L., et al. (2024). *Scaling and Evaluating Sparse Autoencoders*. OpenAI.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
