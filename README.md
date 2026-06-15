# Mini SAE Trainer

[![CI](https://github.com/your-username/mini-sae-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/mini-sae-trainer/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)

Train Sparse Autoencoders on GPT-2 Small residual stream activations and explore learned features through an interactive dashboard.

---

## Overview

Sparse Autoencoders (SAEs) are a leading mechanistic interpretability tool for decomposing neural network activations into interpretable features. This project provides a complete, from-scratch implementation of the SAE training pipeline — activation collection, training, analysis, and visualisation — with no SAELens dependency.

Built for single-GPU environments (8GB VRAM) using GPT-2 Small (124M) and the TinyStories corpus for fast iteration.

### The Superposition Hypothesis

Neural networks represent more features than they have neurons. This is possible because features are encoded as *directions* in activation space, allowing them to overlap (superpose) across the same neurons — a phenomenon known as the **superposition hypothesis** (Elman 1990; Anthropic 2022). A single neuron can activate for multiple unrelated concepts: for example, a neuron in GPT-2 has been observed to fire for football, Texas, and country music simultaneously (the infamous "Dallas Cowboys neuron"). This makes direct neuron-by-neuron analysis misleading. SAEs solve this by learning an overcomplete dictionary of feature directions, then reconstructing each activation from a sparse combination of these features, effectively disentangling the superposed concepts.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Training Pipeline                   │
│                                                      │
│  TinyStories ──► GPT-2 Small ──► HDF5 Store         │
│                     (layer 8)      (50K tokens)      │
│                                        │             │
│                                        ▼             │
│  Sparse Autoencoder ◄── HDF5ActivationDataset        │
│   (d_sae=6144)         (RAM-cached chunks)           │
│       │                                              │
│       ▼                                              │
│  Feature Analysis Pipeline                           │
│   ┌─────────────────┐                                │
│   │ top_activations  │                                │
│   │ auto_label       │                                │
│   │ keyword_search   │                                │
│   │ geometry         │                                │
│   └─────────────────┘                                │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend (:8000)                  │
│  7 endpoints: features, search, prompt, similar...   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           Next.js Dashboard (:3000)                  │
│  Feature Explorer │ Detail │ Prompt │ Search         │
│  Dark theme │ Recharts │ Zustand │ SWR              │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.11** (required; 3.12+ has known PyTorch version issues on some platforms)
- **Node.js 20+** (for the dashboard)
- **RAM**: 8GB minimum for CPU training; 16GB recommended for the full pipeline
- **GPU** (optional): Any CUDA GPU with 8GB+ VRAM (T4, A100, RTX 3070+). If unavailable, everything runs on CPU (slower but functional)
- **Disk**: ~5GB free for model weights, dataset cache, and activations
- **Git** (for cloning)

---

### Setup Options

Choose **one** of the following:

#### Option A: Automated Setup (Recommended)

```bash
# Linux / macOS
chmod +x setup.sh && ./setup.sh

# Windows PowerShell
.\setup.ps1
```

This will:
1. Create a Python virtual environment at `.venv/`
2. Install all Python dependencies (pinned versions)
3. Install spaCy (for automated feature labelling)
4. Create `data/`, `models/`, `results/` directories
5. Pre-download GPT-2 Small tokenizer and model weights (~500MB)
6. Pre-download TinyStories dataset (~200MB cache)
7. Create `.env` from `.env.example` (edit it to add your WandB API key)
8. Install Node.js dependencies in `dashboard/`

#### Option B: Manual Setup

```bash
# 1. Clone
git clone https://github.com/your-username/mini-sae-trainer.git
cd mini-sae-trainer

# 2. Python virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Pre-download models and dataset (prevents hangups later)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
AutoTokenizer.from_pretrained('gpt2')
AutoModelForCausalLM.from_pretrained('gpt2', output_hidden_states=True)
load_dataset('roneneldan/TinyStories', split='train', trust_remote_code=True)
print('Downloads complete')
"

# 5. Create directories
mkdir -p data models results notebooks tests

# 6. Configure environment
cp .env.example .env   # Edit .env to set WANDB_API_KEY (optional, not needed for local use)

# 7. Install dashboard dependencies
cd dashboard && npm install && cd ..

# 8. Verify everything works
python -m pytest tests/ -v
```

---

### Running the Full Pipeline

The pipeline has 4 stages. You can run all at once or step through individually.

#### Stage 1: Collect Activations

```bash
# Minimal run (50K tokens, ~30 seconds on GPU, ~2 minutes on CPU)
python src/data/collect_activations.py --config configs/collect.yaml

# Or with explicit args:
python src/data/collect_activations.py \
    --model_name gpt2 \
    --layer 8 \
    --hook_point residual_post \
    --max_tokens 50000 \
    --batch_size 8 \
    --output_path data/activations.h5
```

**What happens**: Downloads TinyStories (~200MB HF cache), tokenises with GPT-2 tokenizer, runs GPT-2 Small in inference mode (no gradients), extracts residual stream activations at layer 8, flattens to individual token vectors, mean-centres them, and saves to HDF5.

**Expected output**:
- `data/activations.h5` — main activation store with datasets: `activations` (50,048 × 768), `mean` (768,), `std` (768,), `sample` (10,000 × 768)
- `data/activations.mean.npy` — mean vector (also saved separately for reuse)

**File size**: ~150MB for 50K tokens; scales linearly with `--max_tokens`.

**Verify**: Check `data/activations.h5` exists and is > 50MB.

#### Stage 2: Train SAE

```bash
# Fast demo (d_sae=128, 500 steps, ~2 min on CPU, 94% explained variance)
python src/sae/train.py --config configs/train_small.yaml

# Full training (d_sae=6144, 200K steps, ~1 hour on T4)
python src/sae/train.py --config configs/train_medium.yaml

# Or with explicit args:
python src/sae/train.py \
    --activations_path data/activations.h5 \
    --d_sae 6144 \
    --lambda_l1 8e-5 \
    --lr 4e-4 \
    --batch_size 1024 \
    --n_steps 200000 \
    --output_dir models/sae_medium
```

**What happens**: Loads the HDF5 activations (with RAM caching for speed), initialises a SparseAutoencoder, runs the training loop with Adam (no weight decay), applies decoder normalisation after every step, logs metrics to WandB, resamples dead features every 5000 steps, and saves checkpoints.

**Training configs compared**:

| Config | d_sae | Expansion | Steps | Time (CPU) | Time (T4 GPU) | EV |
|---|---|---|---|---|---|---|
| `train_small.yaml` | 128 | 0.17× | 500 | ~2 min | ~10 sec | ~94% |
| `train_medium.yaml` | 6144 | 8× | 200K | ~10 hr | ~1 hr | ~85% |
| `train_large.yaml` | 24576 | 32× | 500K | ~2 days | ~4 hr | ~80% |

**Metrics logged to WandB** (or console if WandB is offline):
- `total_loss`, `l2_loss`, `l1_loss` — loss components
- `l0_norm` — mean active features per token (target: 10–50)
- `explained_variance` — reconstruction quality (target: >80%)
- `dead_features` — features that never fire (should decrease after resampling)
- `lr`, `gradient_norm` — training dynamics

**Verify**: Watch the explained variance climb and L0 norm stabilise. For `train_small.yaml`, expect EV > 90% by step 300.

**Important notes**:
- CPU training is viable but use `cache_in_ram=True` (default in configs) — without it, HDF5 gzip decompression makes each epoch ~1000× slower
- If using CUDA, the first batch triggers GPU warmup which can take ~10 seconds
- Dead feature resampling happens at step 5000, 10000, etc. — you may see a temporary loss spike followed by recovery

#### Stage 3: Analyse Features

```bash
python src/analysis/pipeline.py --config configs/analyze.yaml

# Or with explicit args:
python src/analysis/pipeline.py \
    --sae-path models/sae_medium/sae_final.pt \
    --activations-path data/activations.h5 \
    --output-dir results
```

**What happens**: Runs 4 analysis steps:
1. **Top activations** — streams corpus through SAE, finds top-K activating contexts per feature
2. **Feature index** — auto-labels each feature by its most common activating token, builds search index
3. **Geometry** — computes pairwise decoder cosine similarities, finds feature clusters
4. Saves all results to `results/` as JSON files

**Expected output**:
- `results/top_activations.json` — per-feature top contexts with activation values
- `results/feature_index.json` — auto-labels, search terms, frequency stats
- `results/geometry_analysis.json` — pairwise similarities, similar feature pairs
- `results/geometry/` — diagnostic plots (if matplotlib available)

**Verify**: Check `results/feature_index.json` exists and contains entries for most features (some may be dead).

#### Stage 4: Launch Dashboard

You need **two terminals** running simultaneously.

```bash
# Terminal 1 — FastAPI Backend (port 8000)
# ------------------------------------------
# Activate the virtual environment first, then:
uvicorn src.api.main:app --reload --port 8000

# Expected output (after ~10 seconds for model loading):
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Loading SAE from models/sae_medium/sae_final.pt
# INFO:     SAE loaded: d_model=768, d_sae=6144
# INFO:     GPT-2 activation extractor loaded
# INFO:     API ready

# Verify the API is working:
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true,"gpt2_loaded":true,"device":"cpu"}

curl http://localhost:8000/api/stats
# → {"d_model":768,"d_sae":6144,"l0_sparsity":7.84,"explained_variance":0.943,...}
```

```bash
# Terminal 2 — Next.js Dashboard (port 3000)
# -------------------------------------------
cd dashboard
npm run dev

# Expected output:
# ▲ Next.js 14.1.0
# Local: http://localhost:3000
```

Open **http://localhost:3000** in your browser.

**Dashboard pages**:
| Route | What it does |
|---|---|
| `/` | Feature Explorer — sortable table of all features, paginated |
| `/features/123` | Feature Detail — top contexts, histogram, similar features, "Test This Feature" |
| `/prompt` | Prompt Analyser — enter text, see per-token feature activations as a heatmap |
| `/search?q=punctuation` | Search — find features by keyword with 300ms debounce |

**If the API is not running**: The dashboard will show loading spinners and graceful error states. It is designed to work with a live API — the static export mode (for Vercel) uses pre-bundled JSON instead.

---

### Docker (Alternative to Manual Setup)

```bash
# Build and start both services
docker compose up --build

# Run in background
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

**Services**:
| Service | Port | URL |
|---|---|---|
| FastAPI | 8000 | http://localhost:8000/docs |
| Dashboard | 3000 | http://localhost:3000 |

**Volumes** (persist data between runs):
- `./models:/app/models` — SAE checkpoints
- `./data:/app/data` — activation HDF5 files

**Important**: The Docker setup requires pre-trained SAE weights in `models/` and activation data in `data/`. Run the training pipeline locally first, or download pre-trained artifacts.

---

### Using Pre-trained Case Study Artifacts (Skip Straight to Dashboard)

The repository includes a pre-trained SAE (d_sae=128) and analysis data from the case study. Use these to explore the dashboard immediately without running the full pipeline:

```bash
# 1. Setup (if not done already)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd dashboard && npm install && cd ..

# 2. Start the API (loads the pre-trained model automatically)
uvicorn src.api.main:app --reload --port 8000

# 3. In another terminal, start the dashboard
cd dashboard && npm run dev
```

The API will load `models/case_study/sae_final.pt` and corresponding analysis data from `results/` (if present). If no analysis data exists, the API will still work but some features will be limited.

Pre-trained artifacts included:
- `models/case_study/sae_final.pt` — SAE trained for 500 steps, 94.3% EV
- `data/activations_cs.h5` — 50K activations used for training

---

### Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `ImportError: No module named torch` | Virtual environment not activated | Run `source .venv/bin/activate` (Linux) or `.venv\Scripts\Activate.ps1` (Windows) |
| `CUDA out of memory` | Batch size too large for GPU | Reduce `batch_size` in config YAML (try 512 or 256) |
| Training is very slow ( < 1 it/s ) | `cache_in_ram=False` or no GPU | Enable `cache_in_ram: true` in train config; install PyTorch with CUDA |
| `wandb: Network error` | No internet or no WandB account | Set `WANDB_MODE=disabled` in `.env` or unset `WANDB_API_KEY` |
| `h5py: Unable to open file` | Activations not collected yet | Run `python src/data/collect_activations.py --config configs/collect.yaml` first |
| `No module named 'spacy'` | spaCy not installed | Run `python -m spacy download en_core_web_sm` or ignore (optional dependency) |
| `API returns "model not loaded"` | No SAE checkpoint found | Train an SAE first, or use the pre-trained case study model at `models/case_study/sae_final.pt` |
| `Next.js: Module not found` | Node deps not installed | Run `cd dashboard && npm install` |
| Dashboard shows "Loading..." forever | API not running on port 8000 | Start the API with `uvicorn src.api.main:app --reload --port 8000` |
| `docker compose: command not found` | Docker not installed | Install Docker Desktop, or run without Docker (use manual setup above) |
| `Error 28: No space left on device` | HF cache filled disk | Clear cache: `rm -rf ~/.cache/huggingface/datasets` (Linux) or delete `%USERPROFILE%\.cache\huggingface` (Windows)

## Project Structure

```
mini-sae-trainer/
├── configs/              # YAML hyperparameter configs
│   ├── collect.yaml
│   ├── train_small.yaml
│   ├── train_medium.yaml
│   ├── train_large.yaml
│   ├── analyze.yaml
│   └── api.yaml
├── dashboard/            # Next.js 14 frontend
│   ├── app/              # 4 routes
│   │   ├── page.tsx      # Feature Explorer (sort + paginate)
│   │   ├── features/
│   │   │   └── [id]/
│   │   │       └── page.tsx  # Feature Detail
│   │   ├── prompt/
│   │   │   └── page.tsx  # Prompt Analyser
│   │   └── search/
│   │       └── page.tsx  # Feature Search
│   └── components/       # Shared UI
│       ├── TokenHeatmap.tsx
│       ├── FeatureCard.tsx
│       ├── ActivationHistogram.tsx
│       └── LoadingSkeleton.tsx
├── data/                 # Activations (HDF5, gitignored)
├── docs/
│   ├── theory.md         # SAE theory with LaTeX
│   └── feature_case_studies.md  # Interpretability analysis
├── notebooks/
│   └── 03_feature_case_studies.ipynb  # Interactive exploration
├── models/               # SAE checkpoints (gitignored)
├── results/              # Analysis output (gitignored)
├── src/
│   ├── api/
│   │   └── main.py       # FastAPI (7 endpoints)
│   ├── analysis/
│   │   ├── pipeline.py   # Orchestrator
│   │   ├── top_activations.py
│   │   ├── prompt_activations.py
│   │   ├── auto_label.py
│   │   └── geometry.py
│   ├── data/
│   │   ├── collect_activations.py  # HDF5 extraction pipeline
│   │   ├── hdf5_dataset.py         # PyTorch Dataset
│   │   └── corpus.py               # TinyStories loader
│   ├── sae/
│   │   ├── model.py      # SparseAutoencoder
│   │   ├── train.py      # Training loop
│   │   ├── loss.py       # Loss functions
│   │   └── resampling.py # Dead feature handling
│   ├── utils/
│   │   ├── config.py     # YAML config loader
│   │   ├── seeds.py      # Reproducibility
│   │   └── logging.py    # Logging setup
│   └── model.py          # ActivationExtractor
├── tests/
│   └── test_api.py       # 11 pytest tests
├── Dockerfile.api        # Multi-stage Python build
├── Dockerfile.dashboard  # Multi-stage Node build
├── docker-compose.yml    # API + Dashboard
├── Makefile              # GNU/Linux commands
└── Makefile.ps1          # Windows PowerShell commands
```

### Live Demo

A live demo is available at: **https://mini-sae-trainer.vercel.app**

*Note: the demo runs with precomputed static JSON data. The FastAPI backend is not required for the dashboard's core browsing functionality.*

## Configuration

All hyperparameters live in `configs/*.yaml`. Training defaults:

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

## Feature Case Studies

The trained SAE (d_sae=128, 94.3% EV) revealed 8 interpretable feature categories:

| Feature | Label | Monosemanticity | Concept |
|---|---|---|---|---|
| 40 | `"time"` | High | Narrative time marker / story formula opening |
| 95 | `","` | High | Dialogue attribution comma |
| 64 | `"end"` | High | Story ending / temporal conclusion |
| 120 | `"mouse"` | Concept-level high | Animal character introduction |
| 87 | `"down"` | Low-Medium | Directional descriptions |
| 53 | `"said"` | High | Narration verb |
| 98 | `"."` | High | Sentence-ending period |
| 71 | `","` | Medium | Adjective list comma |
| 47 | `"mer"` | Low | Character name suffix |

See [docs/feature_case_studies.md](docs/feature_case_studies.md) for full analysis with activating contexts.

## Key Design Decisions

- **No SAELens dependency**: SAE architecture and training written from scratch in raw PyTorch.
- **HDF5 with RAM caching**: Chunked gzip-compressed HDF5 + `cache_in_ram=True` gives ~120 it/s on CPU vs ~0.09 it/s without.
- **Dual Makefiles**: `Makefile` (GNU/Linux) + `Makefile.ps1` (Windows PowerShell) for cross-platform convenience.
- **Static export fallback**: Dashboard can run as a fully static Next.js export if the API is unavailable.

## Results

On TinyStories (50K tokens, d_sae=128, 500 steps):

```
Final Loss:       6.60
L2 Loss:          1.88
L1 Loss:          0.00047
L0 Norm:          7.84 active features per token
Explained Variance: 94.3%
```

## Deployment

### Hugging Face Spaces

1. Create a Space with Docker runtime
2. Push the repo and set `docker-compose.yml` as the Space Dockerfile
3. The dashboard will be available at `https://your-username-mini-sae-trainer.hf.space`

### Manual Docker

```bash
docker compose up --build -d
```

## Citation

```bibtex
@software{mini_sae_trainer_2025,
  title = {Mini SAE Trainer: From-Scratch Sparse Autoencoders for GPT-2},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-username/mini-sae-trainer}
}
```

## References

- Elman, J. L. (1990). *Finding Structure in Time*. Cognitive Science.
- Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic.
- Templeton, A., et al. (2024). *Scaling Monosemanticity*. Anthropic.
- Gao, L., et al. (2024). *Scaling and Evaluating Sparse Autoencoders*. OpenAI.
- TinyStories: *https://huggingface.co/datasets/roneneldan/TinyStories*
#   s a e - i n t e r p - d a s h b o a r d  
 