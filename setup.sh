#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "=== Mini SAE Trainer — Setup ==="

# --- Python virtual environment ---
if [ ! -d ".venv" ]; then
  echo "[1/6] Creating Python virtual environment..."
  python3.11 -m venv .venv
fi

source .venv/bin/activate

echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "[3/6] Downloading spaCy model..."
python -m spacy download en_core_web_sm 2>/dev/null || true

# --- Data directories ---
echo "[4/6] Creating data directories..."
mkdir -p data models results notebooks tests
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "       Created .env from .env.example — edit it to set your WANDB_API_KEY"
fi

# --- HuggingFace cache (pre-download GPT-2 + TinyStories) ---
echo "[5/6] Pre-downloading GPT-2 model and TinyStories dataset..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

print('Downloading GPT-2 tokenizer...')
AutoTokenizer.from_pretrained('gpt2')
print('Downloading GPT-2 model...')
AutoModelForCausalLM.from_pretrained('gpt2', output_hidden_states=True)
print('Downloading TinyStories dataset...')
load_dataset('roneneldan/TinyStories', split='train', trust_remote_code=True)
print('Done.')
"

# --- Node.js dependencies ---
echo "[6/6] Installing Node.js dependencies..."
cd dashboard
npm install
cd "$ROOT_DIR"

echo ""
echo "=== Setup complete ==="
echo "Activate the environment:  source .venv/bin/activate"
echo "Run activation collection: python -m src.data.collect_activations --config configs/collect.yaml"
echo "Run SAE training:         python -m src.sae.train --config configs/train_medium.yaml"
echo "Run analysis pipeline:    python -m src.analysis.pipeline --config configs/analyze.yaml"
echo "Start API server:         uvicorn src.api.main:app --reload --port 8000"
echo "Start Next.js dashboard:  cd dashboard && npm run dev"
