param()

$ErrorActionPreference = "Stop"
$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT_DIR

Write-Host "=== Mini SAE Trainer — Setup ===" -ForegroundColor Cyan

# --- Python virtual environment ---
if (-not (Test-Path ".venv")) {
  Write-Host "[1/6] Creating Python virtual environment..." -ForegroundColor Yellow
  python -m venv .venv
}

.\.venv\Scripts\Activate.ps1

Write-Host "[2/6] Installing Python dependencies..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Write-Host "[3/6] Downloading spaCy model..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm 2>$null

# --- Data directories ---
Write-Host "[4/6] Creating data directories..." -ForegroundColor Yellow
@("data", "models", "results", "notebooks", "tests") | % { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "       Created .env from .env.example — edit it to set your WANDB_API_KEY" -ForegroundColor Yellow
}

# --- HuggingFace cache ---
Write-Host "[5/6] Pre-downloading GPT-2 model and TinyStories dataset..." -ForegroundColor Yellow
python -c @"
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

print('Downloading GPT-2 tokenizer...')
AutoTokenizer.from_pretrained('gpt2')
print('Downloading GPT-2 model...')
AutoModelForCausalLM.from_pretrained('gpt2', output_hidden_states=True)
print('Downloading TinyStories dataset...')
load_dataset('roneneldan/TinyStories', split='train', trust_remote_code=True)
print('Done.')
"@

# --- Node.js dependencies ---
Write-Host "[6/6] Installing Node.js dependencies..." -ForegroundColor Yellow
Set-Location dashboard
npm install
Set-Location $ROOT_DIR

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Activate the environment:  .\.venv\Scripts\Activate.ps1"
Write-Host "Run activation collection: python -m src.data.collect_activations --config configs/collect.yaml"
Write-Host "Run SAE training:         python -m src.sae.train --config configs/train_medium.yaml"
Write-Host "Run analysis pipeline:    python -m src.analysis.pipeline --config configs/analyze.yaml"
Write-Host "Start API server:         uvicorn src.api.main:app --reload --port 8000"
Write-Host "Start Next.js dashboard:  cd dashboard && npm run dev"
