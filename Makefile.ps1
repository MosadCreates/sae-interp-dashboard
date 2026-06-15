param(
  [string]$Target = "help"
)

$COLLECT_CONFIG = "configs/collect.yaml"
$TRAIN_CONFIG   = "configs/train_medium.yaml"
$ANALYZE_CONFIG = "configs/analyze.yaml"
$API_CONFIG     = "configs/api.yaml"

function Invoke-Collect {
  python -m src.data.collect_activations --config $COLLECT_CONFIG
}

function Invoke-Train {
  python -m src.sae.train --config $TRAIN_CONFIG
}

function Invoke-Analyze {
  python -m src.analysis.pipeline --config $ANALYZE_CONFIG
}

function Invoke-Api {
  uvicorn src.api.main:app --reload --port 8000
}

function Invoke-Dashboard {
  Push-Location dashboard
  npm run dev
  Pop-Location
}

function Invoke-All {
  Invoke-Collect
  Invoke-Train
  Invoke-Analyze
}

function Invoke-Clean {
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue @(
    "data/activations.h5",
    "results/*.json",
    "results/geometry/"
  )
}

function Invoke-Test {
  python -m pytest tests/ -v
}

function Invoke-Help {
  @"
Usage: .\Makefile.ps1 -Target <target>

Targets:
  collect     Collect GPT-2 activations from TinyStories
  train       Train the SAE (default: train_medium config)
  analyze     Run feature analysis pipeline
  api         Start FastAPI server on port 8000
  dashboard   Start Next.js dev server
  all         Run collect -> train -> analyze
  clean       Remove generated data and results
  test        Run pytest
"@
}

switch ($Target) {
  "collect"   { Invoke-Collect }
  "train"     { Invoke-Train }
  "analyze"   { Invoke-Analyze }
  "api"       { Invoke-Api }
  "dashboard" { Invoke-Dashboard }
  "all"       { Invoke-All }
  "clean"     { Invoke-Clean }
  "test"      { Invoke-Test }
  default     { Invoke-Help }
}
