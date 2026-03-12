#!/usr/bin/env bash
set -euo pipefail

python3 scripts/compute_rapm_variants.py \
  --season 2025 \
  --ridge 4000 \
  --ridge-grid 4000 \
  --player-filter freshmen \
  --top-n 30
