#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for s in {2015..2026}; do
  python3 scripts/update_cbbd.py --season "$s" --players --games
done

find data/raw -maxdepth 4 -type f -name "*.csv" | head -n 50
