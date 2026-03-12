#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# All plays are required for RAPM (substitutions). This is large.
for s in 2025 2026; do
  python3 scripts/update_cbbd.py --season "$s" --plays-all --skip-existing --sleep 0.2
done
