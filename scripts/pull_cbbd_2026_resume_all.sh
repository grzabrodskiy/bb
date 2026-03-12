#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Season-wide tables
python3 scripts/update_cbbd.py --season 2026 --games --players

# Team-level tables (resumable)
python3 scripts/update_cbbd.py --season 2026 --plays-all --skip-existing --sleep 0.2
python3 scripts/update_cbbd.py --season 2026 --lineups-all --skip-existing --sleep 0.2
