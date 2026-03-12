# BB Basketball Data Project

Goal: build a refreshable, source-aware pipeline for NBA, NCAA, and international data.

## Quick Start
1. Create a .env file from the example and set your CBBD API key.
2. Install dependencies.
3. Run the CBBD ingest script.

```bash
cp .env.example .env
pip install -r requirements.txt
python scripts/update_cbbd.py --season 2025 --team "Dayton" --shooting-only --games --players
python scripts/update_cbbd.py --season 2025 --team "Dayton" --lineups
python scripts/update_cbbd.py --season 2025 --lineups-all
python scripts/compute_rapm_from_plays.py --season 2025 --ridge 100 --min_possessions 200 --max-players 3000
python scripts/compute_epm_lite_from_plays.py --season 2025 --top-n 100
python scripts/compute_rapm_variants.py --season 2025 --top-n 100
python scripts/compute_rapm_variants.py --season 2025 --player-filter freshmen --ridge 4000 --ridge-grid 4000 --top-n 30
python scripts/generate_athlete_viz_html.py --input-csv data/processed/rapm_variants_season_2025_freshmen.csv --top-n 30
python scripts/download_espn_player_bios.py --season 2025 --skip-existing --workers 16
./scripts/pull_cbbd_2026_resume_all.sh
```

Visualization generator writes five HTML dashboard variants (`v1`..`v5`) into `data/processed/viz/` and enriches each player with season box-score context (auto-inferred season, or pass `--season YYYY`).

## Project Structure
- configs/: source configuration and season ranges
- data/raw/: raw API responses
- data/staging/: lightly cleaned data
- data/processed/: normalized, model-ready tables
- src/ingest/: one file per source
- src/transform/: normalization and feature build steps
- scripts/: runnable entrypoints

## Data Format
- Raw ingests are stored as CSV for easy use in Google Sheets and Python.

## Data Sources (Free)
- CollegeBasketballData (CBBD) API for NCAA PBP + shot locations
- NBA endpoints and other sources will be added next

## Documentation
- Current status: `docs/PROJECT_STATUS.md`
- Change history: `docs/CHANGELOG.md`
- Maintenance rule: update both files with each material code/data workflow change.
