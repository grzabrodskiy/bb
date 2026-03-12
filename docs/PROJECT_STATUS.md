# Project Status

Last updated: 2026-03-08

## Latest Update (2026-03-08)
- NBA success RAPM pipeline now supports one pooled model with entrant-cohort scoring over a year range:
  - script updated: `scripts/train_nba_success_rapm_model.py`
  - latest run: single model fit + scored entrant cohort `exit years 2018..2023` (`n=9,640`)
  - updated outputs:
    - `data/processed/nba_success_rapm_predictions_exit_2018_2023.csv`
    - `data/processed/nba_success_rapm_model_metrics.csv`
    - `data/processed/nba_success_rapm_targets_best2.csv`
    - `data/processed/nba_success_rapm_holdout_actual_vs_predicted_2017.csv`
- NBA success dashboard popup UX updates:
  - season-tab labels now use team abbreviations (league source is still conveyed by tab color).
  - `Birth Year (est)` removed from popup measurements.
  - age values in popup measurements are now displayed as integers.
- NBA season-history fallback was extended in dashboard generation:
  - script updated: `scripts/generate_nba_impact_actual_vs_pred_dashboard.py`
  - if `player_stats_export.csv` is missing a player/season, history now backfills from Basketball-Reference advanced seasons.
  - this restores missing historical NBA seasons for players like Josh Okogie in popup season tabs.
- Position-factor dashboard measurement policy was tightened:
  - script updated: `scripts/generate_nba_position_factor_dashboard.py`
  - combine is treated as primary source and crafted as fallback through unified `measurement_*` columns.
  - combine/crafted are no longer modeled as separate competing measurement features.
- Regenerated dashboards:
  - `data/processed/viz/nba_impact_dashboard_real_vs_predicted_2017.html`
  - `data/processed/viz/nba_position_factor_dashboard.html`
- Added a common-sense diagnostics artifact:
  - `data/processed/nba_success_common_sense_checks_2018_2023.txt`
  - includes target coverage limits, measurement sign stability checks, and source-coverage findings.
- Visualization index is now production-only:
  - removed experimental RAPM/legacy dashboard links from app home.
  - app home now contains only 3 dashboards:
    - NBA Performance Prediction
    - Success Drivers by Position
    - Pick Number Prediction
- NBA Success dashboard now supports year switching across generated holdout pages (`2018`, `2019`, `2022`) via selector (index + in-dashboard).
- NBA Success popup tabs were simplified:
  - removed `Bio` tab.
  - `Age` moved to `Measurements`.
  - team and position are shown per season tab.
  - position text in popup is expanded to full words (for example `PG` -> `Point Guard`).
- Success Drivers by Position dashboard now keeps `Positive` and `Negative` sections and additionally includes:
  - `Measurement Factors`
  - `Improvement Factors` (year-over-year / peak-gap features)

- Core `NBA Success Dashboard` (`2022` holdout) popup is now fully season-tab based:
  - each college and NBA season appears as its own tab (`College YYYY/YY`, `NBA YYYY/YY`)
  - tabs are color-coded by source (college vs NBA)
  - `Bio` and `Measurements` remain separate tabs.
- Popup payload now includes season-history fields:
  - `college_seasons`, `college_team_history`, `college_position_history`
  - `nba_seasons`, `nba_team_history`, `nba_position_history`
- Popup sizing is stabilized across tabs (keeps largest tab height), so modal no longer resizes while switching tabs.
- Measurement fallback/imputation was strengthened for dashboard display:
  - more players now show filled wingspan/reach-related values even when combine rows are missing.
- New core dashboard added:
  - `data/processed/viz/nba_position_factor_dashboard.html`
  - purpose: show the strongest positive/negative factors for NBA success by position (`Guard`, `Forward`, `Center`) using the existing NBA-success modeling dataset.
- Index updated to include the new position-factor dashboard and refresh cache busting for the core NBA Success page.

## Summary
- NCAA CBBD ingest pipeline is active.
- 2025 plays are fully downloaded.
- 2026 plays are partially downloaded and resumable with skip-existing mode.
- 2026 refresh is currently blocked by CBBD monthly API quota (`429`).
- RAPM pipeline has been refactored for laptop execution and now uses event-based possession estimation when official possession fields are missing.
- EPM-lite pilot is implemented by blending RAPM with a box-score prior.
- Multi-technique RAPM experiment pipeline is implemented for pure-stat comparisons.
- Standalone HTML visualization pipeline is implemented for model explainability views.
- Visualization folder now includes a styled `index.html` landing page for easier local navigation.
- Visualization views now include enriched player box-score context (totals, per-40, efficiency, ratings, win shares) sourced from season player stats.
- Visualization views now also surface ESPN player bio context (position, height/weight, age, DOB, birthplace, jersey) when available.
- All generated visualization pages now include a built-in "Back to Index" button linking to the local dashboard landing page.
- Visualization index now uses season + population selectors to route into the corresponding dashboards (currently season `2025`, groups `all`/`freshmen`).
- Visualization index is now organized into `Core Dashboards` (`Draft Pick`, `NBA Success`) with all other screens grouped under `Experimental`.
- Core index routing is set to the `2022` NBA Success page (`nba_impact_dashboard_real_vs_predicted_2022.html`) for ongoing model iteration.
- Core `2022` NBA Success holdout table now enforces cohort integrity:
  - removes forward-leak rows from later draft classes (for example non-2022 entrants appearing via earlier college seasons)
  - cohort definition now follows 2022 class entrants:
    - drafted in 2022 are included (even if first NBA playing season is later, e.g. injury delay)
    - undrafted entries are included for the 2022/23 entry window.
  - popup bio start years now display as season ranges (for example `2022/23`) to avoid year-format ambiguity.
  - predicted RAPM display values are now quantile-calibrated to holdout observed RAPM distribution to reduce one-sided (all-negative) display bias.
- Core `2022` NBA Success dashboard now uses label-based success logic:
  - success rule: exact RAPM label match
  - label bands:
    - `fantastic >= 3.0`
    - `excellent [0.75, 3.0)`
    - `good [-0.25, 0.75)`
    - `average [-1.0, -0.25)`
    - `bad < -1.0`
  - also shows RAPM diagnostics side-by-side (`Actual RAPM`, `Pred RAPM`, `Abs RAPM Err`)
  - latest 2022 snapshot:
    - RAPM coverage: `34/43`
    - RAPM MAE/RMSE: `1.196 / 1.592`
    - RAPM hits (`|err| <= 1.0`): `19/34` (`55.9%`)
    - label hits (exact): `15/34` (`44.1%`)
    - label hits (within 1): `27/34` (`79.4%`)
  - visible UI is now label/RAPM-only (rank columns/cards removed from the dashboard view)
  - miss explanations are now conflict-oriented (what overprojected/underprojected and which opposing signal likely drove miss), with special boundary text for near-threshold misses.
  - player rows are now clickable and open a modal popup with separate tabs for:
    - `College Stats`
    - `NBA Stats`
    - `Bio`
  - popup abbreviation help now uses click-to-open explanations on driver tags (not hover-only).
  - measurement popup now uses fallback sources so values are not blank when combine rows are unavailable:
    - height: combine -> bio -> external model DB
    - weight: combine -> bio -> external model DB
    - age: bio -> external model DB
    - wingspan: combine -> crafted -> other source -> estimated
    - standing reach: combine -> other source -> estimated
    - hand/body-fat fields: combine -> estimated
    - popup now shows measurement values only (source labels removed)
  - official combine ingest refreshed for draft years `2015..2023` from NBA stats endpoint.
  - crafted fallback ingest added:
    - `data/raw/external/craftednba/player_traits_length.csv`
  - current core 2022 holdout popup measurement coverage:
    - height: `42/43`
    - weight: `41/43`
    - wingspan: `42/43`
    - standing reach: `42/43`
- Tenure handling update:
  - training now uses tenure-neutral all-years profile engineering so players are not directly penalized for staying more/less years in college.
  - direct tenure/class-year penalty features are removed from model inputs; evaluation relies on production/efficiency/profile quality signals instead.
  - robust variant now also supports dropping one worst season (when player has at least 3 college seasons) via trimmed multi-year averages.
- True RAPM-peak (`best-2-year`) pipeline remains available under experimental dashboards (`2017` holdout):
  - source files:
    - `data/processed/nba_success_rapm_holdout_actual_vs_predicted_2017.csv`
    - `data/processed/nba_success_rapm_predictions_draft_2025.csv`
  - trainer supports feature profiles (`full`/`no_ext`/`core`) and threshold-focused objectives.
  - trainer now includes tenure-neutral multi-year college profiling:
    - `career_avg_*`, `trend_*`, `peak_gap_*`
    - robust `career_trim1_avg_*` (drop one worst season when >=3 college seasons)
  - direct tenure/class-year penalty fields are excluded from model input selection.
  - table includes drafted `NBA Team` via draft-history merge.
- Modern RAPM source exploration is in progress:
  - added `scripts/download_tbd_rapm_history.py` (thebasketballdatabase page parser)
  - generated `data/raw/external/rapm_history/rapm_history_regular_tbd_2022scope.csv` with season range up to `2024`
  - known gap: many recent 2022 rookies currently have empty one-year RAPM rows on that source, so strict 2022 best-2-year RAPM coverage is still incomplete.
- Dashboard 1 detail panel now uses tabs to separate `Stat Measures`, `Bio`, and `Box Score Context`.
- Visualization set now includes two additional diagnostic views:
  - `Reliability Frontier` (impact vs possessions + agreement weighting)
  - `Consensus Range` (cross-model rank spread strips per player)
- Free player bio ingest is available via ESPN public athlete endpoint using CBBD `athlete_source_id`.
- CBBD plays storage is now space-optimized with gzip compression (`plays.csv.gz`) and loader/downloader compatibility.
- NBA draft history labels are now available locally for model supervision (`2015`-`2025` picks).
- Additional free external draft datasets have been downloaded for feature expansion:
  - RealGM-based draft model tables (`model_db`, `draft_db`, `draft_db_nba`)
  - NBA combine/drill historical archives (anthropometrics, strength/agility, spot-up, non-stationary drills, draft history).
- Additional free NBA impact datasets have been downloaded from `nbarapm.com`:
  - endpoints: `TimedecayRAPM`, `current_comp`, `player_stats_export`, `mamba`, `DARKO`, `lebron`, `raptor`
  - includes a current RAPM target proxy table:
    - `data/processed/nba_rapm_target_current_timedecay.csv` (`505` rows)
  - includes a 2026 NBA player-stat subset:
    - `data/processed/nba_player_stats_export_2026.csv` (`500` rows)
- Additional free NBA advanced player tables have been downloaded from Basketball-Reference:
  - `data/raw/nba/bref/player_advanced_2010_2026.csv` (`11,140` rows)
- Historical season-level RAPM data is now available from a free public source:
  - source: `https://basketball-analytics.gitlab.io/rapm-data/`
  - downloaded seasons: `1996-97` through `2018-19` (regular season + playoffs)
  - outputs:
    - `data/raw/external/rapm_history/rapm_history_all.csv`
    - `data/raw/external/rapm_history/rapm_history_regular.csv`
    - `data/raw/external/rapm_history/rapm_history_playoffs.csv`
- NBA draft hurdle-model pipeline is now implemented:
  - stage 1: predict `drafted` probability
  - stage 2: predict `pick_number` conditional on drafted
  - produces season-level prospect predictions (latest: season `2026`).
  - latest holdout (test year `2025`) from local run:
    - drafted classifier: `ROC-AUC=0.9609`, `PR-AUC=0.2785`, `Brier=0.1095`
    - pick regression (drafted-only): `MAE=14.607`, `RMSE=16.865`
- Draft model visualization page is now available in local viz hub:
  - `data/processed/viz/nba_draft_predictor_season_2026.html`
  - includes holdout summary metrics, top-10 summary, interactive filters, and scatter/table explorer.
  - includes a free-text executive summary that explains model setup, observed quality, and interpretation caveats.
- Draft diagnostics visualization page is now available in local viz hub:
  - `data/processed/viz/nba_draft_dashboard1_real_vs_predicted_2025.html`
  - compares real 2025 picks to model expected picks
  - includes model coverage rate, pick-error diagnostics, and real-vs-expected scatter.
  - now highlights successful vs unsuccessful covered predictions with row backgrounds and a top success counter (missing-data rows excluded).
  - now includes a per-row `Miss Reason` explanation column with plain-English, player-specific details (rank gap, draft chance, likely pick if drafted, expected-pick math, and playing-time context), also shown in point hover text.
  - table now uses concise miss-only tags (`Miss Tag`) such as `Low play time: 198m`; tags are intentionally blank for successful rows and missing-data rows.
  - miss tags are now sourced from a dedicated miss-audit output with player-specific model-driver causes (for example low usage, low minutes, turnover/rebounding profile, or late expected pick), rather than generic rank-only labels.
  - miss-tag prioritization now avoids outcome-only wording and favors direct driver statements (for example `Older prospect profile: 5y since first season`).
  - holdout coverage export is now produced directly by `scripts/train_nba_draft_predictors.py` (`data/processed/nba_draft_holdout_2025_actual_top60_with_model_coverage.csv`) to keep dashboard inputs in sync with the latest model run.
  - Unicode name normalization now transliterates Cyrillic characters (including `ё`) in draft/feature matching; this fixes previously missed joins such as `Egor Dёmin` vs `Egor Demin` and now marks him as covered in dashboard1.
- NBA impact dashboard1-style diagnostics page is now available in local viz hub:
  - `data/processed/viz/nba_impact_dashboard1_real_top60_2022.html`
  - focuses on real top-60 picks in the 2022 holdout cohort
  - includes hit/miss shading and miss tags using profile signals (minutes, efficiency, usage, turnover/activity profile).
  - now uses explicit rank labels (`Actual Impact Rank`, `Pred Impact Rank`) and includes per-player current NBA RAPM columns (`NBA RAPM`, `RAPM Rank`) from `TimedecayRAPM` when matched.
  - miss tags are now fully performance-based (no model-wording fallbacks).
  - miss explanations are now short stat-only reasons (`Miss Detail`) with numeric drivers (for example minutes, TS, usage role, turnovers, playmaking/rebounding/defensive activity).
  - page now includes a `Column Definitions` section that explains how `Actual Impact Rank`, `Pred Impact Rank`, and `Abs Err` are computed.
  - now includes explicit `Drivers (+)` and `Drivers (-)` columns plus `Miss Explanation` text for each miss.
  - now includes `Pred Rank (All Players)` (full season pool rank, can be `>60`) in addition to drafted-eval rank used for hit/miss.
  - miss rule shown in-page: covered row with `abs rank error > 10`.
- Top-60 miss recovery investigation (real picks `<=60` but model rank `>60`) has been run:
  - covered real 2025 picks in model universe: `45`
  - baseline captured in top-60 expected board: `24`
  - ranking-formula-only sweeps did not improve captured count beyond `24`
  - model hyperparameter sweeps did not improve captured count beyond `24`
  - pure-stats post-hoc board adjustment improved captured count to `26` (net `+2`) with trade-off:
    - rescued: `Max Shulga`, `Chaz Lanier`, `Alijah Martin`
    - dropped: `Ace Bailey`
  - reproducible script:
    - `scripts/analyze_top60_miss_recovery.py`
  - generated outputs:
    - `data/processed/draft_rank_formula_sweep_2025.csv`
    - `data/processed/draft_model_hyperparam_sweep_2025.csv`
    - `data/processed/draft_posthoc_rerank_boost_sweep_2025.csv`
    - `data/processed/draft_holdout_2025_boosted_rerank_analysis.csv`
    - `data/processed/draft_holdout_2025_boosted_rerank_summary.txt`
- Draft model has been upgraded with additional feature engineering and external draft-feature enrichment:
  - external features merged from `data/raw/external/nba_draft_model/model_db.csv`:
    - exact season matches: `736`
    - name-based backfill matches: disabled by default (`--external-backfill-max-gap 0`)
    - exact school-mismatch rows are now dropped to avoid false same-name joins
  - added player lifecycle features (`first_seen_season`, `years_since_first_seen`, first-year/upperclass flags)
  - added engineered features:
    - log-volume, age/tier indicators, assist-to-turnover, size proxy
    - season-relative percentile features for key production/efficiency fields
    - key missingness indicators for bios/external features
    - team-season context features from CBBD games (`team_win_pct`, margin, ELO context, conference-game rate)
  - updated training strategy:
    - classifier negative downsampling (ratio `20:1`)
    - recency weighting by season for both classification and pick regression
  - latest holdout (test year `2025`) after upgrade:
    - drafted classifier: `ROC-AUC=0.9768`, `PR-AUC=0.3595`, `Brier=0.0340`
    - pick regression (drafted-only): `MAE=10.369`, `RMSE=12.849`
    - draft-board capture:
      - top-60 by drafted probability: `20/45`
      - top-60 by expected pick board: `22/45`
    - note: calibration/error improved, but expected-pick top-60 capture fell vs prior (`24/45`), so ranking objective still needs targeted work.
  - top-board sanity check after stricter external joins:
    - `Charles Bediako` moved from rank `2` to rank `1508` for season `2026`
    - indicates previous name-backfill leakage has been removed
- Added a clean-board filter workflow for 2026 draft rankings using only local pure-stat fields:
  - script: `scripts/filter_draft_predictions.py`
  - default rule:
    - require non-empty conference
    - `minutes >= 300`
    - `years_since_first_seen <= 4`
  - latest output:
    - `data/processed/nba_draft_predictions_season_2026_clean_board.csv` (`2,752` rows)
    - `data/processed/nba_draft_predictions_season_2026_clean_board_top30.csv`
  - note: `age` is currently missing for season `2026` rows, so legal age-based draft eligibility checks are not yet enforceable from local data.
- Latest 2025 variant leaderboards were refreshed in one all-formula run (`--top-n 100`).
- Additional 2025 variants rerun completed with forced lambda `1000` (`--ridge 1000 --ridge-grid 1000`) for stronger regularization comparison.
- Additional 2025 variants rerun completed with forced lambda `4000` (`--ridge 4000 --ridge-grid 4000 --top-n 30`) for high-regularization comparison.
- Additional 2025 variants rerun completed for freshmen-only output with forced lambda `4000` (`--player-filter freshmen --ridge 4000 --ridge-grid 4000 --top-n 30`).
- Coverage QA note (2025 Duke / Cooper Flagg):
  - Flagg appears in `37/39` Duke games in `plays` `on_floor` data.
  - Flagg season player stats report `games=37`.
  - Current evidence indicates Flagg ranking is not driven by missing Duke game downloads.
- New-joiner NBA impact prediction pipeline is now implemented (college -> RAPM-family NBA proxy):
  - script: `scripts/train_nba_new_joiner_impact_model.py`
  - target definition:
    - best-2-NBA-season impact proxy from free RAPM-family datasets:
      - `LEBRON`, `DARKO dpm`, `MAMBA`, `RAPTOR`, `BRef BPM`, `BRef WS/48`
    - per-metric z-score + mean composite (`nba_impact_target_z`)
    - requires at least `2` available target metrics for training rows
  - latest holdout (draft year `2022`) metrics:
    - selected model (RAPM-hit-focused): `rf_d14_l2`
    - proxy target quality: `RMSE=0.7313`, `MAE=0.5636`, `Spearman=0.3236`
    - RAPM-aligned quality (`TimedecayRAPM` matched subset):
      - `RAPM_MAE=1.1600`, `RAPM_RMSE=1.609`
      - `RAPM hits (|err|<=1.0)=21/34` (`61.8%`)
    - rank quality:
      - `MAE_rank=11.30`, `within5=17/43`, `within10=21/43`
  - quality fixes applied in latest run:
    - drafted-row dedupe guard (`season + name + pick`, keep highest-minute row)
    - cumulative multi-year `best_*` college features added (best season to date across all college years)
    - explicit feature weighting for NBA-transferable stat/bio/measurement signals
    - model sweep expanded with deeper RF/ExtraTrees variants
    - model selection now supports RAPM-aligned objectives (`rapm_mae`, `rapm_rmse`, `rapm_hit100`)
  - latest predicted new-joiner board output:
    - `data/processed/nba_new_joiner_impact_predictions_draft_2025.csv` (`46` rows)
  - auxiliary outputs:
    - `data/processed/nba_impact_targets_window2.csv`
    - `data/processed/nba_new_joiner_impact_model_metrics.csv`
    - `data/processed/nba_new_joiner_impact_model_report.txt`
    - `data/processed/nba_new_joiner_impact_holdout_actual_vs_predicted_2022.csv`
  - holdout export now contains both rank scopes:
    - `pred_rank_drafted` (covered drafted cohort; used for eval hit/miss)
    - `pred_rank_all_players` (all players in holdout season)
  - visualization:
    - `data/processed/viz/nba_impact_dashboard_real_vs_predicted_2022.html`
    - `data/processed/viz/nba_impact_dashboard1_real_top60_2022.html`
- NBA success RAPM-peak pipeline is now implemented (college -> best-2-year RAPM peak):
  - script: `scripts/train_nba_success_rapm_model.py`
  - target definition:
    - mean of each player's top 2 post-draft regular-season RAPM values
    - RAPM source: `data/raw/external/rapm_history/rapm_history_regular.csv`
  - feature scope:
    - college stats and engineered profile features
    - bios already in training table (`age`, `height`, `weight`)
    - combine measurements merged where available (`height w/o shoes`, `weight`, `wingspan`, `standing reach`, hand size, body fat)
  - latest holdout run (`test-draft-year=2017`):
    - selected model: `gbr_03_d2`
    - `RAPM_RMSE=1.1955`, `RAPM_MAE=0.8334`, `Spearman=-0.0225`
    - hit rates: `Hit<=0.75=26/43`, `Hit<=1.0=30/43`
    - rank quality: `MAE_rank=14.88`, `within5=9/43`, `within10=16/43`
  - outputs:
    - `data/processed/nba_success_rapm_targets_best2.csv`
    - `data/processed/nba_success_rapm_model_metrics.csv`
    - `data/processed/nba_success_rapm_predictions_draft_2025.csv`
    - `data/processed/nba_success_rapm_holdout_actual_vs_predicted_2017.csv`
    - `data/processed/nba_success_rapm_model_report.txt`
  - visualizations:
    - `data/processed/viz/nba_success_dashboard_real_vs_predicted_2017_rapm.html`
    - `data/processed/viz/nba_success_dashboard1_top60_2017_rapm.html`
  - latest rerun (current files on disk):
    - `gbr_03_d2` selected with `RAPM_RMSE=1.1955`, `RAPM_MAE=0.8334`
    - hit rates: `Hit<=0.75=26/43`, `Hit<=1.0=30/43`

## Current Data Coverage
- `data/raw/cbbd/plays/season=2025`
  - API teams: `364`
  - local non-empty team files: `364`
  - missing: `0`
  - storage format: compressed `plays.csv.gz`
- `data/raw/cbbd/plays/season=2026`
  - API teams: `365` (last successful team-list snapshot)
  - local non-empty team files: `72`
  - missing vs last snapshot target: `293`
  - note: current API team-list calls return `429` until quota resets
  - storage format: compressed `plays.csv.gz`
  - quick EPM-lite snapshot run available (partial data only):
    - `data/processed/epm_lite_top20_season_2026.csv`
- `data/raw/cbbd/lineups/season=2026`
  - local non-empty team files: `0`
  - note: lineups endpoint currently blocked by monthly quota (`429`)
- `data/raw/cbbd/lineups/season=2025`
  - API teams: `364`
  - local non-empty team files: `7` (partial)
  - note: current RAPM flow does not require this folder
- `data/raw/espn/player_bios/season=2025`
  - output rows: `9,788`
  - successful fetches: `9,766`
  - non-ok rows: `22` (primarily ESPN `404` athlete IDs)
- `data/raw/nba/draft/nba_draft_history_2015_2025.csv`
  - rows: `653`
  - draft years: `2015`-`2025`
  - unique key (`draft_year`, `pick_overall`) duplicates: `0`
- `data/raw/nba/bref/player_advanced_2010_2026.csv`
  - rows: `11,140`
  - seasons: `2010`-`2026`
  - source: Basketball-Reference player advanced season tables
- `data/raw/external/nba_draft_model/`
  - `draft_db.csv` (rows: `1,292`, cols: `161`)
  - `model_db.csv` (rows: `1,303`, cols: `161`, seasons `2008-09`..`2023-24`)
  - `draft_db_nba.csv` (rows: `1,318`, cols: `164`)
- `data/raw/external/nba_stats_draft/`
  - downloaded and extracted archives:
    - `history` (draft history by year)
    - `antro` (combine anthropometrics)
    - `strengthagility` (combine strength/agility)
    - `spotup` (spot-up shooting drill)
    - `nonstationary` (movement shooting drill)
- `data/raw/external/nbarapm/`
  - downloaded endpoint snapshots with csv/json + manifest:
    - `TimedecayRAPM` (`505` rows)
    - `current_comp` (`524` rows)
    - `player_stats_export` (`2,500` rows)
    - `mamba` (`4,853` rows)
    - `DARKO` (`15,035` rows)
    - `lebron` (`8,622` rows)
    - `raptor` (`12,500` rows)
  - manifest:
    - `data/raw/external/nbarapm/manifest.csv`

## What Was Implemented
- Incremental play download support:
  - `scripts/update_cbbd.py` now supports `--skip-existing`.
  - Existing non-empty team files are skipped for `--team` and `--plays-all`.
  - `scripts/pull_cbbd_plays_2025_2026_allteams.sh` now defaults to skip-existing mode.
  - Existing non-empty lineup files are now skipped for `--lineups` and `--lineups-all` when `--skip-existing` is used.
  - Skip-existing detection now also treats compressed files as existing data (`plays.csv.gz` / `lineups.csv.gz`).
  - Added season `2026` resumable full-refresh script:
    - `scripts/pull_cbbd_2026_resume_all.sh`
    - executes `games/players`, then resumable plays, then resumable lineups.

- RAPM runtime/engineering changes:
  - Reduced memory pressure by loading only required play columns.
  - Deduplication performed during load.
  - Aggregation over unique lineup stints before solving ridge system.
  - Added `--max-players` player-pool cap for laptop-safe linear solve.
  - Added event-based possession estimation fallback:
    - `FGA - ORB + TO + 0.44 * FTA`
    - used when official possession columns are not available
  - Added player metadata enrichment (`player_name`, `player_team`) in RAPM output.
  - Added D1 conference-tier filters:
    - `--team-tier`: `all`, `high`, `mid`, `high_mid`
    - `--tier-filter-mode`: `both` or `team`
  - RAPM output now includes `player_conference` when metadata exists.
  - Optimized lineup extraction from `on_floor` using parser caching and tuple-based iteration for faster laptop execution.
  - Increased default `--min_possessions` to `200` for stability.
  - Notebook now reuses shared script logic.
  - `load_plays()` now reads both `plays.csv` and `plays.csv.gz` (prefers uncompressed when both exist).

- EPM-lite workflow:
  - Script: `scripts/compute_epm_lite_from_plays.py`
  - Steps:
    - Compute RAPM from play stints.
    - Build a box prior from per-40 and efficiency box stats.
    - Shrink prior by minutes/games reliability.
    - Blend RAPM and prior with possession-based weights.
  - Main output columns:
    - `epm_lite`, `rapm`, `box_prior`, `rapm_weight`, `prior_minutes_weight`

- RAPM variants workflow:
  - Script: `scripts/compute_rapm_variants.py`
  - Techniques currently implemented:
    - Fixed-lambda ridge RAPM
    - Game-level CV lambda selection
    - Robust Huber RAPM (IRLS)
    - Bayesian RAPM with stat-only box prior mean
    - Close-game (garbage-time filtered) RAPM
    - Home-court-adjusted RAPM
    - Offense/Defense RAPM decomposition
    - Empirical-Bayes possessions shrink
  - Output player filters:
    - `--player-filter all` (default)
    - `--player-filter freshmen` (first-seen season inferred from local historical player-season stats)
  - Primary outputs:
    - `data/processed/rapm_variants_season_{SEASON}.csv`
    - `data/processed/rapm_variants_summary_season_{SEASON}.txt`
    - filtered runs include suffixes (for example `_freshmen`)

- Visualization workflow:
  - Core module: `src/dashboard/generate_athlete_viz_html.py`
  - CLI wrapper script: `scripts/generate_athlete_viz_html.py`
  - Input: any variants-style model CSV with player + RAPM columns.
  - Automatically enriches records with player season stats from `data/raw/cbbd/players/season={SEASON}/player_season_stats.csv` (season inferred from filename or set with `--season`).
  - Also enriches records with ESPN bio data from `data/raw/espn/player_bios/season={SEASON}/player_bios.csv` when present.
- Output: seven standalone HTML views in `data/processed/viz/`
  - evidence cards leaderboard
  - archetype scatter map
  - head-to-head comparison lab
  - group explorer (conference/team weighted bars)
  - model agreement matrix (rank consistency)
  - reliability frontier (impact-confidence composite)
  - consensus range (rank spread visualization)

- Free player bio workflow:
  - Script: `scripts/download_espn_player_bios.py`
  - Source: ESPN public athlete endpoint (`sports.core.api.espn.com`)
  - Input join key: CBBD `athlete_source_id`
  - Output: `data/raw/espn/player_bios/season={SEASON}/player_bios.csv`
  - Key fields: height, weight, DOB, birthplace, position, display names

- NBA draft label workflow:
  - Script: `scripts/download_nba_draft_history.py`
  - Source: Basketball-Reference draft pages (`NBA_{YEAR}.html`)
  - Output: `data/raw/nba/draft/nba_draft_history_2015_2025.csv`
  - Columns: `draft_year`, `pick_overall`, `pick_round`, `team_abbr`, `player_name`, `college_name`, `player_bref_id`

- NBA draft predictor workflow:
  - Training-table builder: `scripts/build_draft_training_table.py`
    - aggregates one row per athlete-season from CBBD player stats
    - merges ESPN bios (`height_in`, `weight_lb`, `age`) when available
    - joins draft labels by normalized name + season/draft-year alignment
    - output: `data/processed/nba_draft_training_table.csv`
  - Hurdle trainer: `scripts/train_nba_draft_predictors.py`
    - stage 1: weighted logistic model for `drafted` (pure numpy/pandas)
    - stage 2: ridge regression for `pick_number` on drafted-only rows
    - time-based holdout evaluation by `--test-year`
    - outputs:
      - `data/processed/nba_draft_model_report.txt`
      - `data/processed/nba_draft_predictions_season_{SEASON}.csv`

## Known Issues / Risks
- CBBD provider monthly call quota is currently exhausted (`429 Monthly call quota exceeded`) and blocks remaining 2026 team-level ingest.
- RAPM possession estimates are derived from event heuristics in files without official possession counters.
- Possession heuristics improve stability vs prior proxy approach, but are still an approximation.
- `min_possessions` threshold and player-pool filters still materially affect top-player lists.
- EPM-lite prior is heuristic (not jointly estimated in one regression), so coefficient weights should be treated as tunable.
- Offense/defense decomposition is sensitive to lineup parsing quality and possession approximation.
- Model rankings remain sensitive to player-pool cap and lambda grid choices.

## Resume Commands
- Continue 2026 missing-team download:
```bash
python3 scripts/update_cbbd.py --season 2026 --plays-all --skip-existing --sleep 0.2
```

- Run full resumable 2026 refresh sequence:
```bash
./scripts/pull_cbbd_2026_resume_all.sh
```

- Generate HTML visualization dashboards from model CSV:
```bash
python3 scripts/generate_athlete_viz_html.py --input-csv data/processed/rapm_variants_season_2025_freshmen.csv --top-n 30
```

- Recompute RAPM (current approximation):
```bash
python3 scripts/compute_rapm_from_plays.py --season 2025 --ridge 100 --min_possessions 200 --max-players 3000
```

- Recompute RAPM for D1 high+mid majors only:
```bash
python3 scripts/compute_rapm_from_plays.py --season 2025 --ridge 100 --min_possessions 200 --max-players 3000 --team-tier high_mid --tier-filter-mode both
```

- Compute EPM-lite (all players in current season dataset):
```bash
python3 scripts/compute_epm_lite_from_plays.py --season 2025 --top-n 0
```

- Compute quick 2026 EPM-lite snapshot (current partial ingest):
```bash
python3 scripts/compute_epm_lite_from_plays.py --season 2026 --top-n 20
```

- Compute multi-technique RAPM variants (2025 example):
```bash
python3 scripts/compute_rapm_variants.py --season 2025 --top-n 100
```

- Compute multi-technique RAPM variants for freshmen only (2025 example):
```bash
python3 scripts/compute_rapm_variants.py --season 2025 --player-filter freshmen --ridge 4000 --ridge-grid 4000 --top-n 30
```

- Download free ESPN player bios (height/weight/etc.) for a season:
```bash
python3 scripts/download_espn_player_bios.py --season 2025 --skip-existing --workers 16
```

- Download NBA draft history labels (for drafted/pick supervision):
```bash
python3 scripts/download_nba_draft_history.py --start-year 2015 --end-year 2025
```

- Build draft training table:
```bash
python3 scripts/build_draft_training_table.py --start-season 2015 --end-season 2026 --min-games 5 --min-minutes 100
```

- Train/evaluate hurdle draft model and score next season prospects:
```bash
python3 scripts/train_nba_draft_predictors.py --input-csv data/processed/nba_draft_training_table.csv --test-year 2025 --predict-season 2026
```

- Build cleaner 2026 board from model outputs (D1 conference + minutes + experience window):
```bash
python3 scripts/filter_draft_predictions.py --input-csv data/processed/nba_draft_predictions_season_2026.csv --out-csv data/processed/nba_draft_predictions_season_2026_clean_board.csv --out-top-csv data/processed/nba_draft_predictions_season_2026_clean_board_top30.csv --top-n 30 --min-minutes 300 --max-years-since-first-seen 4
```

- Download NBA RAPM-related datasets from nbarapm.com:
```bash
python3 scripts/download_nbarapm_datasets.py
```

- Analyze and attempt recovery of real top-60 picks ranked outside top-60:
```bash
python3 scripts/analyze_top60_miss_recovery.py --test-year 2025
```

## Documentation Maintenance Rule
- For every code or data-workflow change:
  1. Add a dated entry to `docs/CHANGELOG.md`.
  2. Update `docs/PROJECT_STATUS.md` if current behavior, coverage, or known risks changed.
