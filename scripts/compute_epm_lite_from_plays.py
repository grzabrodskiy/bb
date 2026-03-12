from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow direct execution via `python3 scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_rapm_from_plays import (
    attach_player_metadata,
    compute_rapm,
    filter_plays_by_tier,
    load_plays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EPM-lite from CBBD plays by blending RAPM with a box-score prior."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--ridge", type=float, default=100.0, help="Ridge penalty for RAPM solve.")
    parser.add_argument("--min-possessions", type=float, default=200.0, help="Min possessions for RAPM pool.")
    parser.add_argument(
        "--max-players",
        type=int,
        default=3000,
        help="Cap RAPM player pool by possessions (0 disables).",
    )
    parser.add_argument(
        "--free-throw-weight",
        type=float,
        default=0.44,
        help="Free throw possession weight for possession estimate fallback.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on team play files for quick tests.",
    )
    parser.add_argument(
        "--team-tier",
        type=str,
        choices=["all", "high", "mid", "high_mid"],
        default="all",
        help="Filter to D1 conference tiers before RAPM.",
    )
    parser.add_argument(
        "--tier-filter-mode",
        type=str,
        choices=["both", "team"],
        default="both",
        help="When filtering by team-tier: require both teams in tier, or only row team.",
    )
    parser.add_argument(
        "--prior-possessions",
        type=float,
        default=1200.0,
        help="Blend prior strength in possessions: lower uses RAPM more aggressively.",
    )
    parser.add_argument(
        "--prior-std",
        type=float,
        default=2.5,
        help="Target standard deviation for box prior (points/100 possessions).",
    )
    parser.add_argument(
        "--prior-minutes-scale",
        type=float,
        default=700.0,
        help="Minutes scale for prior reliability shrinkage.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="How many rows to write (0 writes full table).",
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.quantile(0.01)
    hi = s.quantile(0.99)
    s = s.clip(lower=lo, upper=hi)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(s), dtype=np.float64), index=s.index)
    return (s - mu) / sd


def build_box_prior(season: int, prior_std: float, prior_minutes_scale: float) -> pd.DataFrame:
    path = Path("data/raw/cbbd/players") / f"season={season}" / "player_season_stats.csv"
    if not path.exists():
        print(f"Box prior skipped; missing player stats file: {path}")
        return pd.DataFrame(columns=["player_id", "box_prior", "prior_minutes_weight"])

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["player_id", "box_prior", "prior_minutes_weight"])

    minutes = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0)
    games = pd.to_numeric(df.get("games"), errors="coerce").fillna(0.0)
    minutes = minutes.clip(lower=0.0)
    games = games.clip(lower=0.0)
    safe_minutes = minutes.replace(0.0, np.nan)

    pts40 = 40.0 * pd.to_numeric(df.get("points"), errors="coerce") / safe_minutes
    ast40 = 40.0 * pd.to_numeric(df.get("assists"), errors="coerce") / safe_minutes
    reb40 = 40.0 * pd.to_numeric(df.get("rebounds.total"), errors="coerce") / safe_minutes
    stl40 = 40.0 * pd.to_numeric(df.get("steals"), errors="coerce") / safe_minutes
    blk40 = 40.0 * pd.to_numeric(df.get("blocks"), errors="coerce") / safe_minutes
    tov40 = 40.0 * pd.to_numeric(df.get("turnovers"), errors="coerce") / safe_minutes
    ts = pd.to_numeric(df.get("true_shooting_pct"), errors="coerce")
    net = pd.to_numeric(df.get("net_rating"), errors="coerce")
    porpag = pd.to_numeric(df.get("porpag"), errors="coerce")
    usage = pd.to_numeric(df.get("usage"), errors="coerce")

    raw = (
        0.33 * _zscore(pts40)
        + 0.18 * _zscore(ast40)
        + 0.10 * _zscore(reb40)
        + 0.17 * _zscore(stl40)
        + 0.14 * _zscore(blk40)
        - 0.21 * _zscore(tov40)
        + 0.13 * _zscore(ts)
        + 0.10 * _zscore(net)
        + 0.10 * _zscore(porpag)
        + 0.05 * _zscore(usage)
    )
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    raw_sd = raw.std(ddof=0)
    if not np.isfinite(raw_sd) or raw_sd <= 1e-12:
        scaled = pd.Series(np.zeros(len(raw), dtype=np.float64), index=raw.index)
    else:
        scaled = raw * (prior_std / raw_sd)

    minute_weight = (minutes / (minutes + prior_minutes_scale)).fillna(0.0)
    games_weight = (games / (games + 12.0)).fillna(0.0)
    reliability = (minute_weight * games_weight).clip(lower=0.0, upper=1.0)
    box_prior = (scaled * reliability).fillna(0.0)

    out = pd.DataFrame(
        {
            "player_id": df["athlete_id"].astype(str),
            "box_prior": box_prior.astype(float),
            "prior_minutes_weight": reliability.astype(float),
        }
    )
    out = out.drop_duplicates(subset=["player_id"], keep="first")
    return out


def compute_epm_lite(rapm: pd.DataFrame, prior: pd.DataFrame, prior_possessions: float) -> pd.DataFrame:
    out = rapm.copy()
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(prior, how="left", on="player_id")
    out["box_prior"] = out["box_prior"].fillna(0.0)
    out["prior_minutes_weight"] = out["prior_minutes_weight"].fillna(0.0)
    out["rapm_weight"] = out["possessions"] / (out["possessions"] + float(prior_possessions))
    out["rapm_weight"] = out["rapm_weight"].clip(lower=0.0, upper=1.0)
    out["epm_lite"] = out["rapm_weight"] * out["rapm"] + (1.0 - out["rapm_weight"]) * out["box_prior"]
    out = out.sort_values("epm_lite", ascending=False)
    cols = [
        "player_id",
        "player_name",
        "player_team",
        "player_conference",
        "epm_lite",
        "rapm",
        "box_prior",
        "rapm_weight",
        "prior_minutes_weight",
        "possessions",
    ]
    cols = [c for c in cols if c in out.columns] + [c for c in out.columns if c not in cols]
    return out[cols]


def main() -> None:
    args = parse_args()
    plays = load_plays(args.season, max_files=args.max_files)
    if plays.empty:
        print("No data loaded. Exiting without output.")
        return
    print(f"Loaded {len(plays):,} deduped plays for season={args.season}.")

    if args.team_tier != "all":
        plays = filter_plays_by_tier(plays, team_tier=args.team_tier, mode=args.tier_filter_mode)
        if plays.empty:
            print("No plays left after team-tier filter. Exiting without output.")
            return

    rapm = compute_rapm(
        plays,
        ridge=args.ridge,
        min_possessions=args.min_possessions,
        max_players=args.max_players,
        free_throw_weight=args.free_throw_weight,
    )
    if rapm.empty:
        print("No RAPM output generated; cannot build EPM-lite.")
        return

    rapm = attach_player_metadata(rapm, season=args.season)
    prior = build_box_prior(
        season=args.season,
        prior_std=args.prior_std,
        prior_minutes_scale=args.prior_minutes_scale,
    )
    out = compute_epm_lite(rapm=rapm, prior=prior, prior_possessions=args.prior_possessions)

    args.out.mkdir(parents=True, exist_ok=True)
    tier_suffix = ""
    if args.team_tier != "all":
        tier_suffix = f"_{args.team_tier}_{args.tier_filter_mode}"
    out_path = args.out / f"epm_lite_top{args.top_n if args.top_n > 0 else 'all'}_season_{args.season}{tier_suffix}.csv"

    if args.top_n > 0:
        out.head(args.top_n).to_csv(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
