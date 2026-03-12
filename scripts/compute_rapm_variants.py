from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow direct execution via `python3 scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_rapm_from_plays import (
    _estimate_stint_possessions,
    _new_poss_counters,
    _parse_lineup,
    _to_float,
    _update_poss_counters,
    attach_player_metadata,
    build_lineups_from_on_floor,
    filter_plays_by_tier,
    find_lineup_cols,
    find_possession_col,
    load_plays,
)


@dataclass
class Stint:
    game_id: object
    home_lineup: tuple[str, ...]
    away_lineup: tuple[str, ...]
    poss: float
    home_points: float
    away_points: float
    margin_start: float

    @property
    def point_diff(self) -> float:
        return self.home_points - self.away_points


@dataclass
class RegRow:
    game_id: object
    ids: np.ndarray
    vals: np.ndarray
    y: float
    w: float
    margin_start: float
    home_ids: np.ndarray
    away_ids: np.ndarray
    home_points: float
    away_points: float
    poss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute multiple pure-stat RAPM variants from CBBD play-by-play."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--ridge", type=float, default=100.0, help="Base ridge lambda.")
    parser.add_argument(
        "--ridge-grid",
        type=str,
        default="25,50,100,200,400",
        help="Comma-separated lambdas for game-level CV.",
    )
    parser.add_argument(
        "--min-possessions",
        type=float,
        default=200.0,
        help="Min possessions for player inclusion.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=2500,
        help="Cap player pool by possessions (0 disables).",
    )
    parser.add_argument(
        "--free-throw-weight",
        type=float,
        default=0.44,
        help="Free throw possession weight used for event-based possession estimation.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument(
        "--team-tier",
        type=str,
        choices=["all", "high", "mid", "high_mid"],
        default="all",
        help="Filter to D1 conference tiers before modeling.",
    )
    parser.add_argument(
        "--tier-filter-mode",
        type=str,
        choices=["both", "team"],
        default="both",
        help="When filtering by team-tier: require both teams in tier, or only row team.",
    )
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--cv-val-frac", type=float, default=0.2)
    parser.add_argument(
        "--huber-iters",
        type=int,
        default=5,
        help="IRLS iterations for robust RAPM.",
    )
    parser.add_argument(
        "--huber-delta-mult",
        type=float,
        default=1.5,
        help="Huber cutoff = multiplier * robust residual scale.",
    )
    parser.add_argument(
        "--garbage-margin",
        type=float,
        default=20.0,
        help="Score margin cutoff at stint start for close-game RAPM.",
    )
    parser.add_argument(
        "--eb-shrink-possessions",
        type=float,
        default=1200.0,
        help="Empirical-Bayes shrink factor for possessions-based shrinkage.",
    )
    parser.add_argument(
        "--prior-std",
        type=float,
        default=2.5,
        help="Target std for box-score prior mean in Bayesian RAPM.",
    )
    parser.add_argument(
        "--prior-minutes-scale",
        type=float,
        default=700.0,
        help="Minutes scale for prior reliability shrinkage.",
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--player-filter",
        type=str,
        choices=["all", "freshmen"],
        default="all",
        help=(
            "Filter output players. 'freshmen' uses first-seen season in local "
            "player_season_stats history (athlete_id min season == target season)."
        ),
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


def build_box_prior_map(
    season: int,
    prior_std: float,
    prior_minutes_scale: float,
) -> dict[str, float]:
    path = Path("data/raw/cbbd/players") / f"season={season}" / "player_season_stats.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return {}

    minutes = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0).clip(lower=0.0)
    games = pd.to_numeric(df.get("games"), errors="coerce").fillna(0.0).clip(lower=0.0)
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
    prior = (scaled * reliability).fillna(0.0)
    out = pd.DataFrame({"player_id": df["athlete_id"].astype(str), "prior": prior.astype(float)})
    out = out.drop_duplicates(subset=["player_id"], keep="first")
    return dict(zip(out["player_id"], out["prior"]))


def infer_freshman_ids_from_history(season: int) -> set[str]:
    root = Path("data/raw/cbbd/players")
    files = sorted(root.glob("season=*/player_season_stats.csv"))
    if not files:
        print("Freshman filter unavailable: no historical player_season_stats files found.")
        return set()

    first_seen: dict[str, int] = {}
    for f in files:
        try:
            season_val = int(f.parent.name.split("=", 1)[1])
        except Exception:
            continue
        if season_val > season:
            continue

        try:
            chunk = pd.read_csv(
                f,
                usecols=["athlete_id", "games", "minutes"],
                low_memory=False,
            )
        except Exception:
            continue
        if chunk.empty:
            continue

        games = pd.to_numeric(chunk.get("games"), errors="coerce").fillna(0.0)
        minutes = pd.to_numeric(chunk.get("minutes"), errors="coerce").fillna(0.0)
        # Require some participation signal to avoid counting pure empty rows.
        active = (games > 0) | (minutes > 0)
        chunk = chunk.loc[active, ["athlete_id"]].copy()
        if chunk.empty:
            continue
        chunk["player_id"] = chunk["athlete_id"].astype(str)
        for pid in chunk["player_id"].dropna().tolist():
            prev = first_seen.get(pid)
            if prev is None or season_val < prev:
                first_seen[pid] = season_val

    freshmen = {pid for pid, yr in first_seen.items() if yr == season}
    print(f"Inferred freshmen from history for season={season}: {len(freshmen):,} players.")
    return freshmen


def build_stints(df: pd.DataFrame, free_throw_weight: float) -> list[Stint]:
    lineup_cols = find_lineup_cols(df)
    poss_col = find_possession_col(df)
    if lineup_cols is None:
        df = build_lineups_from_on_floor(df)
        home_col = "home_lineup"
        away_col = "away_lineup"
    else:
        home_col, away_col = lineup_cols

    required_cols = ["game_id", "period", "clock", "home_score", "away_score", home_col, away_col]
    if poss_col:
        required_cols.append(poss_col)
    else:
        required_cols.extend(["play_type", "shooting_play", "is_home_team", "shot_info.range"])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns for stint build: {missing}")

    if home_col != "home_lineup":
        df["home_lineup"] = df[home_col].apply(_parse_lineup)
    if away_col != "away_lineup":
        df["away_lineup"] = df[away_col].apply(_parse_lineup)

    if "shot_info.range" in df.columns and "shot_info_range" not in df.columns:
        df = df.rename(columns={"shot_info.range": "shot_info_range"})

    if df["clock"].dtype == object:
        def clock_to_seconds(x: object) -> float:
            if isinstance(x, str) and ":" in x:
                mins, secs = x.split(":")
                return float(mins) * 60 + float(secs)
            try:
                return float(x)
            except Exception:
                return np.nan
        df["clock_sec"] = df["clock"].apply(clock_to_seconds)
    else:
        df["clock_sec"] = df["clock"].astype(float)

    df = df.sort_values(["game_id", "period", "clock_sec"], ascending=[True, True, False])

    stints: list[Stint] = []

    def add_stint(
        game_id: object,
        prev_home: Optional[tuple[str, ...]],
        prev_away: Optional[tuple[str, ...]],
        start_home_score: float,
        start_away_score: float,
        end_home_score: float,
        end_away_score: float,
        poss: float,
    ) -> None:
        if prev_home is None or prev_away is None:
            return
        if not np.isfinite(poss) or poss <= 0:
            return
        stints.append(
            Stint(
                game_id=game_id,
                home_lineup=prev_home,
                away_lineup=prev_away,
                poss=float(poss),
                home_points=float(end_home_score - start_home_score),
                away_points=float(end_away_score - start_away_score),
                margin_start=float(start_home_score - start_away_score),
            )
        )

    for game_id, g in df.groupby("game_id", sort=False):
        prev_home = None
        prev_away = None
        prev_home_score = None
        prev_away_score = None
        prev_poss = None
        last_home_score = None
        last_away_score = None
        last_poss = None
        poss_counters = _new_poss_counters()

        for row in g.itertuples(index=False):
            home = tuple(getattr(row, "home_lineup"))
            away = tuple(getattr(row, "away_lineup"))
            row_home_score = _to_float(getattr(row, "home_score"))
            row_away_score = _to_float(getattr(row, "away_score"))
            row_poss = _to_float(getattr(row, poss_col)) if poss_col else np.nan
            row_is_home_team = getattr(row, "is_home_team")
            row_play_type = getattr(row, "play_type")
            row_shooting_play = getattr(row, "shooting_play")
            row_shot_range = getattr(row, "shot_info_range", None)

            if prev_home is None:
                prev_home, prev_away = home, away
                prev_home_score = row_home_score
                prev_away_score = row_away_score
                prev_poss = row_poss if poss_col else np.nan
                poss_counters = _new_poss_counters()
                if not poss_col:
                    _update_poss_counters(
                        poss_counters, row_is_home_team, row_play_type, row_shooting_play, row_shot_range
                    )
                last_home_score = row_home_score
                last_away_score = row_away_score
                last_poss = row_poss if poss_col else np.nan
                continue

            if home != prev_home or away != prev_away:
                if poss_col:
                    poss = (
                        last_poss - prev_poss
                        if np.isfinite(last_poss) and np.isfinite(prev_poss) and last_poss >= prev_poss
                        else np.nan
                    )
                else:
                    poss = _estimate_stint_possessions(poss_counters, free_throw_weight)

                add_stint(
                    game_id=game_id,
                    prev_home=prev_home,
                    prev_away=prev_away,
                    start_home_score=prev_home_score,
                    start_away_score=prev_away_score,
                    end_home_score=last_home_score,
                    end_away_score=last_away_score,
                    poss=poss,
                )

                prev_home, prev_away = home, away
                prev_home_score = row_home_score
                prev_away_score = row_away_score
                prev_poss = row_poss if poss_col else np.nan
                poss_counters = _new_poss_counters()

            if not poss_col:
                _update_poss_counters(
                    poss_counters, row_is_home_team, row_play_type, row_shooting_play, row_shot_range
                )
            last_home_score = row_home_score
            last_away_score = row_away_score
            last_poss = row_poss if poss_col else np.nan

        if prev_home is not None and last_home_score is not None and last_away_score is not None:
            if poss_col:
                final_poss = (
                    last_poss - prev_poss
                    if np.isfinite(last_poss) and np.isfinite(prev_poss) and last_poss >= prev_poss
                    else np.nan
                )
            else:
                final_poss = _estimate_stint_possessions(poss_counters, free_throw_weight)
            add_stint(
                game_id=game_id,
                prev_home=prev_home,
                prev_away=prev_away,
                start_home_score=prev_home_score,
                start_away_score=prev_away_score,
                end_home_score=last_home_score,
                end_away_score=last_away_score,
                poss=final_poss,
            )

    return stints


def build_player_pool(
    stints: list[Stint],
    min_possessions: float,
    max_players: int,
) -> tuple[list[str], dict[str, float]]:
    player_poss: dict[str, float] = defaultdict(float)
    for s in stints:
        for p in s.home_lineup:
            player_poss[p] += s.poss
        for p in s.away_lineup:
            player_poss[p] += s.poss
    eligible = [(p, poss) for p, poss in player_poss.items() if poss >= min_possessions]
    eligible.sort(key=lambda x: x[1], reverse=True)
    if max_players and max_players > 0 and len(eligible) > max_players:
        print(f"Capping player pool to top {max_players} by possessions (from {len(eligible)}).")
        eligible = eligible[:max_players]
    players = [p for p, _ in eligible]
    return players, player_poss


def build_reg_rows(stints: list[Stint], idx: dict[str, int]) -> list[RegRow]:
    rows: list[RegRow] = []
    for s in stints:
        home_ids = np.asarray([idx[p] for p in s.home_lineup if p in idx], dtype=np.int64)
        away_ids = np.asarray([idx[p] for p in s.away_lineup if p in idx], dtype=np.int64)
        if home_ids.size == 0 and away_ids.size == 0:
            continue
        ids = np.concatenate([home_ids, away_ids])
        vals = np.concatenate(
            [np.ones(home_ids.size, dtype=np.float64), -np.ones(away_ids.size, dtype=np.float64)]
        )
        y = 100.0 * s.point_diff / s.poss
        rows.append(
            RegRow(
                game_id=s.game_id,
                ids=ids,
                vals=vals,
                y=float(y),
                w=float(s.poss),
                margin_start=float(s.margin_start),
                home_ids=home_ids,
                away_ids=away_ids,
                home_points=float(s.home_points),
                away_points=float(s.away_points),
                poss=float(s.poss),
            )
        )
    return rows


def solve_ridge_rows(
    rows: list[RegRow],
    n_features: int,
    ridge: float,
    prior_mu: Optional[np.ndarray] = None,
    penalty_vec: Optional[np.ndarray] = None,
    extra_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    A = np.zeros((n_features, n_features), dtype=np.float64)
    b = np.zeros(n_features, dtype=np.float64)

    for i, r in enumerate(rows):
        ew = 1.0 if extra_weights is None else float(extra_weights[i])
        if ew <= 0:
            continue
        w = r.w * ew
        if w <= 0:
            continue
        ids = r.ids
        vals = r.vals
        A[np.ix_(ids, ids)] += w * np.outer(vals, vals)
        b[ids] += w * vals * r.y

    if penalty_vec is None:
        A[np.diag_indices(n_features)] += ridge
    else:
        A[np.diag_indices(n_features)] += penalty_vec

    if prior_mu is not None:
        b += ridge * prior_mu

    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def predict_rows(rows: list[RegRow], beta: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        pred[i] = float(np.dot(beta[r.ids], r.vals))
    return pred


def weighted_mse(rows: list[RegRow], beta: np.ndarray) -> float:
    if not rows:
        return float("inf")
    y = np.asarray([r.y for r in rows], dtype=np.float64)
    w = np.asarray([r.w for r in rows], dtype=np.float64)
    pred = predict_rows(rows, beta)
    denom = float(np.sum(w))
    if denom <= 0:
        return float("inf")
    return float(np.sum(w * (y - pred) ** 2) / denom)


def select_ridge_by_game_cv(
    rows: list[RegRow],
    n_features: int,
    lambdas: list[float],
    val_frac: float,
    seed: int,
) -> float:
    game_ids = np.asarray([r.game_id for r in rows], dtype=object)
    uniq = np.unique(game_ids)
    if uniq.size < 5:
        return lambdas[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_val = int(max(1, round(val_frac * len(perm))))
    val_games = set(perm[:n_val].tolist())
    train_rows = [r for r in rows if r.game_id not in val_games]
    val_rows = [r for r in rows if r.game_id in val_games]
    if not train_rows or not val_rows:
        return lambdas[0]

    best_lambda = lambdas[0]
    best_loss = float("inf")
    print(f"CV split: {len(train_rows):,} train rows, {len(val_rows):,} val rows.")
    for lam in lambdas:
        beta = solve_ridge_rows(train_rows, n_features=n_features, ridge=lam)
        loss = weighted_mse(val_rows, beta)
        print(f"  lambda={lam:.3f} val_wmse={loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_lambda = lam
    print(f"Selected lambda_cv={best_lambda:.3f}")
    return best_lambda


def fit_huber_rapm(
    rows: list[RegRow],
    n_features: int,
    ridge: float,
    iters: int,
    delta_mult: float,
) -> np.ndarray:
    beta = solve_ridge_rows(rows, n_features=n_features, ridge=ridge)
    y = np.asarray([r.y for r in rows], dtype=np.float64)
    for it in range(iters):
        pred = predict_rows(rows, beta)
        resid = y - pred
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-9:
            break
        delta = delta_mult * scale
        weights = np.ones_like(resid)
        mask = np.abs(resid) > delta
        weights[mask] = delta / (np.abs(resid[mask]) + 1e-12)
        beta_new = solve_ridge_rows(
            rows,
            n_features=n_features,
            ridge=ridge,
            extra_weights=weights,
        )
        diff = float(np.linalg.norm(beta_new - beta))
        beta = beta_new
        print(f"  huber_iter={it + 1} scale={scale:.4f} delta={delta:.4f} step_norm={diff:.4f}")
        if diff < 1e-5:
            break
    return beta


def fit_hca_rapm(rows: list[RegRow], n_players: int, ridge: float) -> tuple[np.ndarray, float]:
    n = n_players + 1
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    hca_idx = n_players

    for r in rows:
        ids = np.concatenate([r.ids, np.asarray([hca_idx], dtype=np.int64)])
        vals = np.concatenate([r.vals, np.asarray([1.0], dtype=np.float64)])
        A[np.ix_(ids, ids)] += r.w * np.outer(vals, vals)
        b[ids] += r.w * vals * r.y

    penalty = np.full(n, ridge, dtype=np.float64)
    penalty[hca_idx] = 1e-6
    A[np.diag_indices(n)] += penalty
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return beta[:n_players], float(beta[hca_idx])


def fit_od_rapm(rows: list[RegRow], n_players: int, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    n = 2 * n_players
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for r in rows:
        if r.poss <= 0:
            continue
        # Home offense vs away defense
        ids_home_off = r.home_ids
        ids_away_def = r.away_ids + n_players
        ids = np.concatenate([ids_home_off, ids_away_def])
        vals = np.concatenate(
            [np.ones(ids_home_off.size, dtype=np.float64), -np.ones(ids_away_def.size, dtype=np.float64)]
        )
        if ids.size > 0:
            y_home = 100.0 * r.home_points / r.poss
            A[np.ix_(ids, ids)] += r.w * np.outer(vals, vals)
            b[ids] += r.w * vals * y_home

        # Away offense vs home defense
        ids_away_off = r.away_ids
        ids_home_def = r.home_ids + n_players
        ids = np.concatenate([ids_away_off, ids_home_def])
        vals = np.concatenate(
            [np.ones(ids_away_off.size, dtype=np.float64), -np.ones(ids_home_def.size, dtype=np.float64)]
        )
        if ids.size > 0:
            y_away = 100.0 * r.away_points / r.poss
            A[np.ix_(ids, ids)] += r.w * np.outer(vals, vals)
            b[ids] += r.w * vals * y_away

    A[np.diag_indices(n)] += ridge
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return beta[:n_players], beta[n_players:]


def main() -> None:
    args = parse_args()
    plays = load_plays(args.season, max_files=args.max_files)
    if plays.empty:
        print("No plays loaded. Exiting.")
        return
    print(f"Loaded {len(plays):,} deduped plays for season={args.season}.")

    if args.team_tier != "all":
        plays = filter_plays_by_tier(plays, team_tier=args.team_tier, mode=args.tier_filter_mode)
        if plays.empty:
            print("No plays left after team-tier filter. Exiting.")
            return

    stints = build_stints(plays, free_throw_weight=args.free_throw_weight)
    if not stints:
        print("No valid stints built. Exiting.")
        return
    print(f"Built {len(stints):,} stints.")

    players, player_poss = build_player_pool(
        stints,
        min_possessions=args.min_possessions,
        max_players=args.max_players,
    )
    if not players:
        print("No players met min possession threshold. Exiting.")
        return
    print(f"Players in pool: {len(players):,} (min_possessions={args.min_possessions}).")

    idx = {p: i for i, p in enumerate(players)}
    rows = build_reg_rows(stints, idx)
    if not rows:
        print("No model rows after player filtering. Exiting.")
        return
    print(f"Regression rows: {len(rows):,}.")

    lambdas = [float(x.strip()) for x in args.ridge_grid.split(",") if x.strip()]
    lambda_cv = select_ridge_by_game_cv(
        rows,
        n_features=len(players),
        lambdas=lambdas,
        val_frac=args.cv_val_frac,
        seed=args.cv_seed,
    )

    print("Fitting ridge RAPM (fixed lambda)...")
    beta_ridge = solve_ridge_rows(rows, n_features=len(players), ridge=args.ridge)

    print("Fitting ridge RAPM (CV lambda)...")
    beta_cv = solve_ridge_rows(rows, n_features=len(players), ridge=lambda_cv)

    print("Fitting robust Huber RAPM...")
    beta_huber = fit_huber_rapm(
        rows,
        n_features=len(players),
        ridge=lambda_cv,
        iters=args.huber_iters,
        delta_mult=args.huber_delta_mult,
    )

    print("Fitting Bayesian RAPM (box prior mean)...")
    prior_map = build_box_prior_map(
        season=args.season,
        prior_std=args.prior_std,
        prior_minutes_scale=args.prior_minutes_scale,
    )
    prior_vec = np.asarray([prior_map.get(p, 0.0) for p in players], dtype=np.float64)
    beta_bayes_box = solve_ridge_rows(
        rows,
        n_features=len(players),
        ridge=lambda_cv,
        prior_mu=prior_vec,
    )

    print("Fitting close-game RAPM (garbage-time filtered)...")
    rows_close = [r for r in rows if np.isfinite(r.margin_start) and abs(r.margin_start) <= args.garbage_margin]
    if rows_close:
        print(f"  close-game rows: {len(rows_close):,}")
        beta_close = solve_ridge_rows(rows_close, n_features=len(players), ridge=lambda_cv)
    else:
        print("  no close-game rows; falling back to CV ridge.")
        beta_close = beta_cv.copy()

    print("Fitting home-court-adjusted RAPM...")
    beta_hca, home_court = fit_hca_rapm(rows, n_players=len(players), ridge=lambda_cv)

    print("Fitting offense/defense RAPM...")
    beta_off, beta_def = fit_od_rapm(rows, n_players=len(players), ridge=lambda_cv)
    beta_od_net = beta_off + beta_def

    poss_arr = np.asarray([player_poss[p] for p in players], dtype=np.float64)
    eb_weight = poss_arr / (poss_arr + args.eb_shrink_possessions)
    beta_eb = beta_cv * eb_weight

    out = pd.DataFrame(
        {
            "player_id": players,
            "possessions": poss_arr,
            "rapm_ridge": beta_ridge,
            "rapm_cv": beta_cv,
            "rapm_huber": beta_huber,
            "rapm_bayes_box": beta_bayes_box,
            "rapm_close": beta_close,
            "rapm_hca": beta_hca,
            "orapm": beta_off,
            "drapm": beta_def,
            "rapm_od_net": beta_od_net,
            "rapm_eb": beta_eb,
            "eb_weight": eb_weight,
            "box_prior": prior_vec,
        }
    )
    out = attach_player_metadata(out, season=args.season)

    player_suffix = ""
    if args.player_filter == "freshmen":
        freshman_ids = infer_freshman_ids_from_history(args.season)
        if not freshman_ids:
            print("No freshmen identified from local history. Exiting without output.")
            return
        pre_n = len(out)
        out = out[out["player_id"].astype(str).isin(freshman_ids)].copy()
        player_suffix = "_freshmen"
        print(f"Applied freshmen filter: kept {len(out):,}/{pre_n:,} players.")
        if out.empty:
            print("No players left after freshmen filter. Exiting without output.")
            return

    args.out.mkdir(parents=True, exist_ok=True)
    tier_suffix = ""
    if args.team_tier != "all":
        tier_suffix = f"_{args.team_tier}_{args.tier_filter_mode}"
    out_suffix = f"{tier_suffix}{player_suffix}"
    base_path = args.out / f"rapm_variants_season_{args.season}{out_suffix}.csv"
    out.to_csv(base_path, index=False)

    summary_path = args.out / f"rapm_variants_summary_season_{args.season}{out_suffix}.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"season={args.season}\n")
        f.write(f"lambda_fixed={args.ridge}\n")
        f.write(f"lambda_cv={lambda_cv}\n")
        f.write(f"home_court_points_per_100={home_court}\n")
        f.write(f"players={len(players)}\n")
        f.write(f"players_output={len(out)}\n")
        f.write(f"rows={len(rows)}\n")
        f.write(f"rows_close={len(rows_close)}\n")
        f.write(f"garbage_margin={args.garbage_margin}\n")
        f.write(f"player_filter={args.player_filter}\n")

    cols = [
        "rapm_ridge",
        "rapm_cv",
        "rapm_huber",
        "rapm_bayes_box",
        "rapm_close",
        "rapm_hca",
        "rapm_od_net",
        "orapm",
        "drapm",
        "rapm_eb",
    ]
    n = args.top_n if args.top_n > 0 else len(out)
    for c in cols:
        keep = [
            col
            for col in [
                "player_id",
                "player_name",
                "player_team",
                "player_conference",
                c,
                "possessions",
                "rapm_cv",
                "box_prior",
                "eb_weight",
            ]
            if col in out.columns
        ]
        path = args.out / f"rapm_top{n}_{c}_season_{args.season}{out_suffix}.csv"
        out.sort_values(c, ascending=False).head(n)[keep].to_csv(path, index=False)

    print(f"Wrote {base_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
