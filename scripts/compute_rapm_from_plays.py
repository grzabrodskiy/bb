from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


HIGH_MAJOR_CONFERENCES = {
    "acc",
    "big12",
    "bigeast",
    "bigten",
    "sec",
    "pac12",
}

MID_MAJOR_CONFERENCES = {
    "a10",
    "american",
    "mountainwest",
    "wcc",
    "mvc",
    "cusa",
    "mac",
    "sunbelt",
    "aac",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RAPM from CBBD play-by-play with lineup snapshots.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--ridge", type=float, default=100.0, help="Ridge penalty (lambda)")
    parser.add_argument("--min_possessions", type=float, default=200.0, help="Min possessions for player inclusion")
    parser.add_argument(
        "--max-players",
        type=int,
        default=3000,
        help="Cap player pool by possessions before solving ridge (set 0 to disable cap)",
    )
    parser.add_argument(
        "--free-throw-weight",
        type=float,
        default=0.44,
        help="Free throw possession weight used for event-based possession estimation",
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on team play files for quick laptop tests",
    )
    parser.add_argument(
        "--team-tier",
        type=str,
        choices=["all", "high", "mid", "high_mid"],
        default="all",
        help="Filter to D1 conference tiers before RAPM (high/mid/high_mid)",
    )
    parser.add_argument(
        "--tier-filter-mode",
        type=str,
        choices=["both", "team"],
        default="both",
        help="When filtering by team-tier: require both teams in tier, or only row team",
    )
    return parser.parse_args()


def _parse_lineup(val: object) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        out = []
        for x in val:
            if isinstance(x, dict):
                out.append(str(x.get("id") or x.get("name") or x))
            else:
                out.append(str(x))
        return out
    if isinstance(val, dict):
        if "athletes" in val and isinstance(val["athletes"], list):
            return [str(x) for x in val["athletes"]]
        return [str(x) for x in val.values()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # Try JSON first, then Python literal syntax.
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
                return _parse_lineup(parsed)
            except Exception:
                pass
        # fallback: pipe or comma separated
        if "|" in s:
            return [p.strip() for p in s.split("|") if p.strip()]
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(val)]


def find_lineup_cols(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = [c for c in df.columns if "lineup" in c.lower()]
    if not cols:
        return None

    home_candidates = [c for c in cols if "home" in c.lower()]
    away_candidates = [c for c in cols if "away" in c.lower()]

    if not home_candidates or not away_candidates:
        return None

    return home_candidates[0], away_candidates[0]


def find_possession_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["possession_count", "possessions", "home_possessions", "away_possessions", "possession"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_bool(val: object) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and not pd.isna(val):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _to_float(val: object) -> float:
    try:
        out = float(val)
    except (TypeError, ValueError):
        return np.nan
    return out


def _normalize_conference(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip().lower()
    if not s:
        return ""
    return "".join(ch for ch in s if ch.isalnum())


def _allowed_conferences(team_tier: str) -> set[str]:
    if team_tier == "high":
        return set(HIGH_MAJOR_CONFERENCES)
    if team_tier == "mid":
        return set(MID_MAJOR_CONFERENCES)
    if team_tier == "high_mid":
        return set(HIGH_MAJOR_CONFERENCES) | set(MID_MAJOR_CONFERENCES)
    return set()


def filter_plays_by_tier(df: pd.DataFrame, team_tier: str, mode: str = "both") -> pd.DataFrame:
    if team_tier == "all":
        return df

    needed = {"conference", "opponent_conference"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Cannot apply team-tier filter; missing columns: {missing}")
        return df

    allowed = _allowed_conferences(team_tier)
    if not allowed:
        return df

    total_rows = len(df)
    team_conf = df["conference"].map(_normalize_conference)
    if mode == "team":
        mask = team_conf.isin(allowed)
    else:
        opp_conf = df["opponent_conference"].map(_normalize_conference)
        mask = team_conf.isin(allowed) & opp_conf.isin(allowed)

    out = df.loc[mask].copy()
    print(
        f"Tier filter team_tier={team_tier}, mode={mode}: "
        f"kept {len(out):,}/{total_rows:,} plays."
    )
    return out


def _new_poss_counters() -> dict[str, dict[str, float]]:
    return {
        "home": {"fga": 0.0, "fta": 0.0, "orb": 0.0, "to": 0.0},
        "away": {"fga": 0.0, "fta": 0.0, "orb": 0.0, "to": 0.0},
    }


def _update_poss_counters(
    counters: dict[str, dict[str, float]],
    is_home_team: object,
    play_type: object,
    shooting_play: object,
    shot_range: object,
) -> None:
    side_bool = _to_bool(is_home_team)
    if side_bool is None:
        return
    side = "home" if side_bool else "away"

    ptype = str(play_type).strip() if play_type is not None else ""
    srange = str(shot_range).strip().lower() if shot_range is not None else ""
    is_shot = _to_bool(shooting_play) is True

    if is_shot and srange != "free_throw":
        counters[side]["fga"] += 1.0
    if is_shot and srange == "free_throw":
        counters[side]["fta"] += 1.0
    if ptype == "Offensive Rebound":
        counters[side]["orb"] += 1.0
    if "Turnover" in ptype:
        counters[side]["to"] += 1.0


def _estimate_stint_possessions(
    counters: dict[str, dict[str, float]],
    free_throw_weight: float,
) -> float:
    def team_poss(side: str) -> float:
        fga = counters[side]["fga"]
        fta = counters[side]["fta"]
        orb = counters[side]["orb"]
        tov = counters[side]["to"]
        return max(0.0, fga - orb + tov + free_throw_weight * fta)

    home = team_poss("home")
    away = team_poss("away")
    return max(0.0, (home + away) / 2.0)


def attach_player_metadata(out: pd.DataFrame, season: int) -> pd.DataFrame:
    lookup_path = Path("data/raw/cbbd/players") / f"season={season}" / "player_season_stats.csv"
    if not lookup_path.exists():
        return out

    lookup = pd.read_csv(
        lookup_path,
        usecols=["athlete_id", "name", "team", "conference"],
        low_memory=False,
    ).drop_duplicates(subset=["athlete_id"])
    lookup = lookup.rename(
        columns={
            "athlete_id": "player_id_num",
            "name": "player_name",
            "team": "player_team",
            "conference": "player_conference",
        }
    )

    enriched = out.copy()
    enriched["player_id_num"] = pd.to_numeric(enriched["player_id"], errors="coerce")
    enriched = enriched.merge(lookup, how="left", on="player_id_num")
    enriched = enriched.drop(columns=["player_id_num"])

    cols = ["player_id", "player_name", "player_team", "player_conference", "rapm", "possessions"]
    ordered_cols = [c for c in cols if c in enriched.columns] + [c for c in enriched.columns if c not in cols]
    return enriched[ordered_cols]


def build_lineups_from_on_floor(df: pd.DataFrame) -> pd.DataFrame:
    if "on_floor" not in df.columns:
        raise SystemExit("No lineup columns or on_floor data found.")

    # Build home/away team mapping per game.
    game_home: dict[object, object] = {}
    game_away: dict[object, object] = {}
    for row in df.itertuples(index=False):
        game_id = getattr(row, "game_id", None)
        if pd.isna(game_id):
            continue
        is_home = _to_bool(getattr(row, "is_home_team", None))
        team = getattr(row, "team", None)
        opponent = getattr(row, "opponent", None)
        if is_home is True:
            game_home[game_id] = team
            game_away[game_id] = opponent
        elif is_home is False:
            game_home[game_id] = opponent
            game_away[game_id] = team

    parsed_cache: dict[str, tuple[str, list[object]]] = {}

    def parse_cached(lineup: object) -> tuple[str, list[object]]:
        if not isinstance(lineup, str):
            players = _parse_lineup(lineup)
            return ("flat", players)

        cached = parsed_cache.get(lineup)
        if cached is not None:
            return cached

        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(lineup)
                break
            except Exception:
                pass

        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            entries = []
            for p in parsed:
                team = p.get("team")
                pid = p.get("id") or p.get("name")
                entries.append((team, str(pid) if pid is not None else ""))
            result = ("dict", entries)
        else:
            players = _parse_lineup(parsed if parsed is not None else lineup)
            result = ("flat", players)

        parsed_cache[lineup] = result
        return result

    home_lineups: list[list[str]] = []
    away_lineups: list[list[str]] = []
    for row in df.itertuples(index=False):
        game_id = getattr(row, "game_id", None)
        home_team = game_home.get(game_id)
        away_team = game_away.get(game_id)
        lineup = getattr(row, "on_floor", None)

        kind, parsed = parse_cached(lineup)
        if kind == "dict":
            entries = parsed
            home = [pid for team, pid in entries if team == home_team and pid]
            away = [pid for team, pid in entries if team == away_team and pid]
            if not home and not away:
                players = [pid for _, pid in entries if pid]
                mid = len(players) // 2
                home = players[:mid]
                away = players[mid:]
        else:
            players = [str(x) for x in parsed]
            mid = len(players) // 2
            home = players[:mid]
            away = players[mid:]

        home_lineups.append(home)
        away_lineups.append(away)

    out = df.copy()
    out["home_lineup"] = home_lineups
    out["away_lineup"] = away_lineups
    return out


def load_plays(season: int, max_files: Optional[int] = None) -> pd.DataFrame:
    root = Path("data/raw/cbbd/plays") / f"season={season}"
    files_by_team: dict[str, Path] = {}
    # Prefer uncompressed files when both exist; otherwise use gzipped fallback.
    for f in sorted(root.glob("team=*/plays.csv")):
        files_by_team[str(f.parent)] = f
    for f in sorted(root.glob("team=*/plays.csv.gz")):
        files_by_team.setdefault(str(f.parent), f)
    files = [files_by_team[k] for k in sorted(files_by_team)]
    if max_files is not None:
        files = files[:max_files]
    if not files:
        print(f"No plays files found under {root} (csv or csv.gz). Returning empty dataframe.")
        return pd.DataFrame()

    # Keep only needed columns to reduce memory pressure on laptops.
    required_cols = {
        "id",
        "play_id",
        "source_id",
        "game_id",
        "period",
        "clock",
        "home_score",
        "away_score",
        "play_type",
        "shooting_play",
        "on_floor",
        "is_home_team",
        "team",
        "conference",
        "opponent",
        "opponent_conference",
        "shot_info.range",
        "possession_count",
        "possessions",
        "home_possessions",
        "away_possessions",
        "possession",
    }

    dfs = []
    dedupe_key = None
    seen_ids: set[object] = set()
    for f in files:
        header = pd.read_csv(f, nrows=0).columns.tolist()
        usecols = [c for c in header if c in required_cols or "lineup" in c.lower()]
        chunk = pd.read_csv(f, usecols=usecols, low_memory=False)

        if dedupe_key is None:
            for key in ["id", "play_id", "source_id"]:
                if key in chunk.columns:
                    dedupe_key = key
                    break

        if dedupe_key and dedupe_key in chunk.columns:
            mask = ~chunk[dedupe_key].isin(seen_ids)
            new_ids = chunk.loc[mask, dedupe_key].dropna().tolist()
            seen_ids.update(new_ids)
            chunk = chunk.loc[mask]

        if not chunk.empty:
            dfs.append(chunk)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    return df


def compute_rapm(
    df: pd.DataFrame,
    ridge: float,
    min_possessions: float,
    max_players: Optional[int] = 3000,
    free_throw_weight: float = 0.44,
) -> pd.DataFrame:
    lineup_cols = find_lineup_cols(df)
    poss_col = find_possession_col(df)

    if lineup_cols is None:
        try:
            df = build_lineups_from_on_floor(df)
            home_col = "home_lineup"
            away_col = "away_lineup"
        except SystemExit as exc:
            print(str(exc))
            return pd.DataFrame()
    else:
        home_col, away_col = lineup_cols

    required_cols = ["game_id", "period", "clock", "home_score", "away_score", home_col, away_col]
    if poss_col:
        required_cols.append(poss_col)
    else:
        required_cols.extend(["play_type", "shooting_play", "is_home_team", "shot_info.range"])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing required columns for RAPM: {missing}")
        return pd.DataFrame()

    # Parse lineups
    if home_col != "home_lineup":
        df["home_lineup"] = df[home_col].apply(_parse_lineup)
    if away_col != "away_lineup":
        df["away_lineup"] = df[away_col].apply(_parse_lineup)

    # Make dotted column names attribute-safe for itertuples().
    if "shot_info.range" in df.columns and "shot_info_range" not in df.columns:
        df = df.rename(columns={"shot_info.range": "shot_info_range"})

    # Convert clock to seconds remaining in period if it's MM:SS
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

    # Order plays by game, period, clock descending
    df = df.sort_values(["game_id", "period", "clock_sec"], ascending=[True, True, False])

    # Build aggregated stints when lineup changes (memory-safe for laptop runs).
    stint_agg: dict[tuple[tuple[str, ...], tuple[str, ...]], list[float]] = {}
    player_poss: dict[str, float] = defaultdict(float)

    def add_stint(
        home_lineup: Optional[Tuple[str, ...]],
        away_lineup: Optional[Tuple[str, ...]],
        home_points: float,
        away_points: float,
        poss: float,
    ) -> None:
        if home_lineup is None or away_lineup is None:
            return
        if not np.isfinite(poss) or poss <= 0:
            return
        key = (home_lineup, away_lineup)
        if key not in stint_agg:
            stint_agg[key] = [0.0, 0.0]
        stint_agg[key][0] += poss
        stint_agg[key][1] += (home_points - away_points)
        for p in home_lineup:
            player_poss[p] += poss
        for p in away_lineup:
            player_poss[p] += poss

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
                        poss_counters,
                        row_is_home_team,
                        row_play_type,
                        row_shooting_play,
                        row_shot_range,
                    )
                last_home_score = row_home_score
                last_away_score = row_away_score
                last_poss = row_poss if poss_col else np.nan
                continue

            if home != prev_home or away != prev_away:
                # Close previous stint at the prior row (lineup changed at current row).
                home_points = last_home_score - prev_home_score
                away_points = last_away_score - prev_away_score
                if poss_col:
                    poss = (
                        last_poss - prev_poss
                        if np.isfinite(last_poss) and np.isfinite(prev_poss) and last_poss >= prev_poss
                        else np.nan
                    )
                else:
                    poss = _estimate_stint_possessions(poss_counters, free_throw_weight)

                add_stint(prev_home, prev_away, home_points, away_points, poss)

                prev_home, prev_away = home, away
                prev_home_score = row_home_score
                prev_away_score = row_away_score
                prev_poss = row_poss if poss_col else np.nan
                poss_counters = _new_poss_counters()

            if not poss_col:
                _update_poss_counters(
                    poss_counters,
                    row_is_home_team,
                    row_play_type,
                    row_shooting_play,
                    row_shot_range,
                )
            last_home_score = row_home_score
            last_away_score = row_away_score
            last_poss = row_poss if poss_col else np.nan

        if prev_home is not None and last_home_score is not None and last_away_score is not None:
            final_home_points = last_home_score - prev_home_score
            final_away_points = last_away_score - prev_away_score
            if poss_col:
                final_poss = (
                    last_poss - prev_poss
                    if np.isfinite(last_poss) and np.isfinite(prev_poss) and last_poss >= prev_poss
                    else np.nan
                )
            else:
                final_poss = _estimate_stint_possessions(poss_counters, free_throw_weight)
            add_stint(prev_home, prev_away, final_home_points, final_away_points, final_poss)

    if not stint_agg:
        print("No valid stints found after lineup/possession filtering.")
        return pd.DataFrame()

    eligible = [(p, poss) for p, poss in player_poss.items() if poss >= min_possessions]
    eligible.sort(key=lambda x: x[1], reverse=True)
    if max_players and max_players > 0 and len(eligible) > max_players:
        print(f"Capping player pool to top {max_players} by possessions (from {len(eligible)}).")
        eligible = eligible[:max_players]

    players = [p for p, _ in eligible]
    if not players:
        print(f"No players met min_possessions={min_possessions}.")
        return pd.DataFrame()

    print(
        f"Aggregated {len(stint_agg):,} unique stints; "
        f"{len(players):,} players meet min_possessions={min_possessions}."
    )

    idx = {p: i for i, p in enumerate(players)}
    A = np.zeros((len(players), len(players)), dtype=np.float64)
    b = np.zeros(len(players), dtype=np.float64)

    for (home, away), (poss_sum, point_diff_sum) in stint_agg.items():
        row_ids = []
        row_vals = []
        for p in home:
            j = idx.get(p)
            if j is not None:
                row_ids.append(j)
                row_vals.append(1.0)
        for p in away:
            j = idx.get(p)
            if j is not None:
                row_ids.append(j)
                row_vals.append(-1.0)
        if not row_ids:
            continue

        ids = np.asarray(row_ids, dtype=np.int64)
        vals = np.asarray(row_vals, dtype=np.float64)
        A[np.ix_(ids, ids)] += poss_sum * np.outer(vals, vals)
        b[ids] += vals * (point_diff_sum * 100.0)

    A += ridge * np.eye(A.shape[0], dtype=np.float64)
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, b, rcond=None)[0]

    player_poss_arr = np.asarray([player_poss[p] for p in players], dtype=np.float64)

    out = pd.DataFrame({
        "player_id": players,
        "rapm": coef,
        "possessions": player_poss_arr,
    })

    out = out.sort_values("rapm", ascending=False)
    return out


def main() -> None:
    args = parse_args()
    df = load_plays(args.season, max_files=args.max_files)
    if df.empty:
        print("No data loaded. Exiting without output.")
        return
    print(f"Loaded {len(df):,} deduped plays for season={args.season}.")
    if args.team_tier != "all":
        df = filter_plays_by_tier(df, team_tier=args.team_tier, mode=args.tier_filter_mode)
        if df.empty:
            print("No plays left after team-tier filter. Exiting without output.")
            return
    out = compute_rapm(
        df,
        ridge=args.ridge,
        min_possessions=args.min_possessions,
        max_players=args.max_players,
        free_throw_weight=args.free_throw_weight,
    )
    if out.empty:
        print("No RAPM output generated.")
        return
    out = attach_player_metadata(out, season=args.season)

    args.out.mkdir(parents=True, exist_ok=True)
    tier_suffix = ""
    if args.team_tier != "all":
        tier_suffix = f"_{args.team_tier}_{args.tier_filter_mode}"
    out_path = args.out / f"rapm_top100_season_{args.season}{tier_suffix}.csv"
    out.head(100).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
