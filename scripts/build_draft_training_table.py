from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

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

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Basic transliteration for Cyrillic characters that appear in some public draft feeds
# (for example "Dёmin"), so name-key joins remain stable against mixed-script variants.
CYRILLIC_TO_LATIN = {
    ord("а"): "a",
    ord("б"): "b",
    ord("в"): "v",
    ord("г"): "g",
    ord("д"): "d",
    ord("е"): "e",
    ord("ё"): "e",
    ord("ж"): "zh",
    ord("з"): "z",
    ord("и"): "i",
    ord("й"): "i",
    ord("к"): "k",
    ord("л"): "l",
    ord("м"): "m",
    ord("н"): "n",
    ord("о"): "o",
    ord("п"): "p",
    ord("р"): "r",
    ord("с"): "s",
    ord("т"): "t",
    ord("у"): "u",
    ord("ф"): "f",
    ord("х"): "h",
    ord("ц"): "c",
    ord("ч"): "ch",
    ord("ш"): "sh",
    ord("щ"): "sh",
    ord("ъ"): "",
    ord("ы"): "y",
    ord("ь"): "",
    ord("э"): "e",
    ord("ю"): "yu",
    ord("я"): "ya",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training table for NBA draft prediction.")
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument(
        "--draft-csv",
        type=Path,
        default=Path("data/raw/nba/draft/nba_draft_history_2015_2025.csv"),
    )
    parser.add_argument(
        "--players-root",
        type=Path,
        default=Path("data/raw/cbbd/players"),
    )
    parser.add_argument(
        "--bios-root",
        type=Path,
        default=Path("data/raw/espn/player_bios"),
    )
    parser.add_argument(
        "--external-model-db",
        type=Path,
        default=Path("data/raw/external/nba_draft_model/model_db.csv"),
        help="Optional external draft-feature table (RealGM model_db.csv format).",
    )
    parser.add_argument(
        "--external-backfill-max-gap",
        type=int,
        default=0,
        help=(
            "Maximum season gap when backfilling external features from prior seasons. "
            "Set to 0 to disable backfill."
        ),
    )
    parser.add_argument(
        "--min-games",
        type=float,
        default=5.0,
        help="Drop ultra-small samples before labeling.",
    )
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=100.0,
        help="Drop ultra-small samples before labeling.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    s = s.translate(CYRILLIC_TO_LATIN)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_name(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    s = s.translate(CYRILLIC_TO_LATIN)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    tokens = [t for t in s.split() if t and t not in NAME_SUFFIXES]
    return "".join(tokens)


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = _safe_num(values)
    w = _safe_num(weights).fillna(0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def _conference_tier(conference: object) -> str:
    key = normalize_text(conference)
    if key in HIGH_MAJOR_CONFERENCES:
        return "high"
    if key in MID_MAJOR_CONFERENCES:
        return "mid"
    return "other"


def _team_college_match(team_key: str, college_key: str) -> bool:
    if not team_key or not college_key:
        return False
    if team_key == college_key:
        return True
    return (team_key in college_key) or (college_key in team_key)


def _parse_external_season_end(val: object) -> int | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    m = re.search(r"(\d{4})-(\d{2})", s)
    if m:
        y0 = int(m.group(1))
        y1 = int(m.group(2))
        century = (y0 // 100) * 100
        if y1 < (y0 % 100):
            century += 100
        return century + y1
    m = re.search(r"(\d{4})", s)
    if m:
        return int(m.group(1))
    return None


def aggregate_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Multiple rows can exist for an athlete in a season (transfers).
    sum_cols = [
        "games",
        "starts",
        "minutes",
        "points",
        "turnovers",
        "fouls",
        "assists",
        "steals",
        "blocks",
        "field_goals.attempted",
        "field_goals.made",
        "two_point_field_goals.attempted",
        "two_point_field_goals.made",
        "three_point_field_goals.attempted",
        "three_point_field_goals.made",
        "free_throws.attempted",
        "free_throws.made",
        "rebounds.total",
        "rebounds.defensive",
        "rebounds.offensive",
        "win_shares.total",
        "win_shares.defensive",
        "win_shares.offensive",
    ]
    weighted_cols = [
        "usage",
        "offensive_rating",
        "defensive_rating",
        "net_rating",
        "porpag",
        "effective_field_goal_pct",
        "true_shooting_pct",
        "assists_turnover_ratio",
        "free_throw_rate",
        "offensive_rebound_pct",
        "field_goals.pct",
        "two_point_field_goals.pct",
        "three_point_field_goals.pct",
        "free_throws.pct",
        "win_shares.total_per40",
    ]

    sum_cols = [c for c in sum_cols if c in df.columns]
    weighted_cols = [c for c in weighted_cols if c in df.columns]
    out_rows: list[dict[str, object]] = []

    for athlete_id, g in df.groupby("athlete_id", sort=False):
        minutes = _safe_num(g["minutes"]) if "minutes" in g.columns else pd.Series([0.0] * len(g), index=g.index)
        top_idx = minutes.fillna(0.0).idxmax()
        top = g.loc[top_idx]

        row: dict[str, object] = {
            "season": int(top["season"]),
            "athlete_id": str(athlete_id),
            "name": top.get("name"),
            "team": top.get("team"),
            "conference": top.get("conference"),
            "position": top.get("position"),
        }
        for c in sum_cols:
            row[c] = _safe_num(g[c]).fillna(0.0).sum()
        for c in weighted_cols:
            row[c] = _weighted_mean(g[c], minutes)
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    minutes = _safe_num(out.get("minutes", pd.Series(dtype=float)))
    safe_minutes = minutes.replace(0, np.nan)
    for raw_col, per40_col in [
        ("points", "points_per40"),
        ("assists", "assists_per40"),
        ("rebounds.total", "rebounds_total_per40"),
        ("steals", "steals_per40"),
        ("blocks", "blocks_per40"),
        ("turnovers", "turnovers_per40"),
    ]:
        if raw_col in out.columns:
            out[per40_col] = 40.0 * _safe_num(out[raw_col]) / safe_minutes

    if "minutes" in out.columns and "games" in out.columns:
        out["minutes_per_game"] = _safe_num(out["minutes"]) / _safe_num(out["games"]).replace(0, np.nan)

    out = out.rename(
        columns={
            "rebounds.total": "rebounds_total",
            "three_point_field_goals.pct": "three_point_pct",
            "field_goals.pct": "field_goal_pct",
            "free_throws.pct": "free_throw_pct",
            "win_shares.total": "win_shares_total",
            "win_shares.defensive": "win_shares_defensive",
            "win_shares.offensive": "win_shares_offensive",
            "two_point_field_goals.pct": "two_point_pct",
        }
    )
    out["name_key"] = out["name"].map(normalize_name)
    out["team_key"] = out["team"].map(normalize_text)
    out["conference_tier"] = out["conference"].map(_conference_tier)
    return out


def load_player_season(players_root: Path, season: int) -> pd.DataFrame:
    path = players_root / f"season={season}" / "player_season_stats.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if df.empty or "athlete_id" not in df.columns:
        return pd.DataFrame()
    df["season"] = season
    out = aggregate_player_rows(df)
    return out


def _select_best_bio_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "status" in work.columns:
        work["_ok"] = (work["status"].astype(str).str.lower() == "ok").astype(int)
    else:
        work["_ok"] = 0
    cands = [c for c in ["height_in", "weight_lb", "age", "position_name"] if c in work.columns]
    work["_filled"] = work[cands].notna().sum(axis=1) if cands else 0
    work = work.sort_values(["athlete_id", "_ok", "_filled"], ascending=[True, False, False])
    return work.drop_duplicates(subset=["athlete_id"], keep="first")


def merge_bios(df: pd.DataFrame, bios_root: Path) -> pd.DataFrame:
    if df.empty:
        return df
    parts: list[pd.DataFrame] = []
    for season in sorted(df["season"].dropna().astype(int).unique().tolist()):
        path = bios_root / f"season={season}" / "player_bios.csv"
        if not path.exists():
            continue
        bio = pd.read_csv(path, low_memory=False)
        if bio.empty or "athlete_id" not in bio.columns:
            continue
        bio["season"] = season
        cols = ["season", "athlete_id", "height_in", "weight_lb", "age", "position_name", "status"]
        use = [c for c in cols if c in bio.columns]
        sub = _select_best_bio_rows(bio[use].copy())
        parts.append(sub[[c for c in cols if c in sub.columns]])
    if not parts:
        return df
    bdf = pd.concat(parts, ignore_index=True)
    bdf["athlete_id"] = bdf["athlete_id"].astype(str)

    out = df.copy()
    out["athlete_id"] = out["athlete_id"].astype(str)
    out = out.merge(bdf, how="left", on=["season", "athlete_id"], suffixes=("", "_bio"))
    if "position_name" in out.columns and "position" in out.columns:
        out["position_group"] = out["position"].fillna(out["position_name"])
    else:
        out["position_group"] = out.get("position")
    return out


def add_experience_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    first_seen = out.groupby("athlete_id", dropna=False)["season"].transform("min")
    out["first_seen_season"] = first_seen
    out["years_since_first_seen"] = (out["season"] - out["first_seen_season"] + 1).clip(lower=1)
    out["is_first_year"] = (out["years_since_first_seen"] == 1).astype(int)
    out["is_upperclass"] = (out["years_since_first_seen"] >= 3).astype(int)
    return out


def merge_external_model_db(df: pd.DataFrame, path: Path, backfill_max_gap: int = 2) -> pd.DataFrame:
    if df.empty or not path.exists():
        return df

    raw = pd.read_csv(path, low_memory=False)
    if raw.empty or "Name" not in raw.columns or "Season" not in raw.columns:
        return df

    raw = raw.copy()
    raw["name_key"] = raw["Name"].map(normalize_name)
    raw["ext_school_key"] = raw.get("School", pd.Series([""] * len(raw))).map(normalize_text)
    raw["ext_season"] = raw["Season"].map(_parse_external_season_end)
    raw = raw[raw["name_key"] != ""].copy()
    raw = raw[pd.to_numeric(raw["ext_season"], errors="coerce").notna()].copy()
    raw["ext_season"] = pd.to_numeric(raw["ext_season"], errors="coerce").astype(int)

    feature_map = {
        "Draft Day Age": "ext_draft_day_age",
        "RSCI": "ext_rsci",
        "Height": "ext_height_in_modeldb",
        "Weight": "ext_weight_lb_modeldb",
        "Draft Score": "ext_draft_score",
        "Finishing Score": "ext_finishing_score",
        "Shooting Score": "ext_shooting_score",
        "Shot Creation Score": "ext_shot_creation_score",
        "Passing Score": "ext_passing_score",
        "Rebounding Score": "ext_rebounding_score",
        "Athleticism Score": "ext_athleticism_score",
        "Defense Score": "ext_defense_score",
        "College Productivity Score": "ext_college_productivity_score",
        "Percentile Score": "ext_percentile_score",
        "Box Score Creation": "ext_box_score_creation",
        "Rim Shot Creation": "ext_rim_shot_creation",
        "Helio Score": "ext_helio_score",
    }

    keep_cols = ["name_key", "ext_season", "ext_school_key"] + [c for c in feature_map if c in raw.columns]
    ext = raw[keep_cols].rename(columns=feature_map).copy()
    ext = ext.sort_values(["name_key", "ext_season"]).drop_duplicates(subset=["name_key", "ext_season"], keep="last")
    ext_cols = [c for c in ext.columns if c not in {"name_key", "ext_season", "ext_school_key"}]
    for c in ext_cols:
        ext[c] = pd.to_numeric(ext[c], errors="coerce")

    out = df.copy()
    out = out.merge(
        ext,
        how="left",
        left_on=["name_key", "season"],
        right_on=["name_key", "ext_season"],
    )
    out["ext_match_type"] = np.where(out["ext_season"].notna(), "exact", "none")

    # Remove exact matches that clearly conflict on school/team key.
    exact_school_match = np.fromiter(
        (
            _team_college_match(str(t), str(s))
            for t, s in zip(
                out.get("team_key", pd.Series([""] * len(out))).fillna(""),
                out.get("ext_school_key", pd.Series([""] * len(out))).fillna(""),
            )
        ),
        dtype=bool,
        count=len(out),
    )
    has_team_key = out.get("team_key", pd.Series([""] * len(out))).fillna("").astype(str) != ""
    has_ext_school_key = out.get("ext_school_key", pd.Series([""] * len(out))).fillna("").astype(str) != ""
    bad_exact = (out["ext_match_type"] == "exact") & has_team_key & has_ext_school_key & (~exact_school_match)
    if bad_exact.any():
        clear_cols = ["ext_season", "ext_school_key"] + ext_cols
        for c in clear_cols:
            if c in out.columns:
                out.loc[bad_exact, c] = np.nan
        out.loc[bad_exact, "ext_match_type"] = "none"

    # Backfill from most recent prior season for same player name when exact-season row is absent.
    missing_idx = out.index[out["ext_season"].isna()].tolist()
    if missing_idx and backfill_max_gap > 0:
        grouped: dict[str, dict[str, object]] = {}
        for nk, g in ext.groupby("name_key", sort=False):
            gg = g.sort_values("ext_season").reset_index(drop=True)
            grouped[nk] = {
                "seasons": gg["ext_season"].to_numpy(dtype=int),
                "schools": gg["ext_school_key"].fillna("").astype(str).tolist(),
                "frame": gg,
            }

        for idx in missing_idx:
            nk = out.at[idx, "name_key"]
            season = out.at[idx, "season"]
            grp = grouped.get(nk)
            if grp is None or pd.isna(season):
                continue
            team_key = str(out.at[idx, "team_key"]) if "team_key" in out.columns else ""
            if not team_key:
                continue

            seasons = grp["seasons"]  # type: ignore[assignment]
            schools = grp["schools"]  # type: ignore[assignment]
            g = grp["frame"]  # type: ignore[assignment]
            target = int(season)
            pos = int(np.searchsorted(seasons, target, side="left")) - 1
            chosen = -1
            while pos >= 0:
                gap = target - int(seasons[pos])
                if gap <= 0:
                    pos -= 1
                    continue
                if gap > backfill_max_gap:
                    break
                if _team_college_match(team_key, str(schools[pos])):
                    chosen = pos
                    break
                pos -= 1
            if chosen < 0:
                continue

            row = g.iloc[chosen]
            out.at[idx, "ext_season"] = int(row["ext_season"])
            out.at[idx, "ext_school_key"] = row.get("ext_school_key")
            for c in ext_cols:
                out.at[idx, c] = row.get(c)
            out.at[idx, "ext_match_type"] = "backfill"

    # Track whether external school roughly matches current team.
    out["ext_school_match"] = np.fromiter(
        (
            _team_college_match(str(t), str(s))
            for t, s in zip(
                out.get("team_key", pd.Series([""] * len(out))).fillna(""),
                out.get("ext_school_key", pd.Series([""] * len(out))).fillna(""),
            )
        ),
        dtype=bool,
        count=len(out),
    ).astype(int)
    return out


def prepare_draft_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing draft labels: {path}")
    draft = pd.read_csv(path, low_memory=False)
    required = {"draft_year", "pick_overall", "player_name"}
    missing = [c for c in sorted(required) if c not in draft.columns]
    if missing:
        raise SystemExit(f"Draft labels missing columns: {missing}")
    draft = draft.copy()
    draft["draft_year"] = pd.to_numeric(draft["draft_year"], errors="coerce").astype("Int64")
    draft["pick_overall"] = pd.to_numeric(draft["pick_overall"], errors="coerce").astype("Int64")
    draft["name_key"] = draft["player_name"].map(normalize_name)
    draft["college_key"] = draft.get("college_name", pd.Series([""] * len(draft))).map(normalize_text)
    draft = draft.dropna(subset=["draft_year", "pick_overall"])
    draft = draft[draft["name_key"] != ""]
    draft = draft.sort_values(["draft_year", "pick_overall"], kind="stable")
    draft = draft.drop_duplicates(subset=["draft_year", "name_key"], keep="first")
    return draft[
        ["draft_year", "pick_overall", "pick_round", "player_name", "college_name", "name_key", "college_key"]
    ].copy()


def _iter_seasons(start: int, end: int) -> Iterable[int]:
    for s in range(start, end + 1):
        yield s


def main() -> None:
    args = parse_args()
    if args.end_season < args.start_season:
        raise SystemExit("--end-season must be >= --start-season")

    frames: list[pd.DataFrame] = []
    for season in _iter_seasons(args.start_season, args.end_season):
        season_df = load_player_season(players_root=args.players_root, season=season)
        if season_df.empty:
            print(f"season={season}: no player season stats found")
            continue
        before = len(season_df)
        season_df = season_df[
            (_safe_num(season_df.get("games", 0)).fillna(0.0) >= args.min_games)
            | (_safe_num(season_df.get("minutes", 0)).fillna(0.0) >= args.min_minutes)
        ].copy()
        print(f"season={season}: kept {len(season_df):,}/{before:,} players after min sample filter")
        frames.append(season_df)

    if not frames:
        raise SystemExit("No season rows available after filtering.")

    players = pd.concat(frames, ignore_index=True)
    players = merge_bios(players, bios_root=args.bios_root)
    players = add_experience_features(players)
    players = merge_external_model_db(
        players,
        path=args.external_model_db,
        backfill_max_gap=args.external_backfill_max_gap,
    )
    if "ext_match_type" in players.columns:
        vc = players["ext_match_type"].fillna("none").value_counts().to_dict()
        print(f"External feature matches: {vc}")

    draft = prepare_draft_labels(args.draft_csv)
    max_labeled_year = int(draft["draft_year"].max())
    merged = players.merge(
        draft,
        how="left",
        left_on=["season", "name_key"],
        right_on=["draft_year", "name_key"],
        suffixes=("", "_draft"),
    )

    merged["label_known"] = merged["season"] <= max_labeled_year
    merged["drafted"] = pd.NA
    merged.loc[merged["label_known"], "drafted"] = merged.loc[merged["label_known"], "pick_overall"].notna().astype(int)
    merged["pick_number"] = pd.to_numeric(merged["pick_overall"], errors="coerce")
    team_key = merged.get("team_key", pd.Series([""] * len(merged))).fillna("").astype(str).tolist()
    college_key = merged.get("college_key", pd.Series([""] * len(merged))).fillna("").astype(str).tolist()
    merged["draft_match_college_key"] = np.fromiter(
        (_team_college_match(t, c) for t, c in zip(team_key, college_key)),
        dtype=bool,
        count=len(merged),
    )

    col_order = [
        "season",
        "athlete_id",
        "name",
        "team",
        "conference",
        "conference_tier",
        "position",
        "position_group",
        "games",
        "starts",
        "minutes",
        "minutes_per_game",
        "points",
        "points_per40",
        "assists",
        "assists_per40",
        "rebounds_total",
        "rebounds_total_per40",
        "steals",
        "steals_per40",
        "blocks",
        "blocks_per40",
        "turnovers",
        "turnovers_per40",
        "usage",
        "offensive_rating",
        "defensive_rating",
        "net_rating",
        "porpag",
        "effective_field_goal_pct",
        "true_shooting_pct",
        "field_goal_pct",
        "two_point_pct",
        "three_point_pct",
        "free_throw_pct",
        "win_shares_total",
        "win_shares_total_per40",
        "height_in",
        "weight_lb",
        "age",
        "first_seen_season",
        "years_since_first_seen",
        "is_first_year",
        "is_upperclass",
        "ext_match_type",
        "ext_school_match",
        "ext_draft_day_age",
        "ext_rsci",
        "ext_height_in_modeldb",
        "ext_weight_lb_modeldb",
        "ext_draft_score",
        "ext_finishing_score",
        "ext_shooting_score",
        "ext_shot_creation_score",
        "ext_passing_score",
        "ext_rebounding_score",
        "ext_athleticism_score",
        "ext_defense_score",
        "ext_college_productivity_score",
        "ext_percentile_score",
        "ext_box_score_creation",
        "ext_rim_shot_creation",
        "ext_helio_score",
        "name_key",
        "team_key",
        "label_known",
        "drafted",
        "pick_number",
        "pick_round",
        "player_name",
        "college_name",
        "draft_match_college_key",
    ]
    cols = [c for c in col_order if c in merged.columns] + [c for c in merged.columns if c not in col_order]
    out = merged[cols].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    known = out[out["label_known"] == True]  # noqa: E712
    drafted = known[known["drafted"] == 1]
    print(f"Wrote {args.out}")
    print(f"Rows: {len(out):,} total | {len(known):,} labeled | {len(drafted):,} drafted")
    if not drafted.empty:
        by_season = drafted.groupby("season")["athlete_id"].count().to_dict()
        print(f"Drafted matches by season: {by_season}")


if __name__ == "__main__":
    main()
