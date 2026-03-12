from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NBA success dashboard (actual vs predicted holdout + class board).")
    p.add_argument(
        "--holdout-csv",
        type=Path,
        default=Path("data/processed/nba_new_joiner_impact_holdout_actual_vs_predicted_2022.csv"),
    )
    p.add_argument(
        "--pred-csv",
        type=Path,
        default=Path("data/processed/nba_new_joiner_impact_predictions_draft_2025.csv"),
    )
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("data/processed/nba_new_joiner_impact_model_metrics.csv"),
    )
    p.add_argument(
        "--out-html",
        type=Path,
        default=Path("data/processed/viz/nba_impact_dashboard_real_vs_predicted_2022.html"),
    )
    p.add_argument(
        "--rapm-csv",
        type=Path,
        default=Path("data/raw/external/nbarapm/TimedecayRAPM.csv"),
        help="Optional current RAPM table to display actual RAPM columns.",
    )
    p.add_argument(
        "--draft-history-csv",
        type=Path,
        default=Path("data/raw/nba/draft/nba_draft_history_2015_2025.csv"),
        help="Optional NBA draft history CSV for draft-team enrichment by year+pick.",
    )
    p.add_argument(
        "--combine-dir",
        type=Path,
        default=Path("data/raw/external/nba_stats_draft/antro/antro"),
        help="Optional NBA combine anthro directory (Draft_antro_YYYY.csv).",
    )
    p.add_argument(
        "--nba-advanced-csv",
        type=Path,
        default=Path("data/raw/nba/bref/player_advanced_2010_2026.csv"),
        help="Optional NBA advanced seasons CSV for NBA career start/year counts.",
    )
    p.add_argument(
        "--nba-display-csv",
        type=Path,
        default=Path("data/raw/external/nbarapm/player_stats_export.csv"),
        help="Optional NBA player stats export CSV for display-only NBA stat summaries.",
    )
    p.add_argument(
        "--crafted-measurements-csv",
        type=Path,
        default=Path("data/raw/external/craftednba/player_traits_length.csv"),
        help="Optional CraftedNBA measurements CSV (height/wingspan).",
    )
    p.add_argument(
        "--training-csv",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
        help="Optional training table for college season-by-season history payloads.",
    )
    return p.parse_args()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _season_label(end_year: object) -> str:
    y = pd.to_numeric(pd.Series([end_year]), errors="coerce").iloc[0]
    if pd.isna(y):
        return ""
    y = int(float(y))
    if y < 1901:
        return ""
    return f"{y - 1}/{str(y)[-2:]}"


def _coalesce_numeric(
    df: pd.DataFrame,
    candidates: list[tuple[str, str]],
) -> tuple[pd.Series, pd.Series]:
    vals = pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    src = pd.Series([""] * len(df), index=df.index, dtype="object")
    for col, label in candidates:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        mask = vals.isna() & x.notna()
        if mask.any():
            vals.loc[mask] = x.loc[mask]
            src.loc[mask] = label
    return vals, src


def _impute_linear(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    min_fit_rows: int = 12,
) -> tuple[pd.Series, pd.Series]:
    y = pd.to_numeric(df.get(target_col), errors="coerce")
    xs = [pd.to_numeric(df.get(c), errors="coerce") for c in feature_cols]
    valid = y.notna().copy()
    for x in xs:
        valid &= x.notna()
    if int(valid.sum()) < min_fit_rows:
        return y, pd.Series([False] * len(df), index=df.index, dtype=bool)

    x_mat = np.column_stack([np.ones(int(valid.sum()))] + [x.loc[valid].to_numpy(dtype=float) for x in xs])
    y_vec = y.loc[valid].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)

    miss = y.isna().copy()
    for x in xs:
        miss &= x.notna()
    if not miss.any():
        return y, pd.Series([False] * len(df), index=df.index, dtype=bool)

    pred_mat = np.column_stack([np.ones(int(miss.sum()))] + [x.loc[miss].to_numpy(dtype=float) for x in xs])
    y_out = y.copy()
    y_out.loc[miss] = pred_mat @ beta
    return y_out, miss


def _map_to_empirical_rapm(
    pred_scores: pd.Series,
    calib_pred: pd.Series,
    calib_actual: pd.Series,
) -> pd.Series:
    p = pd.to_numeric(pred_scores, errors="coerce")
    cp = pd.to_numeric(calib_pred, errors="coerce")
    ca = pd.to_numeric(calib_actual, errors="coerce")
    valid = cp.notna() & ca.notna()
    if int(valid.sum()) < 8:
        return pd.Series([float("nan")] * len(p), index=p.index, dtype=float)

    cpv = cp.loc[valid].to_numpy(dtype=float)
    cav = ca.loc[valid].to_numpy(dtype=float)
    cp_sorted = np.sort(cpv)
    # Target distribution from observed actual RAPM values.
    ca_sorted = np.sort(cav)
    qgrid = np.linspace(0.0, 1.0, len(cp_sorted))

    out = pd.Series([float("nan")] * len(p), index=p.index, dtype=float)
    mask = p.notna()
    if not bool(mask.any()):
        return out
    pct = np.interp(
        p.loc[mask].to_numpy(dtype=float),
        cp_sorted,
        qgrid,
        left=0.0,
        right=1.0,
    )
    out.loc[mask] = np.quantile(ca_sorted, pct)
    return out


def _name_key(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.lower().strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def load_combine_lookup(combine_dir: Path) -> pd.DataFrame:
    if not combine_dir.exists():
        return pd.DataFrame(columns=["season", "name_key"])

    rows: list[pd.DataFrame] = []
    for p in sorted(combine_dir.glob("Draft_antro_*.csv")):
        m = re.search(r"Draft_antro_(\d{4})\.csv$", p.name)
        if not m:
            continue
        draft_year = int(m.group(1))
        df = pd.read_csv(p, low_memory=False)
        if "PLAYER_NAME" not in df.columns:
            continue
        keep = {
            "PLAYER_NAME": "combine_player_name",
            "HEIGHT_WO_SHOES": "combine_height_wo_shoes_in",
            "WEIGHT": "combine_weight_lb",
            "WINGSPAN": "combine_wingspan_in",
            "STANDING_REACH": "combine_standing_reach_in",
            "BODY_FAT_PCT": "combine_body_fat_pct",
            "HAND_LENGTH": "combine_hand_length_in",
            "HAND_WIDTH": "combine_hand_width_in",
        }
        cols = [c for c in keep if c in df.columns]
        if not cols:
            continue
        out = df[cols].rename(columns={c: keep[c] for c in cols}).copy()
        out["season"] = draft_year
        out["name_key"] = out["combine_player_name"].map(_name_key)
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["season", "name_key"])

    cdf = pd.concat(rows, ignore_index=True)
    num_cols = [
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "combine_body_fat_pct",
        "combine_hand_length_in",
        "combine_hand_width_in",
    ]
    for c in num_cols:
        if c in cdf.columns:
            cdf[c] = pd.to_numeric(cdf[c], errors="coerce")
    cdf["combine_wingspan_minus_height"] = cdf["combine_wingspan_in"] - cdf["combine_height_wo_shoes_in"]
    cdf["combine_standing_reach_minus_height"] = (
        cdf["combine_standing_reach_in"] - cdf["combine_height_wo_shoes_in"]
    )
    cdf = cdf[cdf["name_key"] != ""].copy()
    cdf = cdf.sort_values(["season", "name_key"]).drop_duplicates(subset=["season", "name_key"], keep="first")
    return cdf


def load_crafted_lookup(crafted_csv: Path) -> pd.DataFrame:
    if not crafted_csv.exists():
        return pd.DataFrame(columns=["name_key"])
    df = pd.read_csv(crafted_csv, low_memory=False)
    if "name_key" not in df.columns:
        if "player_name" in df.columns:
            df["name_key"] = df["player_name"].map(_name_key)
        else:
            return pd.DataFrame(columns=["name_key"])
    keep = ["name_key"]
    for c in ["crafted_height_in", "crafted_wingspan_in", "crafted_length_in", "player_name", "crafted_slug"]:
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy()
    for c in ["crafted_height_in", "crafted_wingspan_in", "crafted_length_in"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if {"crafted_wingspan_in", "crafted_height_in"}.issubset(set(out.columns)):
        out["crafted_wingspan_minus_height"] = out["crafted_wingspan_in"] - out["crafted_height_in"]
    out = out[out["name_key"] != ""].copy()
    out = out.sort_values(["name_key"], kind="stable").drop_duplicates(subset=["name_key"], keep="first")
    return out


def load_nba_career_lookup(advanced_csv: Path) -> pd.DataFrame:
    if not advanced_csv.exists():
        return pd.DataFrame(columns=["name_key"])
    df = pd.read_csv(advanced_csv, low_memory=False)
    if "Player" not in df.columns or "year" not in df.columns:
        return pd.DataFrame(columns=["name_key"])
    out = df[["Player", "year"]].copy()
    out["name_key"] = out["Player"].map(_name_key)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["year"]).copy()
    if out.empty:
        return pd.DataFrame(columns=["name_key"])
    grp = (
        out.groupby("name_key", dropna=False)["year"]
        .agg(nba_start_year="min", nba_last_year="max", nba_years_played="nunique")
        .reset_index()
    )
    return grp


def _combine_nba_season_rows(primary: object, fallback: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_years: set[int] = set()
    for src in (primary, fallback):
        if not isinstance(src, list):
            continue
        for item in src:
            if not isinstance(item, dict):
                continue
            yr = pd.to_numeric(pd.Series([item.get("season_end_year")]), errors="coerce").iloc[0]
            if pd.isna(yr):
                continue
            y = int(float(yr))
            if y in seen_years:
                continue
            seen_years.add(y)
            rows.append(item)
    rows.sort(key=lambda d: int(pd.to_numeric(pd.Series([d.get("season_end_year")]), errors="coerce").fillna(-1).iloc[0]))
    return rows


def _backfill_nba_season_team_rows(
    season_rows: object,
    nba_team_abbr: object,
    nba_team_display: object,
    nba_start_year: object,
) -> object:
    if not isinstance(season_rows, list):
        return season_rows
    team_abbr = str("" if nba_team_abbr is None else nba_team_abbr).strip()
    team_display = str("" if nba_team_display is None else nba_team_display).strip()
    fallback_team = team_abbr or team_display
    if not fallback_team:
        return season_rows

    start_yr = pd.to_numeric(pd.Series([nba_start_year]), errors="coerce").iloc[0]
    start_yr_int = int(float(start_yr)) if pd.notna(start_yr) else None

    rows: list[dict[str, object]] = []
    filled = False
    blank_nba_idx: list[int] = []
    blank_like = {"", "nan", "na", "n/a", "null", "none", "undefined"}
    for i, item in enumerate(season_rows):
        if not isinstance(item, dict):
            rows.append(item)
            continue
        row = dict(item)
        team = str(row.get("team", "")).strip()
        team_norm = team.lower()
        kind = str(row.get("kind", "")).strip().lower()
        yr = pd.to_numeric(pd.Series([row.get("season_end_year")]), errors="coerce").iloc[0]
        yr_int = int(float(yr)) if pd.notna(yr) else None
        if kind == "nba" and team_norm in blank_like:
            row["team"] = ""
            blank_nba_idx.append(i)
            if start_yr_int is not None and yr_int == start_yr_int:
                row["team"] = fallback_team
                filled = True
        rows.append(row)

    if not filled and blank_nba_idx:
        # If nba_start_year is unavailable, fill the earliest NBA season gap.
        blank_pairs: list[tuple[int, int]] = []
        for i in blank_nba_idx:
            item = rows[i]
            if not isinstance(item, dict):
                continue
            yr = pd.to_numeric(pd.Series([item.get("season_end_year")]), errors="coerce").iloc[0]
            yr_int = int(float(yr)) if pd.notna(yr) else 9999
            blank_pairs.append((yr_int, i))
        if blank_pairs:
            blank_pairs.sort(key=lambda t: t[0])
            rows[blank_pairs[0][1]]["team"] = fallback_team
    return rows


def load_nba_advanced_season_lookup(advanced_csv: Path) -> pd.DataFrame:
    if not advanced_csv.exists():
        return pd.DataFrame(columns=["name_key"])
    df = pd.read_csv(advanced_csv, low_memory=False)
    if "Player" not in df.columns or "year" not in df.columns:
        return pd.DataFrame(columns=["name_key"])

    out = df.copy()
    out["name_key"] = out["Player"].map(_name_key)
    out["season_end_year"] = pd.to_numeric(out["year"], errors="coerce")
    out["nba_games_adv"] = pd.to_numeric(out.get("G"), errors="coerce")
    out["nba_minutes_adv"] = pd.to_numeric(out.get("MP"), errors="coerce")
    out["nba_bpm_adv"] = pd.to_numeric(out.get("BPM"), errors="coerce")
    out["nba_ws48_adv"] = pd.to_numeric(out.get("WS/48"), errors="coerce")
    out["nba_team_adv"] = out.get("Tm", pd.Series([""] * len(out), index=out.index)).astype(str).str.strip()
    out["nba_position_adv"] = out.get("Pos", pd.Series([""] * len(out), index=out.index)).astype(str).str.strip()
    out = out[(out["name_key"] != "") & out["season_end_year"].notna()].copy()
    if out.empty:
        return pd.DataFrame(columns=["name_key"])
    out["season_end_year"] = out["season_end_year"].astype(int)

    rows: list[dict[str, object]] = []
    for nk, g in out.groupby("name_key", sort=False, dropna=False):
        season_rows: list[dict[str, object]] = []
        team_hist: list[str] = []
        pos_hist: list[str] = []
        g = g.sort_values(["season_end_year"], kind="stable")
        for yr, sy in g.groupby("season_end_year", sort=True):
            sy = sy.copy()
            non_tot_teams = [
                t for t in sy["nba_team_adv"].astype(str).tolist() if t and t.lower() != "nan" and t != "TOT"
            ]
            teams = list(dict.fromkeys(non_tot_teams))
            poss = [p for p in sy["nba_position_adv"].astype(str).tolist() if p and p.lower() != "nan"]
            poss = list(dict.fromkeys(poss))

            tot = sy[sy["nba_team_adv"] == "TOT"]
            if not tot.empty:
                best = tot.sort_values(["nba_minutes_adv", "nba_games_adv"], ascending=[False, False], kind="stable").iloc[0]
            else:
                best = sy.sort_values(["nba_minutes_adv", "nba_games_adv"], ascending=[False, False], kind="stable").iloc[0]

            season_label = _season_label(int(yr))
            if teams:
                team_hist.append(f"{'/'.join(teams)} ({season_label})")
            if poss:
                pos_hist.append(f"{'/'.join(poss)} ({season_label})")

            season_rows.append(
                {
                    "season_end_year": int(yr),
                    "season_label": season_label,
                    "kind": "nba",
                    "team": "/".join(teams) if teams else str(best.get("nba_team_adv", "")).strip(),
                    "position": "/".join(poss) if poss else str(best.get("nba_position_adv", "")).strip(),
                    "games": pd.to_numeric(best.get("nba_games_adv"), errors="coerce"),
                    "minutes": pd.to_numeric(best.get("nba_minutes_adv"), errors="coerce"),
                    "bpm": pd.to_numeric(best.get("nba_bpm_adv"), errors="coerce"),
                    "ws48": pd.to_numeric(best.get("nba_ws48_adv"), errors="coerce"),
                }
            )

        latest = g.sort_values(["season_end_year", "nba_minutes_adv"], ascending=[False, False], kind="stable").iloc[0]
        rows.append(
            {
                "name_key": nk,
                "nba_team_history_adv": ", ".join(team_hist),
                "nba_position_history_adv": ", ".join(pos_hist),
                "nba_seasons_adv": season_rows,
                "nba_team_display_adv": str(latest.get("nba_team_adv", "")).strip(),
                "nba_position_adv": str(latest.get("nba_position_adv", "")).strip(),
                "nba_games_adv": pd.to_numeric(latest.get("nba_games_adv"), errors="coerce"),
                "nba_minutes_adv": pd.to_numeric(latest.get("nba_minutes_adv"), errors="coerce"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["name_key"])
    return pd.DataFrame(rows)


def load_nba_display_lookup(display_csv: Path) -> pd.DataFrame:
    if not display_csv.exists():
        return pd.DataFrame(columns=["name_key"])
    df = pd.read_csv(display_csv, low_memory=False)
    name_col = None
    for c in ["ShortName", "player_name", "Player"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return pd.DataFrame(columns=["name_key"])

    keep_map = {
        name_col: "player_name",
        "year": "nba_display_year",
        "TeamAbbreviation": "nba_team_display",
        "Pos": "nba_position",
        "GamesPlayed": "nba_games",
        "Minutes": "nba_minutes",
        "Pts75": "nba_points_per75",
        "PASSING_Assists/100": "nba_assists_per100",
        "Steals_100": "nba_steals_per100",
        "Blocks_100": "nba_blocks_per100",
        "TS_percent": "nba_ts_pct",
        "3P_PERC": "nba_three_point_pct",
        "3PA_100": "nba_three_pa_per100",
        "FTOV_100": "nba_tov_per100",
    }
    cols = [c for c in keep_map if c in df.columns]
    out = df[cols].rename(columns={c: keep_map[c] for c in cols}).copy()
    out["name_key"] = out["player_name"].map(_name_key)
    if "nba_display_year" in out.columns:
        out["nba_display_year"] = pd.to_numeric(out["nba_display_year"], errors="coerce")
    for c in [
        "nba_games",
        "nba_minutes",
        "nba_points_per75",
        "nba_assists_per100",
        "nba_steals_per100",
        "nba_blocks_per100",
        "nba_ts_pct",
        "nba_three_point_pct",
        "nba_three_pa_per100",
        "nba_tov_per100",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if {"nba_steals_per100", "nba_blocks_per100"}.issubset(set(out.columns)):
        out["nba_stocks_per100"] = out["nba_steals_per100"] + out["nba_blocks_per100"]
    out = out[out["name_key"] != ""].copy()
    if out.empty:
        return out

    # Build readable season-by-season history strings and payload rows.
    hist_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    if "nba_display_year" in out.columns:
        for nk, g in out.groupby("name_key", sort=False, dropna=False):
            teams_hist: list[str] = []
            pos_hist: list[str] = []
            gy = g.dropna(subset=["nba_display_year"]).copy()
            if gy.empty:
                continue
            gy["nba_display_year"] = gy["nba_display_year"].astype(int)
            for yr, h in gy.groupby("nba_display_year", sort=True):
                season_lbl = _season_label(yr)
                teams = [
                    t
                    for t in h.get("nba_team_display", pd.Series([], dtype=object)).astype(str).str.strip().tolist()
                    if t and t.lower() != "nan" and t != "TOT"
                ]
                teams = list(dict.fromkeys(teams))
                poss = [
                    p
                    for p in h.get("nba_position", pd.Series([], dtype=object)).astype(str).str.strip().tolist()
                    if p and p.lower() != "nan"
                ]
                poss = list(dict.fromkeys(poss))
                if teams:
                    teams_hist.append(f"{'/'.join(teams)} ({season_lbl})")
                if poss:
                    pos_hist.append(f"{'/'.join(poss)} ({season_lbl})")

                hr = h.sort_values(["nba_minutes", "nba_games"], ascending=[False, False], kind="stable").iloc[0]
                season_rows.append(
                    {
                        "name_key": nk,
                        "season_end_year": int(yr),
                        "season_label": season_lbl,
                        "kind": "nba",
                        "team": "/".join(teams) if teams else str(hr.get("nba_team_display", "")).strip(),
                        "position": "/".join(poss) if poss else str(hr.get("nba_position", "")).strip(),
                        "games": pd.to_numeric(hr.get("nba_games"), errors="coerce"),
                        "minutes": pd.to_numeric(hr.get("nba_minutes"), errors="coerce"),
                        "points_per75": pd.to_numeric(hr.get("nba_points_per75"), errors="coerce"),
                        "assists_per100": pd.to_numeric(hr.get("nba_assists_per100"), errors="coerce"),
                        "steals_per100": pd.to_numeric(hr.get("nba_steals_per100"), errors="coerce"),
                        "blocks_per100": pd.to_numeric(hr.get("nba_blocks_per100"), errors="coerce"),
                        "stocks_per100": pd.to_numeric(hr.get("nba_stocks_per100"), errors="coerce"),
                        "ts_pct": pd.to_numeric(hr.get("nba_ts_pct"), errors="coerce"),
                        "three_point_pct": pd.to_numeric(hr.get("nba_three_point_pct"), errors="coerce"),
                        "three_pa_per100": pd.to_numeric(hr.get("nba_three_pa_per100"), errors="coerce"),
                        "tov_per100": pd.to_numeric(hr.get("nba_tov_per100"), errors="coerce"),
                    }
                )
            hist_rows.append(
                {
                    "name_key": nk,
                    "nba_team_history": ", ".join(teams_hist),
                    "nba_position_history": ", ".join(pos_hist),
                }
            )
    hist_df = (
        pd.DataFrame(hist_rows)
        if hist_rows
        else pd.DataFrame(columns=["name_key", "nba_team_history", "nba_position_history"])
    )
    seasons_df = (
        pd.DataFrame(season_rows)
        if season_rows
        else pd.DataFrame(columns=["name_key", "season_end_year", "season_label", "kind"])
    )

    sort_cols: list[str] = []
    if "nba_display_year" in out.columns:
        sort_cols.append("nba_display_year")
    if "nba_minutes" in out.columns:
        sort_cols.append("nba_minutes")
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="stable")
    else:
        out = out.sort_values(["name_key"], kind="stable")
    out = out.drop_duplicates(subset=["name_key"], keep="first")
    if not hist_df.empty:
        out = out.merge(hist_df, how="left", on="name_key")
    if not seasons_df.empty:
        agg = (
            seasons_df.sort_values(["name_key", "season_end_year"], kind="stable")
            .groupby("name_key", dropna=False)
            .apply(lambda g: g.drop(columns=["name_key"]).to_dict(orient="records"))
            .rename("nba_seasons")
            .reset_index()
        )
        out = out.merge(agg, how="left", on="name_key")
    return out


def load_college_season_lookup(training_csv: Path) -> pd.DataFrame:
    if not training_csv.exists():
        return pd.DataFrame(columns=["name_key", "season", "college_seasons", "college_team_history", "college_position_history"])
    df = pd.read_csv(training_csv, low_memory=False)
    if not {"name", "season", "team"}.issubset(set(df.columns)):
        return pd.DataFrame(columns=["name_key", "season", "college_seasons", "college_team_history", "college_position_history"])
    w = df.copy()
    w["name_key"] = w["name"].map(_name_key)
    w["season"] = pd.to_numeric(w["season"], errors="coerce")
    w["minutes"] = pd.to_numeric(w.get("minutes"), errors="coerce")
    w = w.dropna(subset=["season"]).copy()
    w = w[w["name_key"] != ""].copy()
    w = w.sort_values(["name_key", "season", "minutes"], ascending=[True, True, False], kind="stable")
    w = w.drop_duplicates(subset=["name_key", "season"], keep="first").copy()

    for c in [
        "games",
        "minutes",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "steals_per40",
        "blocks_per40",
        "usage",
        "true_shooting_pct",
        "three_point_pct",
        "three_point_attempt_rate",
        "turnovers_per40",
        "net_rating",
    ]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    if "stocks_per40" not in w.columns:
        w["stocks_per40"] = pd.to_numeric(w.get("steals_per40"), errors="coerce").fillna(0.0) + pd.to_numeric(
            w.get("blocks_per40"), errors="coerce"
        ).fillna(0.0)
    if "assist_to_turnover" not in w.columns:
        ast = pd.to_numeric(w.get("assists_per40"), errors="coerce")
        tov = pd.to_numeric(w.get("turnovers_per40"), errors="coerce")
        w["assist_to_turnover"] = ast / tov.replace(0, pd.NA)

    rows: list[dict[str, object]] = []
    for nk, g in w.groupby("name_key", sort=False, dropna=False):
        g = g.sort_values(["season"], kind="stable")
        team_hist: list[str] = []
        pos_hist: list[str] = []
        season_payload: list[dict[str, object]] = []
        for _, r in g.iterrows():
            sy = int(float(r["season"]))
            sl = _season_label(sy)
            team = str(r.get("team", "")).strip()
            pos = str(r.get("position", "")).strip()
            if not pos:
                pos = str(r.get("position_group", "")).strip()
            if team and team.lower() != "nan":
                team_hist.append(f"{team} ({sl})")
            if pos and pos.lower() != "nan":
                pos_hist.append(f"{pos} ({sl})")
            season_payload.append(
                {
                    "season_end_year": sy,
                    "season_label": sl,
                    "kind": "college",
                    "team": team,
                    "position": pos,
                    "games": pd.to_numeric(r.get("games"), errors="coerce"),
                    "minutes": pd.to_numeric(r.get("minutes"), errors="coerce"),
                    "points_per40": pd.to_numeric(r.get("points_per40"), errors="coerce"),
                    "assists_per40": pd.to_numeric(r.get("assists_per40"), errors="coerce"),
                    "rebounds_total_per40": pd.to_numeric(r.get("rebounds_total_per40"), errors="coerce"),
                    "steals_per40": pd.to_numeric(r.get("steals_per40"), errors="coerce"),
                    "blocks_per40": pd.to_numeric(r.get("blocks_per40"), errors="coerce"),
                    "stocks_per40": pd.to_numeric(r.get("stocks_per40"), errors="coerce"),
                    "usage": pd.to_numeric(r.get("usage"), errors="coerce"),
                    "true_shooting_pct": pd.to_numeric(r.get("true_shooting_pct"), errors="coerce"),
                    "three_point_pct": pd.to_numeric(r.get("three_point_pct"), errors="coerce"),
                    "three_point_attempt_rate": pd.to_numeric(r.get("three_point_attempt_rate"), errors="coerce"),
                    "assist_to_turnover": pd.to_numeric(r.get("assist_to_turnover"), errors="coerce"),
                    "turnovers_per40": pd.to_numeric(r.get("turnovers_per40"), errors="coerce"),
                    "net_rating": pd.to_numeric(r.get("net_rating"), errors="coerce"),
                }
            )
            rows.append(
                {
                    "name_key": nk,
                    "season": float(sy),
                    "college_seasons": list(season_payload),
                    "college_team_history": ", ".join(team_hist),
                    "college_position_history": ", ".join(pos_hist),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["name_key", "season", "college_seasons", "college_team_history", "college_position_history"])
    out = pd.DataFrame(rows)
    out = out.sort_values(["name_key", "season"], kind="stable").drop_duplicates(subset=["name_key", "season"], keep="last")
    return out


def rapm_label(v: object) -> str:
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if pd.isna(x):
        return ""
    x = float(x)
    if x >= 3.0:
        return "fantastic"
    if x >= 0.75:
        return "excellent"
    if x >= -0.25:
        return "good"
    if x >= -1.0:
        return "average"
    return "bad"


def label_to_ord(lbl: object) -> float:
    m = {"bad": 0.0, "average": 1.0, "good": 2.0, "excellent": 3.0, "fantastic": 4.0}
    s = "" if lbl is None else str(lbl).strip().lower()
    return m.get(s, float("nan"))


def _build_driver_and_miss_text(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "stocks_per40" not in df.columns:
        df["stocks_per40"] = pd.to_numeric(df.get("blocks_per40"), errors="coerce").fillna(0.0) + pd.to_numeric(
            df.get("steals_per40"), errors="coerce"
        ).fillna(0.0)
    if "assist_to_turnover" not in df.columns:
        ast = pd.to_numeric(df.get("assists_per40"), errors="coerce")
        tov = pd.to_numeric(df.get("turnovers_per40"), errors="coerce")
        df["assist_to_turnover"] = ast / tov.replace(0, pd.NA)

    metric_defs = [
        ("minutes", "minutes", True, lambda v: f"MIN {int(round(v))}"),
        ("true_shooting_pct", "TS", True, lambda v: f"TS {v:.3f}"),
        ("three_point_attempt_rate", "3PA rate", True, lambda v: f"3PA/FGA {v:.3f}"),
        ("three_point_pct", "3P%", True, lambda v: f"3P% {v:.1f}"),
        ("points_per40", "scoring", True, lambda v: f"PTS/40 {v:.1f}"),
        ("assists_per40", "playmaking", True, lambda v: f"AST/40 {v:.1f}"),
        ("rebounds_total_per40", "rebounding", True, lambda v: f"REB/40 {v:.1f}"),
        ("stocks_per40", "stocks", True, lambda v: f"STL+BLK/40 {v:.1f}"),
        ("net_rating", "net", True, lambda v: f"NET {v:.1f}"),
        ("turnovers_per40", "turnovers", False, lambda v: f"TOV/40 {v:.1f}"),
        ("measurement_wingspan_in", "wingspan", True, lambda v: f"WING {v:.1f}"),
        ("measurement_height_in", "height", True, lambda v: f"HT {v:.1f}"),
        ("measurement_standing_reach_in", "reach", True, lambda v: f"REACH {v:.1f}"),
    ]
    metric_fmt = {
        "minutes": lambda v: f"MIN {int(round(v))}",
        "three_point_attempt_rate": lambda v: f"3PA/FGA {v:.3f}",
        "three_point_pct": lambda v: f"3P% {v:.1f}",
        "true_shooting_pct": lambda v: f"TS {v:.3f}",
        "assists_per40": lambda v: f"AST/40 {v:.1f}",
        "turnovers_per40": lambda v: f"TOV/40 {v:.1f}",
        "assist_to_turnover": lambda v: f"AST/TOV {v:.2f}",
        "stocks_per40": lambda v: f"STL+BLK/40 {v:.1f}",
        "rebounds_total_per40": lambda v: f"REB/40 {v:.1f}",
        "usage": lambda v: f"USG {v:.1f}",
        "net_rating": lambda v: f"NET {v:.1f}",
        "measurement_wingspan_in": lambda v: f"WING {v:.1f}",
        "measurement_height_in": lambda v: f"HT {v:.1f}",
        "measurement_standing_reach_in": lambda v: f"REACH {v:.1f}",
    }
    higher_better = {
        "minutes": True,
        "three_point_attempt_rate": True,
        "three_point_pct": True,
        "true_shooting_pct": True,
        "assists_per40": True,
        "turnovers_per40": False,
        "assist_to_turnover": True,
        "stocks_per40": True,
        "rebounds_total_per40": True,
        "usage": True,
        "net_rating": True,
        "measurement_wingspan_in": True,
        "measurement_height_in": True,
        "measurement_standing_reach_in": True,
    }
    group_defs = {
        "shooting": ["three_point_attempt_rate", "three_point_pct", "true_shooting_pct"],
        "playmaking": ["assists_per40", "assist_to_turnover", "turnovers_per40"],
        "defense/rebounding": ["stocks_per40", "rebounds_total_per40", "net_rating"],
        "role": ["minutes", "usage"],
        "physical tools": [
            "measurement_wingspan_in",
            "measurement_height_in",
            "measurement_standing_reach_in",
        ],
    }

    for col, _, _, _ in metric_defs:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in higher_better:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pct_col_map: dict[str, str] = {}
    for col, label, higher_is_better, _ in metric_defs:
        s = pd.to_numeric(df.get(col), errors="coerce")
        if not higher_is_better:
            s = -s
        pcol = f"pct_{re.sub(r'[^a-z0-9]+', '_', label.lower())}"
        df[pcol] = s.rank(pct=True, method="average")
        pct_col_map[col] = pcol

    for col, hb in higher_better.items():
        if col in pct_col_map:
            continue
        s = pd.to_numeric(df.get(col), errors="coerce")
        if not hb:
            s = -s
        pcol = f"pct_{re.sub(r'[^a-z0-9]+', '_', col.lower())}"
        df[pcol] = s.rank(pct=True, method="average")
        pct_col_map[col] = pcol

    def _group_signal(row: pd.Series, group_name: str, mode: str) -> str:
        cols = group_defs[group_name]
        scored: list[tuple[float, str]] = []
        for c in cols:
            pcol = pct_col_map.get(c)
            p = row.get(pcol) if pcol else float("nan")
            v = pd.to_numeric(row.get(c), errors="coerce")
            if pd.isna(p) or pd.isna(v):
                continue
            fmt = metric_fmt.get(c, lambda x: f"{c} {x:.2f}")
            scored.append((float(p), fmt(float(v))))
        if not scored:
            return ""
        scored.sort(key=lambda x: x[0], reverse=(mode == "strong"))
        return scored[0][1]

    def _group_strength(row: pd.Series, group_name: str) -> float:
        vals: list[float] = []
        for c in group_defs[group_name]:
            pcol = pct_col_map.get(c)
            p = row.get(pcol) if pcol else float("nan")
            if pd.notna(p):
                vals.append(float(p))
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    pos_driver: list[str] = []
    neg_driver: list[str] = []
    miss_explain: list[str] = []
    for _, r in df.iterrows():
        scored: list[tuple[float, str]] = []
        weak_scored: list[tuple[float, str]] = []
        for col, label, _, fmt_fn in metric_defs:
            p = r.get(pct_col_map[col])
            v = r.get(col)
            if pd.isna(p) or pd.isna(v):
                continue
            phrase = fmt_fn(float(v))
            scored.append((float(p), phrase))
            weak_scored.append((float(p), phrase))

        scored.sort(key=lambda x: x[0], reverse=True)
        weak_scored.sort(key=lambda x: x[0])
        pos = [t for p, t in scored if p >= 0.65][:2]
        neg = [t for p, t in weak_scored if p <= 0.35][:2]
        if not pos and scored:
            pos = [scored[0][1]]
        if not neg and weak_scored:
            neg = [weak_scored[0][1]]
        pos_driver.append("; ".join(pos))
        neg_driver.append("; ".join(neg))

        actual_rank = pd.to_numeric(r.get("actual_rank"), errors="coerce")
        pred_rank = pd.to_numeric(r.get("pred_rank"), errors="coerce")
        rank_err = pd.to_numeric(r.get("rank_error"), errors="coerce")
        abs_rank_err = abs(float(rank_err)) if pd.notna(rank_err) else float("nan")
        overpred = pd.notna(actual_rank) and pd.notna(pred_rank) and float(pred_rank) < float(actual_rank)

        if pd.isna(actual_rank) or pd.isna(pred_rank) or pd.isna(rank_err):
            miss_explain.append("Rank unavailable for this player.")
        elif abs_rank_err <= 10:
            miss_explain.append("")
        else:
            g_scores: list[tuple[float, str]] = []
            for gn in group_defs:
                s = _group_strength(r, gn)
                if pd.notna(s):
                    g_scores.append((float(s), gn))
            if not g_scores:
                miss_explain.append("Profile conflict across signals; label miss persists.")
                continue
            g_scores.sort(key=lambda x: x[0], reverse=True)
            strong_g = g_scores[0][1]
            weak_g = g_scores[-1][1]
            strong_sig = _group_signal(r, strong_g, "strong")
            weak_sig = _group_signal(r, weak_g, "weak")
            if overpred:
                miss_explain.append(
                    f"Rank miss {abs_rank_err:.0f}: too optimistic on {strong_g} ({strong_sig}); weak {weak_g} ({weak_sig})."
                )
            else:
                miss_explain.append(
                    f"Rank miss {abs_rank_err:.0f}: too pessimistic due to weak {weak_g} ({weak_sig}) despite strong {strong_g} ({strong_sig})."
                )

    return pd.DataFrame(
        {
            "driver_positive_short": pd.Series(pos_driver, index=df_in.index, dtype="object"),
            "driver_negative_short": pd.Series(neg_driver, index=df_in.index, dtype="object"),
            "miss_explain_short": pd.Series(miss_explain, index=df_in.index, dtype="object"),
        },
        index=df_in.index,
    )


def build_html(
    holdout_json: str,
    board_json: str,
    metrics_json: str,
    holdout_year: str,
    predict_year: str,
    year_options_json: str,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NBA Success Dashboard: Best-2-Year RAPM ({holdout_year} Holdout)</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
    :root {{
      --bg0:#f5f1e9; --bg1:#e7ddcc; --ink:#10262a; --muted:#5c6e72;
      --card:rgba(255,253,247,0.80); --stroke:rgba(16,38,42,0.18);
      --ok:#0b7a75; --warn:#c27d2a; --bad:#9f4a3f;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); font-family:"Space Grotesk",sans-serif;
      background:radial-gradient(1300px 500px at 4% -12%, rgba(11,122,117,.12), transparent 60%),
                 radial-gradient(980px 420px at 96% -12%, rgba(194,125,42,.15), transparent 58%),
                 linear-gradient(160deg,var(--bg0) 0%,var(--bg1) 100%);
      min-height:100vh;
    }}
    .wrap {{ max-width:1260px; margin:0 auto; padding:24px; }}
    .hero,.panel,.metric {{ border:1px solid var(--stroke); border-radius:16px; background:var(--card); }}
    .hero {{ padding:18px 20px; margin-bottom:12px; }}
    .hero-top {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-end; flex-wrap:wrap; }}
    .hero h1 {{ margin:0; font-family:"Fraunces",serif; font-size:clamp(1.4rem,2.4vw,2.05rem); }}
    .hero p {{ margin:8px 0 0; color:var(--muted); }}
    .year-switch label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }}
    .year-switch select {{
      width:160px; padding:8px 10px; border-radius:10px; border:1px solid var(--stroke); background:#fffdf8;
      color:var(--ink); font-family:inherit;
    }}
    .back {{
      display:inline-block; margin-bottom:12px; text-decoration:none; color:var(--ink);
      border:1px solid var(--stroke); border-radius:10px; padding:7px 11px; background:rgba(255,255,255,.75);
      font-size:.88rem;
    }}
    .metrics {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin-bottom:12px; }}
    .metric {{ padding:10px 11px; }}
    .metric .k {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.03em; }}
    .metric .v {{ margin-top:3px; font-weight:700; font-size:1.08rem; font-variant-numeric:tabular-nums; }}
    .controls {{ display:grid; grid-template-columns:1.2fr .85fr .85fr .85fr; gap:10px; margin-bottom:12px; }}
    .controls label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }}
    .controls input,.controls select {{
      width:100%; padding:8px 10px; border-radius:10px; border:1px solid var(--stroke); background:#fffdf8;
      color:var(--ink); font-family:inherit;
    }}
    .panel {{ padding:14px; }}
    table {{ width:100%; border-collapse:collapse; font-size:.89rem; }}
    th {{
      text-align:left; color:var(--muted); font-size:.73rem; text-transform:uppercase; letter-spacing:.03em;
      border-bottom:1px solid var(--stroke); padding:7px 6px;
    }}
    th.sortable {{
      cursor:pointer;
      user-select:none;
    }}
    th.sortable .th-lbl {{
      display:inline-flex;
      align-items:center;
      gap:5px;
      white-space:nowrap;
    }}
    th.sortable .sort-arrow {{
      font-size:.7rem;
      opacity:.45;
      line-height:1;
    }}
    th.sortable.sort-asc .sort-arrow,
    th.sortable.sort-desc .sort-arrow {{
      opacity:.95;
    }}
    td {{ border-bottom:1px dashed rgba(16,38,42,.12); padding:7px 6px; vertical-align:middle; }}
    .mono {{ font-variant-numeric:tabular-nums; }}
    tr.row-good td {{ background:rgba(11,122,117,.09); }}
    tr.row-mid td {{ background:rgba(194,125,42,.10); }}
    tr.row-bad td {{ background:rgba(159,74,63,.10); }}
    .player-link {{
      color:var(--ink);
      font-weight:600;
      text-decoration:underline;
      text-underline-offset:2px;
      cursor:pointer;
    }}
    .abbr {{
      border:0;
      border-bottom:1px dotted rgba(16,38,42,.45);
      background:transparent;
      color:inherit;
      cursor:pointer;
      font:inherit;
      padding:0;
      line-height:inherit;
    }}
    .abbr.abbr-stat {{
      color:#1f4f5a;
      border-bottom-color:rgba(31,79,90,.45);
      background:rgba(31,79,90,.08);
      border-radius:4px;
      padding:0 3px;
    }}
    .abbr.abbr-measurement {{
      color:#0b7a75;
      border-bottom-color:rgba(11,122,117,.45);
      background:rgba(11,122,117,.10);
      border-radius:4px;
      padding:0 3px;
    }}
    .abbr.abbr-improvement {{
      color:#c27d2a;
      border-bottom-color:rgba(194,125,42,.45);
      background:rgba(194,125,42,.10);
      border-radius:4px;
      padding:0 3px;
    }}
    .abbr-tip {{
      position:fixed;
      z-index:1200;
      display:none;
      max-width:300px;
      padding:8px 10px;
      border-radius:10px;
      border:1px solid rgba(16,38,42,.25);
      background:#fffdf8;
      color:var(--ink);
      box-shadow:0 10px 24px rgba(16,38,42,.18);
      font-size:.82rem;
      line-height:1.35;
    }}
    .note {{ margin-top:8px; color:var(--muted); font-size:.8rem; }}
    .subhead {{ margin:18px 0 8px; font-family:"Fraunces",serif; font-size:1.2rem; }}
    .detail-head {{ display:flex; justify-content:space-between; align-items:flex-end; gap:10px; margin-bottom:8px; }}
    .detail-title {{ font-family:"Fraunces",serif; font-size:1.08rem; }}
    .detail-meta {{ color:var(--muted); font-size:.82rem; }}
    .detail-tabs {{ display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap; }}
    .tab-btn {{
      border:1px solid var(--stroke); border-radius:10px; padding:6px 10px;
      background:#fffdf8; color:var(--ink); font:inherit; font-size:.84rem; cursor:pointer;
    }}
    .tab-btn:disabled {{
      opacity:.45;
      cursor:not-allowed;
    }}
    .tab-btn.active {{ background:rgba(11,122,117,.12); border-color:rgba(11,122,117,.4); font-weight:700; }}
    .tab-btn.tab-college {{ border-color:rgba(194,125,42,.45); background:rgba(194,125,42,.08); }}
    .tab-btn.tab-college.active {{ background:rgba(194,125,42,.20); border-color:rgba(194,125,42,.75); }}
    .tab-btn.tab-nba {{ border-color:rgba(11,122,117,.45); background:rgba(11,122,117,.08); }}
    .tab-btn.tab-nba.active {{ background:rgba(11,122,117,.20); border-color:rgba(11,122,117,.75); }}
    .detail-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px 10px; }}
    .kv {{ border:1px dashed rgba(16,38,42,.18); border-radius:10px; padding:8px; background:rgba(255,255,255,.55); }}
    .kv .k {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.03em; }}
    .kv .v {{ margin-top:3px; font-weight:700; font-variant-numeric:tabular-nums; }}
    .kv .v.v-high {{
      color:#0b7a75;
      background:rgba(11,122,117,.12);
      border-radius:6px;
      padding:1px 6px;
      display:inline-block;
    }}
    .kv .v.v-low {{
      color:#9f4a3f;
      background:rgba(159,74,63,.12);
      border-radius:6px;
      padding:1px 6px;
      display:inline-block;
    }}
    .detail-season-wrap {{
      border:1px dashed rgba(16,38,42,.18);
      border-radius:12px;
      background:rgba(255,255,255,.55);
      overflow:auto;
    }}
    .detail-season-table {{
      width:100%;
      min-width:780px;
      border-collapse:collapse;
      font-size:.84rem;
    }}
    .detail-season-table th {{
      position:sticky;
      top:0;
      background:rgba(255,253,248,.96);
      z-index:2;
    }}
    .detail-season-table td.v-high {{
      color:#0b7a75;
      background:rgba(11,122,117,.12);
      font-weight:700;
    }}
    .detail-season-table td.v-low {{
      color:#9f4a3f;
      background:rgba(159,74,63,.12);
      font-weight:700;
    }}
    .detail-season-table tr.career-row td {{
      border-top:2px solid rgba(16,38,42,.22);
      background:rgba(248,243,234,.88);
      font-weight:700;
    }}
    .modal-overlay {{
      position:fixed; inset:0; display:none; align-items:center; justify-content:center;
      background:rgba(16,38,42,.45); z-index:60; padding:16px;
    }}
    .modal-card {{
      width:min(980px, 96vw); max-height:90vh; overflow:auto;
      border:1px solid var(--stroke); border-radius:16px; background:var(--card);
      box-shadow:0 18px 48px rgba(16,38,42,.25); padding:14px;
    }}
    .detail-close {{
      border:1px solid var(--stroke); border-radius:10px; padding:6px 10px;
      background:#fffdf8; color:var(--ink); font:inherit; font-size:.84rem; cursor:pointer;
    }}
    @media (max-width:1020px) {{
      .metrics {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
      .controls {{ grid-template-columns:1fr; }}
      .detail-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <a class="back" href="index.html">Back to Index</a>
    <section class="hero">
      <div class="hero-top">
        <h1>NBA Success Dashboard: Best-2-Year RAPM ({holdout_year} Holdout)</h1>
        <div class="year-switch">
          <label for="holdoutYearSel">Holdout Year</label>
          <select id="holdoutYearSel"></select>
        </div>
      </div>
      <p>Compares actual vs predicted NBA impact using RAPM values and rank error diagnostics. Model is trained on pooled entrant cohorts from 2018-2023.</p>
    </section>
    <section class="metrics" id="metrics"></section>
    <section class="panel">
      <div class="controls">
        <div><label for="q">Search player/team</label><input id="q" placeholder="e.g. Banchero, Duke" /></div>
        <div><label for="band">Rank outcome band</label><select id="band"><option value="all">All</option><option value="good">Hit (abs err <= 10)</option><option value="mid">Close (11-20)</option><option value="bad">Miss (>20)</option></select></div>
        <div><label for="maxpick">Max real pick shown</label><input id="maxpick" type="number" min="1" max="60" value="60" /></div>
        <div><label for="rowsN">Rows shown</label><input id="rowsN" type="number" min="10" max="200" value="60" /></div>
      </div>
      <table>
        <thead>
          <tr>
            <th class="sortable" data-sort-key="player"><span class="th-lbl">Player <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="college"><span class="th-lbl">College <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="pick"><span class="th-lbl">Pick <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="nba_team"><span class="th-lbl">NBA Team <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="actual_rapm"><span class="th-lbl">Actual RAPM <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="pred_rapm"><span class="th-lbl">Pred RAPM <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="actual_rank"><span class="th-lbl">Actual Rank <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="pred_rank"><span class="th-lbl">Pred Rank <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="outcome"><span class="th-lbl">Outcome <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="drivers_pos"><span class="th-lbl">Drivers (+) <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="drivers_neg"><span class="th-lbl">Drivers (-) <span class="sort-arrow">↕</span></span></th>
            <th class="sortable" data-sort-key="miss_explain"><span class="th-lbl">Miss Explanation <span class="sort-arrow">↕</span></span></th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="note">Rank-focused evaluation. Success rule: abs rank error <= 10. Row colors: green <=10, amber 11-20, red >20.</div>
    </section>

  </div>
  <div id="abbrTip" class="abbr-tip"></div>
  <div class="modal-overlay" id="detailModal">
    <section class="modal-card">
      <div class="detail-head">
        <div>
          <div class="detail-title" id="detailTitle">Player</div>
          <div class="detail-meta" id="detailMeta"></div>
        </div>
        <button class="detail-close" id="detailClose">Close</button>
      </div>
      <div class="detail-tabs" id="detailTabs"></div>
      <div class="detail-grid" id="detailSeason"></div>
      <div class="detail-grid" id="detailMeas" style="display:none;"></div>
    </section>
  </div>
  <script>
  const HOLD = {holdout_json};
  const BOARD = {board_json};
  const MET = {metrics_json};
  const YEAR_OPTIONS = {year_options_json};
  const DASH_VER = "20260309b";
  const fmt = (v,d=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "n/a";
  const esc = (s) => String(s ?? "");
  const escHtml = (s) => String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
  const escAttr = escHtml;
  const q = document.getElementById("q");
  const holdoutYearSel = document.getElementById("holdoutYearSel");
  const band = document.getElementById("band");
  const maxpick = document.getElementById("maxpick");
  const rowsN = document.getElementById("rowsN");
  const rows = document.getElementById("rows");
  const mainTableHead = document.querySelector(".panel table thead");
  const detailModal = document.getElementById("detailModal");
  const detailClose = document.getElementById("detailClose");
  const detailTitle = document.getElementById("detailTitle");
  const detailMeta = document.getElementById("detailMeta");
  const detailCard = document.querySelector(".modal-card");
  const detailTabs = document.getElementById("detailTabs");
  const detailSeason = document.getElementById("detailSeason");
  const detailMeas = document.getElementById("detailMeas");
  const abbrTip = document.getElementById("abbrTip");
  let visibleHoldRows = [];
  let currentDetailRec = null;
  let currentDetailTabId = "meas";
  let sortState = {{ key: "actual_rank", dir: "asc" }};

  const DRIVER_TIPS = {{
    "MIN": "Total minutes played.",
    "PTS/40": "Points scored per 40 minutes.",
    "AST/40": "Assists per 40 minutes.",
    "REB/40": "Total rebounds per 40 minutes.",
    "STL/40": "Steals per 40 minutes.",
    "BLK/40": "Blocks per 40 minutes.",
    "STL+BLK/40": "Stocks per 40 minutes (steals + blocks).",
    "USG": "Usage rate: share of team possessions used by the player.",
    "TS": "True shooting percentage: overall scoring efficiency including 2s, 3s, and free throws.",
    "3P%": "Three-point field-goal percentage.",
    "3PA/FGA": "Share of shot attempts that are three-pointers.",
    "AST/TOV": "Assist-to-turnover ratio.",
    "TOV/40": "Turnovers per 40 minutes.",
    "NET": "Net rating: team points per 100 possessions with this profile context.",
    "HT": "Height in inches.",
    "WING": "Wingspan in inches.",
    "REACH": "Standing reach in inches.",
  }};
  const DRIVER_TYPE = {{
    "MIN": "stat",
    "PTS/40": "stat",
    "AST/40": "stat",
    "REB/40": "stat",
    "STL/40": "stat",
    "BLK/40": "stat",
    "STL+BLK/40": "stat",
    "USG": "stat",
    "TS": "stat",
    "3P%": "stat",
    "3PA/FGA": "stat",
    "AST/TOV": "stat",
    "TOV/40": "stat",
    "NET": "stat",
    "HT": "measurement",
    "WING": "measurement",
    "REACH": "measurement",
  }};
  const SORT_DEFS = {{
    player: {{ type: "text", get: (d) => String(d?.name || "") }},
    college: {{ type: "text", get: (d) => String(d?.team || "") }},
    pick: {{ type: "num", get: (d) => num(d?.pick_number) }},
    nba_team: {{ type: "text", get: (d) => String(expandTeam(bestNbaTeamLabel(d)) || "") }},
    actual_rapm: {{ type: "num", get: (d) => num(d?.nba_rapm_current) }},
    pred_rapm: {{ type: "num", get: (d) => num(d?.pred_rapm_current_est) }},
    actual_rank: {{ type: "num", get: (d) => num(d?.actual_rank) }},
    pred_rank: {{ type: "num", get: (d) => num(d?.pred_rank) }},
    outcome: {{ type: "text", get: (d) => String(outcomeText(rankErr(d))) }},
    drivers_pos: {{ type: "text", get: (d) => String(d?.driver_positive_short || "") }},
    drivers_neg: {{ type: "text", get: (d) => String(d?.driver_negative_short || "") }},
    miss_explain: {{ type: "text", get: (d) => String(d?.miss_explain_short || "") }},
  }};

  function compareSortValues(a, b, type) {{
    if (type === "num") {{
      const av = Number(a);
      const bv = Number(b);
      const af = Number.isFinite(av);
      const bf = Number.isFinite(bv);
      if (af && bf) {{
        if (av < bv) return -1;
        if (av > bv) return 1;
        return 0;
      }}
      if (af) return -1;
      if (bf) return 1;
      return 0;
    }}
    return String(a || "").localeCompare(String(b || ""), undefined, {{ sensitivity: "base" }});
  }}

  function activeSortDef() {{
    const key = sortState?.key;
    if (key && SORT_DEFS[key]) return [key, SORT_DEFS[key]];
    return ["actual_rank", SORT_DEFS.actual_rank];
  }}

  function updateSortHeaderState() {{
    if (!mainTableHead) return;
    const [activeKey] = activeSortDef();
    const dir = sortState?.dir === "desc" ? "desc" : "asc";
    mainTableHead.querySelectorAll("th.sortable").forEach((th) => {{
      const key = th.dataset.sortKey;
      const isActive = key === activeKey;
      th.classList.toggle("sort-asc", isActive && dir === "asc");
      th.classList.toggle("sort-desc", isActive && dir === "desc");
      const arrow = th.querySelector(".sort-arrow");
      if (arrow) arrow.textContent = isActive ? (dir === "asc" ? "▲" : "▼") : "↕";
    }});
  }}
  const NBA_TEAM_FULL = {{
    ATL: "Atlanta Hawks",
    BOS: "Boston Celtics",
    BKN: "Brooklyn Nets",
    CHA: "Charlotte Hornets",
    CHO: "Charlotte Hornets",
    CHI: "Chicago Bulls",
    CLE: "Cleveland Cavaliers",
    DAL: "Dallas Mavericks",
    DEN: "Denver Nuggets",
    DET: "Detroit Pistons",
    GSW: "Golden State Warriors",
    HOU: "Houston Rockets",
    IND: "Indiana Pacers",
    LAC: "Los Angeles Clippers",
    LAL: "Los Angeles Lakers",
    MEM: "Memphis Grizzlies",
    MIA: "Miami Heat",
    MIL: "Milwaukee Bucks",
    MIN: "Minnesota Timberwolves",
    NOP: "New Orleans Pelicans",
    NOH: "New Orleans Hornets",
    NOK: "New Orleans/Oklahoma City Hornets",
    NYK: "New York Knicks",
    OKC: "Oklahoma City Thunder",
    ORL: "Orlando Magic",
    PHI: "Philadelphia 76ers",
    PHO: "Phoenix Suns",
    POR: "Portland Trail Blazers",
    SAC: "Sacramento Kings",
    SAS: "San Antonio Spurs",
    TOR: "Toronto Raptors",
    UTA: "Utah Jazz",
    WAS: "Washington Wizards",
    WSH: "Washington Wizards",
    SEA: "Seattle SuperSonics",
    NJN: "New Jersey Nets",
    VAN: "Vancouver Grizzlies",
  }};

  function withDriverTips(text) {{
    const raw = String(text ?? "");
    if (!raw) return "";
    let out = escHtml(raw);
    const keys = Object.keys(DRIVER_TIPS).sort((a, b) => b.length - a.length);
    keys.forEach((k) => {{
      const tip = escAttr(DRIVER_TIPS[k]);
      const kk = escHtml(k);
      const t = escAttr(DRIVER_TYPE[k] || "stat");
      out = out.split(kk).join(`<button type="button" class="abbr abbr-${{t}}" data-tip="${{tip}}">${{kk}}</button>`);
    }});
    return out;
  }}

  function hideAbbrTip() {{
    abbrTip.style.display = "none";
  }}

  function showAbbrTip(btn) {{
    const tip = String(btn?.dataset?.tip || "").trim();
    if (!tip) {{
      hideAbbrTip();
      return;
    }}
    abbrTip.textContent = tip;
    abbrTip.style.display = "block";
    const r = btn.getBoundingClientRect();
    const pad = 10;
    let left = r.left;
    let top = r.bottom + 8;
    const tipRect = abbrTip.getBoundingClientRect();
    if (left + tipRect.width + pad > window.innerWidth) {{
      left = Math.max(pad, window.innerWidth - tipRect.width - pad);
    }}
    if (top + tipRect.height + pad > window.innerHeight) {{
      top = Math.max(pad, r.top - tipRect.height - 8);
    }}
    abbrTip.style.left = `${{Math.round(left)}}px`;
    abbrTip.style.top = `${{Math.round(top)}}px`;
  }}

  function num(v) {{
    const x = Number(v);
    return Number.isFinite(x) ? x : NaN;
  }}

  function fmtAuto(v, d=2) {{
    const x = Number(v);
    if (Number.isFinite(x)) return x.toFixed(d);
    if (Number.isNaN(x)) return "n/a";
    if (v === null || v === undefined || v === "") return "n/a";
    return String(v);
  }}

  function fmtFeetInches(v, d=1) {{
    const raw = Number(v);
    if (!Number.isFinite(raw)) return "n/a";
    const sign = raw < 0 ? "-" : "";
    const abs = Math.abs(raw);
    let feet = Math.floor(abs / 12);
    let inches = abs - (feet * 12);
    const dec = Number.isFinite(Number(d)) ? Math.max(0, Number(d)) : 1;
    inches = Number(inches.toFixed(dec));
    if (inches >= 12) {{
      feet += 1;
      inches = 0;
    }}
    const inchText = dec > 0
      ? inches.toFixed(dec).replace(/[.]0+$/, "").replace(/([.][0-9]*[1-9])0+$/, "$1")
      : String(Math.round(inches));
    return `${{sign}}${{feet}}'${{inchText}}"`;
  }}

  function isMissingText(v) {{
    if (v === null || v === undefined) return true;
    const s = String(v).trim();
    if (!s) return true;
    const low = s.toLowerCase();
    return low === "nan" || low === "na" || low === "n/a" || low === "null" || low === "none" || low === "undefined";
  }}

  function cleanText(v) {{
    return isMissingText(v) ? "" : String(v).trim();
  }}

  const COLLEGE_SEASON_FIELDS = [
    ["team", "College Team", null],
    ["position", "Position", null],
    ["games", "Games", 0],
    ["minutes", "Minutes", 0],
    ["points_per40", "PTS/40", 1],
    ["assists_per40", "AST/40", 1],
    ["rebounds_total_per40", "REB/40", 1],
    ["steals_per40", "STL/40", 1],
    ["blocks_per40", "BLK/40", 1],
    ["stocks_per40", "STL+BLK/40", 1],
    ["usage", "Usage", 1],
    ["true_shooting_pct", "TS", 3],
    ["three_point_pct", "3P%", 1],
    ["three_point_attempt_rate", "3PA/FGA", 3],
    ["assist_to_turnover", "AST/TOV", 2],
    ["turnovers_per40", "TOV/40", 1],
    ["net_rating", "Net Rating", 1],
  ];

  const NBA_SEASON_FIELDS = [
    ["team", "NBA Team", null],
    ["position", "Position", null],
    ["games", "Games", 0],
    ["minutes", "Minutes", 0],
    ["points_per75", "PTS/75", 1],
    ["assists_per100", "AST/100", 1],
    ["steals_per100", "STL/100", 1],
    ["blocks_per100", "BLK/100", 1],
    ["stocks_per100", "STL+BLK/100", 1],
    ["ts_pct", "TS", 3],
    ["three_point_pct", "3P%", 1],
    ["three_pa_per100", "3PA/100", 1],
    ["tov_per100", "TOV/100", 1],
  ];

  const MEAS_FIELDS = [
    ["age_draft_years", "Age (draft, yrs)", 0],
    ["age_current_years_est", "Age (current, yrs)", 0],
    ["measurement_height_in", "Height (ft-in)", 1],
    ["measurement_weight_lb", "Weight (lb)", 1],
    ["measurement_wingspan_in", "Wingspan (ft-in)", 2],
    ["measurement_standing_reach_in", "Standing Reach (ft-in)", 2],
    ["measurement_wingspan_minus_height", "Wingspan - Height (in)", 2],
    ["measurement_reach_minus_height", "Reach - Height (in)", 2],
    ["combine_hand_length_in", "Hand Length (in)", 2],
    ["combine_hand_width_in", "Hand Width (in)", 2],
    ["combine_body_fat_pct", "Body Fat %", 2],
  ];

  const INCH_MEAS_KEYS = new Set([
    "measurement_height_in",
    "measurement_wingspan_in",
    "measurement_standing_reach_in",
  ]);

  function toSeasonRange(endYearLike) {{
    const y = Number(endYearLike);
    if (!Number.isFinite(y)) return null;
    const endYear = Math.floor(y);
    if (endYear < 1901) return null;
    const startYear = endYear - 1;
    const endShort = String(endYear).slice(-2).padStart(2, "0");
    return `${{startYear}}/${{endShort}}`;
  }}

  function getFieldValue(rec, key) {{
    if (!rec) return null;
    if (key === "age_draft_years") {{
      const candidates = [rec.age_draft_years, rec.ext_draft_day_age, rec.measurement_age, rec.age];
      for (const v of candidates) {{
        const x = Number(v);
        if (Number.isFinite(x)) return x;
      }}
      return null;
    }}
    if (key === "age_current_years_est") {{
      const candidates = [rec.age_current_years_est];
      for (const v of candidates) {{
        const x = Number(v);
        if (Number.isFinite(x)) return x;
      }}
      return null;
    }}
    if (key === "birth_year_est") {{
      const x = Number(rec.birth_year_est);
      return Number.isFinite(x) ? x : null;
    }}
    if (key === "college_start_season") {{
      return toSeasonRange(rec.college_start_year);
    }}
    if (key === "nba_start_season") {{
      return toSeasonRange(rec.nba_start_year);
    }}
    return rec[key];
  }}

  function renderKv(container, rec, fields) {{
    container.innerHTML = fields.map(([k, label, d]) => {{
      const v = getFieldValue(rec, k);
      const tone = resolvePopupTone("meas", k, v, d);
      let vv;
      if (d === null) vv = escHtml(v ?? "n/a");
      else if (INCH_MEAS_KEYS.has(k)) vv = fmtFeetInches(v, d);
      else vv = fmtAuto(v, d);
      const cls = tone.cls ? ` ${{tone.cls}}` : "";
      const tip = tone.tip ? ` title="${{escAttr(tone.tip)}}"` : "";
      return `<div class="kv"><div class="k">${{label}}</div><div class="v${{cls}}"${{tip}}>${{vv}}</div></div>`;
    }}).join("");
  }}

  function normalizeSeasonRows(rows, kind) {{
    if (!Array.isArray(rows)) return [];
    return rows
      .filter(r => r && typeof r === "object")
      .map(r => ({{
        ...r,
        kind: cleanText(r.kind) || kind,
        team: cleanText(r.team),
        position: cleanText(r.position),
        season_label: cleanText(r.season_label),
      }}))
      .filter(r => Number.isFinite(Number(r.season_end_year)) || String(r.season_label || "").trim() !== "");
  }}

  function mergeSeasonRows(rec) {{
    const college = normalizeSeasonRows(rec?.college_seasons, "college");
    const nba = normalizeSeasonRows(rec?.nba_seasons, "nba");
    const all = [...college, ...nba];
    all.sort((a, b) => {{
      const ay = Number(a.season_end_year);
      const by = Number(b.season_end_year);
      if (Number.isFinite(ay) && Number.isFinite(by) && ay !== by) return ay - by;
      if (a.kind !== b.kind) return a.kind === "college" ? -1 : 1;
      return String(a.season_label || "").localeCompare(String(b.season_label || ""));
    }});
    return all;
  }}

  function uniqNonEmpty(vals) {{
    return [...new Set((vals || []).map((v) => cleanText(v)).filter(Boolean))];
  }}

  function sumSeasonValues(rec, rows, key) {{
    let s = 0;
    let n = 0;
    (rows || []).forEach((row) => {{
      const x = finiteNum(getSeasonValue(rec, row, key));
      if (!Number.isFinite(x)) return;
      s += x;
      n += 1;
    }});
    return n > 0 ? s : null;
  }}

  function weightedSeasonAvg(rec, rows, key, weightKey="minutes") {{
    let sw = 0;
    let sx = 0;
    let n = 0;
    let s = 0;
    (rows || []).forEach((row) => {{
      const x = finiteNum(getSeasonValue(rec, row, key));
      if (!Number.isFinite(x)) return;
      const w = finiteNum(getSeasonValue(rec, row, weightKey));
      if (Number.isFinite(w) && w > 0) {{
        sw += w;
        sx += w * x;
      }}
      n += 1;
      s += x;
    }});
    if (sw > 0) return sx / sw;
    if (n > 0) return s / n;
    return null;
  }}

  function buildAggregateSeasonRow(rec, kind, rows) {{
    const group = (rows || []).filter((r) => r && r.kind === kind);
    if (!group.length) return null;
    const fields = kind === "nba" ? NBA_SEASON_FIELDS : COLLEGE_SEASON_FIELDS;
    const out = {{
      kind,
      season_label: kind === "nba" ? "All NBA" : "All College",
      team: "",
      position: "",
    }};

    const teams = uniqNonEmpty(group.map((r) => getSeasonValue(rec, r, "team")));
    const positions = uniqNonEmpty(group.map((r) => getSeasonValue(rec, r, "position")));
    out.team = teams.join(", ");
    out.position = positions.join(" / ");

    fields.forEach(([k, _label, d]) => {{
      if (d === null || k === "team" || k === "position") return;
      if (k === "games" || k === "minutes") {{
        out[k] = sumSeasonValues(rec, group, k);
      }} else {{
        out[k] = weightedSeasonAvg(rec, group, k, "minutes");
      }}
    }});

    if (kind === "college") {{
      const astTot = sumSeasonValues(
        rec,
        group.map((r) => {{
          const m = finiteNum(getSeasonValue(rec, r, "minutes"));
          const a40 = finiteNum(getSeasonValue(rec, r, "assists_per40"));
          return {{ ...r, __tmp: Number.isFinite(m) && Number.isFinite(a40) ? (a40 * m) / 40.0 : null }};
        }}),
        "__tmp",
      );
      const tovTot = sumSeasonValues(
        rec,
        group.map((r) => {{
          const m = finiteNum(getSeasonValue(rec, r, "minutes"));
          const t40 = finiteNum(getSeasonValue(rec, r, "turnovers_per40"));
          return {{ ...r, __tmp: Number.isFinite(m) && Number.isFinite(t40) ? (t40 * m) / 40.0 : null }};
        }}),
        "__tmp",
      );
      if (Number.isFinite(astTot) && Number.isFinite(tovTot) && tovTot > 1e-9) {{
        out.assist_to_turnover = astTot / tovTot;
      }}
    }}
    return out;
  }}

  function teamTabAbbr(teamLike) {{
    const raw = cleanText(teamLike);
    if (!raw) return "";
    const stripped = raw.replace(/\\([^)]*\\)/g, "").trim();
    if (!stripped) return "";
    if (stripped.includes("/")) {{
      return stripped.split("/").map((p) => teamTabAbbr(p)).filter(Boolean).join("/");
    }}
    const up = stripped.toUpperCase();
    if (NBA_TEAM_FULL[up]) return up;

    const compact = stripped
      .replace(/university/ig, "")
      .replace(/college/ig, "")
      .replace(/saint/ig, "st")
      .replace(/state/ig, "st")
      .replace(/[^a-zA-Z0-9\\s]/g, " ")
      .trim();
    if (!compact) return up.slice(0, 4);
    const words = compact.split(/\\s+/).filter(Boolean);
    if (!words.length) return up.slice(0, 4);
    if (words.length === 1) {{
      const w = words[0];
      return (w.length <= 4 ? w : w.slice(0, 4)).toUpperCase();
    }}
    const stop = new Set(["OF", "THE", "AT", "AND", "A", "AN"]);
    const initials = words
      .filter((w) => !stop.has(w.toUpperCase()))
      .map((w) => w[0].toUpperCase())
      .join("");
    if (initials.length >= 2 && initials.length <= 4) return initials;
    return words.slice(0, 2).map((w) => w.slice(0, 2).toUpperCase()).join("");
  }}

  function seasonTabLabel(row, rec) {{
    const seasonLabel = cleanText(row?.season_label) || toSeasonRange(row?.season_end_year) || "Season";
    let abbr = teamTabAbbr(row?.team);
    if (!abbr && row?.kind === "nba") {{
      abbr = teamTabAbbr(bestNbaTeamLabel(rec));
    }}
    if (abbr && seasonLabel) return `${{abbr}} ${{seasonLabel}}`;
    if (abbr) return abbr;
    return seasonLabel;
  }}

  function seasonTabClass(row) {{
    return row.kind === "nba" ? "tab-nba" : "tab-college";
  }}

  function expandTeam(v) {{
    const raw = cleanText(v);
    if (!raw) return raw;
    const key = raw.toUpperCase();
    return NBA_TEAM_FULL[key] || raw;
  }}

  function latestNbaTeam(rec) {{
    if (!rec) return "";
    const nbaRows = normalizeSeasonRows(rec?.nba_seasons, "nba");
    if (nbaRows.length) {{
      const sorted = [...nbaRows].sort((a, b) => {{
        const ay = Number(a?.season_end_year);
        const by = Number(b?.season_end_year);
        if (Number.isFinite(ay) && Number.isFinite(by) && ay !== by) return ay - by;
        return String(a?.season_label || "").localeCompare(String(b?.season_label || ""));
      }});
      for (let i = sorted.length - 1; i >= 0; i -= 1) {{
        const t = cleanText(sorted[i]?.team);
        if (t) return t;
      }}
    }}
    const hist = cleanText(rec?.nba_team_history);
    if (hist) {{
      const first = hist.split(",").map((x) => String(x || "").trim()).find(Boolean) || "";
      if (first) {{
        const abbr = first.match(/^([A-Za-z]{2,4})\b/);
        if (abbr) return abbr[1].toUpperCase();
        return first.replace(/\([^)]*\)/g, "").trim();
      }}
    }}
    return "";
  }}

  function bestNbaTeamLabel(rec) {{
    if (!rec) return "";
    const primary = cleanText(rec.nba_team_display) || cleanText(rec.nba_team_abbr);
    if (primary) return primary;
    return String(latestNbaTeam(rec) || "").trim();
  }}

  function expandPosition(v) {{
    const raw = String(v || "").trim();
    if (!raw) return raw;
    const map = {{
      "PG": "Point Guard",
      "SG": "Shooting Guard",
      "SF": "Small Forward",
      "PF": "Power Forward",
      "C": "Center",
      "G": "Guard",
      "F": "Forward",
      "GF": "Guard/Forward",
      "FG": "Forward/Guard",
      "FC": "Forward/Center",
      "CF": "Center/Forward",
      "GC": "Guard/Center",
      "CG": "Center/Guard",
    }};
    const expanded = raw
      .split(/[\\/,&\\-]+/)
      .map((tok) => {{
        const t = String(tok || "").trim().toUpperCase().replace(/[^A-Z]/g, "");
        return map[t] || String(tok || "").trim();
      }})
      .filter(Boolean);
    return [...new Set(expanded)].join(" / ");
  }}

  function getSeasonFallback(rec, row, key) {{
    if (!rec) return null;
    if (key === "team") {{
      if (row.kind === "nba") {{
        return bestNbaTeamLabel(rec) || null;
      }}
      return rec.team || null;
    }}
    if (key === "position") {{
      return row.kind === "nba"
        ? (rec.nba_position || rec.position_group || rec.position || null)
        : (rec.position || rec.position_group || null);
    }}
    const nbaMap = {{
      games: "nba_games",
      minutes: "nba_minutes",
      points_per75: "nba_points_per75",
      assists_per100: "nba_assists_per100",
      steals_per100: "nba_steals_per100",
      blocks_per100: "nba_blocks_per100",
      stocks_per100: "nba_stocks_per100",
      ts_pct: "nba_ts_pct",
      three_point_pct: "nba_three_point_pct",
      three_pa_per100: "nba_three_pa_per100",
      tov_per100: "nba_tov_per100",
    }};
    return row.kind === "nba" ? rec[nbaMap[key] || ""] : rec[key];
  }}

  function getSeasonValue(rec, row, key) {{
    const rowVal = row ? row[key] : null;
    const raw = !isMissingText(rowVal)
      ? rowVal
      : getSeasonFallback(rec, row || {{}}, key);
    if (key === "position") return expandPosition(raw);
    if (key === "team") return expandTeam(raw);
    return raw;
  }}

  function finiteNum(v) {{
    const x = Number(v);
    return Number.isFinite(x) ? x : NaN;
  }}

  function buildPopupBaselines() {{
    const buckets = {{ meas: {{}}, college: {{}}, nba: {{}} }};
    const add = (bucket, key, val) => {{
      const x = finiteNum(val);
      if (!Number.isFinite(x)) return;
      if (!buckets[bucket][key]) buckets[bucket][key] = [];
      buckets[bucket][key].push(x);
    }};
    const pool = [...HOLD, ...BOARD];
    pool.forEach((rec) => {{
      MEAS_FIELDS.forEach(([k, _label, d]) => {{
        if (d === null) return;
        add("meas", k, getFieldValue(rec, k));
      }});
      const seasonRows = mergeSeasonRows(rec);
      seasonRows.forEach((row) => {{
        const bucket = row.kind === "nba" ? "nba" : "college";
        const fields = row.kind === "nba" ? NBA_SEASON_FIELDS : COLLEGE_SEASON_FIELDS;
        fields.forEach(([k, _label, d]) => {{
          if (d === null) return;
          add(bucket, k, getSeasonValue(rec, row, k));
        }});
      }});
    }});

    const out = {{ meas: {{}}, college: {{}}, nba: {{}} }};
    ["meas", "college", "nba"].forEach((bucket) => {{
      Object.entries(buckets[bucket]).forEach(([k, vals]) => {{
        if (!Array.isArray(vals) || vals.length < 20) return;
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const varPop = vals.reduce((a, b) => a + ((b - mean) ** 2), 0) / vals.length;
        const sd = Math.sqrt(varPop);
        if (!Number.isFinite(sd) || sd <= 1e-9) return;
        out[bucket][k] = {{ mean, sd, n: vals.length }};
      }});
    }});
    return out;
  }}

  const POPUP_BASELINES = buildPopupBaselines();

  function resolvePopupTone(bucket, key, rawVal, d) {{
    if (d === null) return {{ cls: "", tip: "" }};
    const x = finiteNum(rawVal);
    if (!Number.isFinite(x)) return {{ cls: "", tip: "" }};
    const stats = POPUP_BASELINES?.[bucket]?.[key];
    if (!stats) return {{ cls: "", tip: "" }};
    const z = (x - stats.mean) / stats.sd;
    const meanTxt = Number(stats.mean).toFixed(d);
    const sdTxt = Number(stats.sd).toFixed(Math.max(1, d));
    const tip = `Avg ${{meanTxt}} (sd ${{sdTxt}}, n=${{stats.n}}), z=${{z.toFixed(2)}}`;
    if (z >= 1.0) return {{ cls: "v-high", tip }};
    if (z <= -1.0) return {{ cls: "v-low", tip }};
    return {{ cls: "", tip }};
  }}

  function seasonRowsForKind(rec, kind) {{
    const rows = normalizeSeasonRows(
      kind === "nba" ? rec?.nba_seasons : rec?.college_seasons,
      kind,
    );
    rows.sort((a, b) => {{
      const ay = Number(a?.season_end_year);
      const by = Number(b?.season_end_year);
      if (Number.isFinite(ay) && Number.isFinite(by) && ay !== by) return ay - by;
      return String(a?.season_label || "").localeCompare(String(b?.season_label || ""));
    }});
    return rows;
  }}

  function renderSeasonTable(container, rec, kind) {{
    const rows = seasonRowsForKind(rec, kind);
    const fields = kind === "nba" ? NBA_SEASON_FIELDS : COLLEGE_SEASON_FIELDS;
    const bucket = kind === "nba" ? "nba" : "college";
    const career = buildAggregateSeasonRow(rec, kind, rows);
    if (!rows.length) {{
      container.innerHTML = `<div class="note">No ${{kind.toUpperCase()}} season data.</div>`;
      return;
    }}
    const head = ["Season", ...fields.map(([_k, label]) => escHtml(label))]
      .map((h) => `<th>${{h}}</th>`)
      .join("");
    const body = rows.map((row) => {{
      const seasonLabel = cleanText(row?.season_label) || toSeasonRange(row?.season_end_year) || "n/a";
      const cells = fields.map(([k, _label, d]) => {{
        const v = getSeasonValue(rec, row, k);
        if (d === null) {{
          return `<td>${{escHtml(v ?? "n/a")}}</td>`;
        }}
        const tone = resolvePopupTone(bucket, k, v, d);
        const cls = tone.cls ? ` ${{tone.cls}}` : "";
        const tip = tone.tip ? ` title="${{escAttr(tone.tip)}}"` : "";
        return `<td class="mono${{cls}}"${{tip}}>${{fmtAuto(v, d)}}</td>`;
      }}).join("");
      return `<tr><td class="mono">${{escHtml(seasonLabel)}}</td>${{cells}}</tr>`;
    }}).join("");
    const careerCells = fields.map(([k, _label, d]) => {{
      const v = career ? getSeasonValue(rec, career, k) : null;
      if (d === null) {{
        return `<td>${{escHtml(v ?? "n/a")}}</td>`;
      }}
      const tone = resolvePopupTone(bucket, k, v, d);
      const cls = tone.cls ? ` ${{tone.cls}}` : "";
      const tip = tone.tip ? ` title="${{escAttr(tone.tip)}}"` : "";
      return `<td class="mono${{cls}}"${{tip}}>${{fmtAuto(v, d)}}</td>`;
    }}).join("");
    const careerRow = career
      ? `<tr class="career-row"><td class="mono">Career</td>${{careerCells}}</tr>`
      : "";
    const intro = kind === "nba"
      ? `<div class="note" style="margin:0 0 6px;">Draft: <strong>${{escHtml(draftPickLabel(rec))}}</strong></div>`
      : "";
    container.innerHTML = `${{intro}}<div class="detail-season-wrap">
      <table class="detail-season-table">
        <thead><tr>${{head}}</tr></thead>
        <tbody>${{body}}${{careerRow}}</tbody>
      </table>
    </div>`;
  }}

  function buildDetailTabs(rec) {{
    const hasCollege = seasonRowsForKind(rec, "college").length > 0;
    const hasNba = seasonRowsForKind(rec, "nba").length > 0;
    detailTabs.innerHTML = [
      `<button type="button" class="tab-btn tab-college" data-tab="college"${{hasCollege ? "" : " disabled"}}>College</button>`,
      `<button type="button" class="tab-btn tab-nba" data-tab="nba"${{hasNba ? "" : " disabled"}}>NBA</button>`,
      `<button type="button" class="tab-btn" data-tab="meas">Measurements</button>`,
    ].join("");
    if (hasCollege) return "college";
    if (hasNba) return "nba";
    return "meas";
  }}

  function setDetailTab(tab) {{
    currentDetailTabId = tab;
    const isMeas = tab === "meas";
    const isCollege = tab === "college";
    const isNba = tab === "nba";
    const isSeason = isCollege || isNba;
    detailSeason.style.display = isSeason ? "block" : "none";
    detailMeas.style.display = isMeas ? "grid" : "none";
    if (isSeason && currentDetailRec) {{
      renderSeasonTable(detailSeason, currentDetailRec, isNba ? "nba" : "college");
    }}
    detailTabs.querySelectorAll(".tab-btn").forEach((btn) => {{
      btn.classList.toggle("active", btn.dataset.tab === tab);
    }});
  }}

  function currentDetailTab() {{
    return currentDetailTabId || "meas";
  }}

  function draftPickLabel(rec) {{
    const pk = Number(rec?.pick_number);
    if (Number.isFinite(pk) && pk > 0) return `Pick #${{Math.round(pk)}}`;
    return "Undrafted";
  }}

  function lockDetailModalSize(activeTab) {{
    if (!detailCard || detailModal.style.display === "none") return;
    const tabOrder = Array.from(detailTabs.querySelectorAll(".tab-btn")).map(btn => btn.dataset.tab);
    if (!tabOrder.length) return;
    const desired = activeTab || currentDetailTab();
    const keepTab = tabOrder.includes(desired) ? desired : tabOrder[0];
    const prevVis = detailCard.style.visibility;
    detailCard.style.visibility = "hidden";
    detailCard.style.minHeight = "";
    let maxH = 0;
    tabOrder.forEach((t) => {{
      setDetailTab(t);
      const h = Math.ceil(detailCard.scrollHeight || 0);
      if (h > maxH) maxH = h;
    }});
    setDetailTab(keepTab);
    if (maxH > 0) detailCard.style.minHeight = `${{maxH}}px`;
    detailCard.style.visibility = prevVis;
  }}

  function showDetail(rec, sourceTag) {{
    if (!rec) return;
    currentDetailRec = rec;
    detailTitle.textContent = `${{rec.name || "Player"}}`;
    const pickTxt = Number.isFinite(Number(rec.pick_number)) ? `Pick #${{Number(rec.pick_number)}}` : "Pick n/a";
    detailMeta.textContent = `${{sourceTag}} • ${{pickTxt}}`;
    const firstTab = buildDetailTabs(rec);
    renderKv(detailMeas, rec, MEAS_FIELDS);
    setDetailTab(firstTab);
    detailCard.style.minHeight = "";
    detailModal.style.display = "flex";
    requestAnimationFrame(() => lockDetailModalSize(firstTab));
  }}

  function hideDetail() {{
    detailModal.style.display = "none";
    if (detailCard) detailCard.style.minHeight = "";
  }}

  document.getElementById("metrics").innerHTML = [
    ["Holdout Draft Year", MET.holdout_year],
    ["Holdout Players", MET.n_holdout],
    ["Matched RAPM", MET.n_rapm],
    ["RAPM MAE", MET.rapm_mae],
    ["RAPM RMSE", MET.rapm_rmse],
    ["RAPM Corr", MET.rapm_corr],
    ["RAPM Hits (<=0.75)", MET.rapm_hits_075],
    ["RAPM Hit Rate (<=0.75)", MET.rapm_hit_rate_075],
    ["RAPM Hits (<=1.0)", MET.rapm_hits],
    ["RAPM Hit Rate", MET.rapm_hit_rate],
    ["Rank MAE", MET.rank_mae],
    ["Rank Hits (<=5)", MET.rank_hits_5],
    ["Rank Hit Rate (<=5)", MET.rank_hit_rate_5],
    ["Rank Hits (<=10)", MET.rank_hits_10],
    ["Rank Hit Rate (<=10)", MET.rank_hit_rate_10],
  ].map(([k,v]) => `<div class="metric"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");

  function rapmErr(d) {{
    const a = num(d.nba_rapm_current);
    const p = num(d.pred_rapm_current_est);
    if (!Number.isFinite(a) || !Number.isFinite(p)) return NaN;
    return Math.abs(a - p);
  }}

  function rankErr(d) {{
    const eRaw = num(d.rank_error);
    if (Number.isFinite(eRaw)) return Math.abs(eRaw);
    const a = num(d.actual_rank);
    const p = num(d.pred_rank);
    if (!Number.isFinite(a) || !Number.isFinite(p)) return NaN;
    return Math.abs(p - a);
  }}

  function bandClass(err) {{
    if (!Number.isFinite(err)) return "na";
    if (err <= 10) return "good";
    if (err <= 20) return "mid";
    return "bad";
  }}

  function outcomeText(err) {{
    if (!Number.isFinite(err)) return "n/a";
    if (err <= 10) return "Hit";
    if (err <= 20) return "Close";
    return "Miss";
  }}

  function filtered() {{
    const qq = q.value.trim().toLowerCase();
    const bb = band.value;
    const mp = Math.max(1, Math.min(60, parseInt(maxpick.value || "60", 10)));
    return HOLD.filter(d => {{
      const pk = Number(d.pick_number);
      if (Number.isFinite(pk) && pk > mp) return false;
      const txt = `${{d.name || ""}} ${{d.team || ""}}`.toLowerCase();
      const okQ = !qq || txt.includes(qq);
      const err = rankErr(d);
      const cls = bandClass(err);
      const okB = bb === "all" || (Number.isFinite(err) && bb === cls);
      return okQ && okB;
    }});
  }}

  function render() {{
    const n = Math.max(10, Math.min(200, parseInt(rowsN.value || "60", 10)));
    const [sortKey, sortDef] = activeSortDef();
    const sortDir = sortState?.dir === "desc" ? -1 : 1;
    const data = filtered().slice().sort((a,b) => {{
      const primary = compareSortValues(sortDef.get(a), sortDef.get(b), sortDef.type);
      if (primary !== 0) return primary * sortDir;
      const ar = num(a.actual_rank);
      const br = num(b.actual_rank);
      if (Number.isFinite(ar) && Number.isFinite(br)) return ar - br;
      if (Number.isFinite(ar)) return -1;
      if (Number.isFinite(br)) return 1;
      const ap = num(a.pick_number);
      const bp = num(b.pick_number);
      if (Number.isFinite(ap) && Number.isFinite(bp)) return ap - bp;
      return String(a.name || "").localeCompare(String(b.name || ""));
    }});
    const shown = data.slice(0, n);
    visibleHoldRows = shown;
    updateSortHeaderState();

    rows.innerHTML = shown.map((d, i) => {{
      const rankAbsErr = rankErr(d);
      const cls = bandClass(rankAbsErr);
      const outcome = outcomeText(rankAbsErr);
      const missTxt = outcome === "Miss" ? withDriverTips(d.miss_explain_short || "") : "";
      return `<tr class="${{cls === "na" ? "" : "row-" + cls}}">
        <td><a href="#" class="player-link" data-src="hold" data-idx="${{i}}">${{esc(d.name)}}</a></td>
        <td>${{esc(d.team)}}</td>
        <td class="mono">${{Number.isFinite(Number(d.pick_number)) ? Number(d.pick_number) : "n/a"}}</td>
        <td class="mono">${{esc(expandTeam(bestNbaTeamLabel(d)))}}</td>
        <td class="mono">${{fmt(d.nba_rapm_current,3)}}</td>
        <td class="mono">${{fmt(d.pred_rapm_current_est,3)}}</td>
        <td class="mono">${{fmt(d.actual_rank,0)}}</td>
        <td class="mono">${{fmt(d.pred_rank,0)}}</td>
        <td>${{outcome}}</td>
        <td>${{withDriverTips(d.driver_positive_short || "")}}</td>
        <td>${{withDriverTips(d.driver_negative_short || "")}}</td>
        <td>${{missTxt}}</td>
      </tr>`;
    }}).join("");

  }}

  document.addEventListener("click", (ev) => {{
    const ab = ev.target.closest(".abbr");
    if (ab) {{
      ev.preventDefault();
      ev.stopPropagation();
      showAbbrTip(ab);
      return;
    }}
    hideAbbrTip();

    const a = ev.target.closest("a.player-link");
    if (!a) return;
    ev.preventDefault();
    const src = a.dataset.src;
    const idx = parseInt(a.dataset.idx || "-1", 10);
    if (!Number.isFinite(idx) || idx < 0) return;
    if (src === "hold" && idx < visibleHoldRows.length) {{
      showDetail(visibleHoldRows[idx], "Holdout");
    }}
  }});
  detailClose.addEventListener("click", hideDetail);
  detailModal.addEventListener("click", (ev) => {{
    if (ev.target === detailModal) hideDetail();
  }});
  document.addEventListener("keydown", (ev) => {{
    if (ev.key === "Escape") hideDetail();
  }});
  detailTabs.addEventListener("click", (ev) => {{
    const btn = ev.target.closest(".tab-btn");
    if (!btn) return;
    const tab = btn.dataset.tab;
    if (!tab) return;
    setDetailTab(tab);
    lockDetailModalSize(tab);
  }});
  window.addEventListener("resize", () => {{
    if (detailModal.style.display !== "none") lockDetailModalSize(currentDetailTab());
  }});
  if (mainTableHead) {{
    mainTableHead.addEventListener("click", (ev) => {{
      const tgt = ev.target;
      const el = (tgt && typeof tgt.closest === "function")
        ? tgt
        : (tgt && tgt.parentElement ? tgt.parentElement : null);
      const th = el ? el.closest("th.sortable") : null;
      if (!th) return;
      const key = th.dataset.sortKey;
      if (!key || !SORT_DEFS[key]) return;
      if (sortState.key === key) {{
        sortState = {{ key, dir: sortState.dir === "asc" ? "desc" : "asc" }};
      }} else {{
        sortState = {{ key, dir: "asc" }};
      }}
      render();
    }});
  }}
  [q, band, maxpick, rowsN].forEach(el => el.addEventListener("input", render));
  if (Array.isArray(YEAR_OPTIONS) && YEAR_OPTIONS.length > 0) {{
    holdoutYearSel.innerHTML = YEAR_OPTIONS.map((y) => `<option value="${{y}}">${{y}}</option>`).join("");
    holdoutYearSel.value = String(MET.holdout_year || "");
    holdoutYearSel.addEventListener("change", () => {{
      const y = holdoutYearSel.value;
      if (!y) return;
      window.location.href = `nba_impact_dashboard_real_vs_predicted_${{y}}.html?v=${{DASH_VER}}`;
    }});
  }} else {{
    holdoutYearSel.innerHTML = `<option value="{holdout_year}">{holdout_year}</option>`;
    holdoutYearSel.disabled = true;
  }}
  render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if not args.holdout_csv.exists():
        raise SystemExit(f"Missing holdout CSV: {args.holdout_csv}")
    if not args.pred_csv.exists():
        raise SystemExit(f"Missing predictions CSV: {args.pred_csv}")
    if not args.metrics_csv.exists():
        raise SystemExit(f"Missing metrics CSV: {args.metrics_csv}")

    hold = pd.read_csv(args.holdout_csv, low_memory=False)
    pred = pd.read_csv(args.pred_csv, low_memory=False)
    met = pd.read_csv(args.metrics_csv, low_memory=False)

    for c in ["actual_impact_z", "pred_impact_z", "pick_number"]:
        if c in hold.columns:
            hold[c] = _num(hold[c])
    for c in ["pred_nba_impact_z", "pick_number"]:
        if c in pred.columns:
            pred[c] = _num(pred[c])

    # Backfill combine measurements directly from official Draft_antro_YYYY files
    # so popup measurements stay populated even when model exports are stale.
    comb = load_combine_lookup(args.combine_dir)
    if not comb.empty:
        for frame_name in ("hold", "pred"):
            dfx = hold.copy() if frame_name == "hold" else pred.copy()
            dfx["season"] = pd.to_numeric(dfx.get("season"), errors="coerce")
            dfx["name_key"] = dfx.get("name", pd.Series([""] * len(dfx))).map(_name_key)
            dfx = dfx.merge(
                comb,
                how="left",
                on=["season", "name_key"],
                suffixes=("", "_cmb"),
            )
            for c in [
                "combine_height_wo_shoes_in",
                "combine_weight_lb",
                "combine_wingspan_in",
                "combine_standing_reach_in",
                "combine_body_fat_pct",
                "combine_hand_length_in",
                "combine_hand_width_in",
                "combine_wingspan_minus_height",
                "combine_standing_reach_minus_height",
            ]:
                cmb = f"{c}_cmb"
                if c not in dfx.columns:
                    dfx[c] = pd.Series([float("nan")] * len(dfx), index=dfx.index, dtype=float)
                dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
                if cmb in dfx.columns:
                    dfx[c] = dfx[c].where(dfx[c].notna(), pd.to_numeric(dfx[cmb], errors="coerce"))
            for c in [x for x in dfx.columns if x.endswith("_cmb")]:
                dfx = dfx.drop(columns=[c])
            if frame_name == "hold":
                hold = dfx
            else:
                pred = dfx

    crafted = load_crafted_lookup(args.crafted_measurements_csv)
    if not crafted.empty:
        for frame_name in ("hold", "pred"):
            dfx = hold.copy() if frame_name == "hold" else pred.copy()
            dfx["name_key"] = dfx.get("name", pd.Series([""] * len(dfx))).map(_name_key)
            dfx = dfx.merge(crafted, how="left", on=["name_key"], suffixes=("", "_crf"))
            for c in ["crafted_height_in", "crafted_wingspan_in", "crafted_length_in", "crafted_wingspan_minus_height"]:
                crf = f"{c}_crf"
                if c not in dfx.columns:
                    dfx[c] = pd.Series([float("nan")] * len(dfx), index=dfx.index, dtype=float)
                dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
                if crf in dfx.columns:
                    dfx[c] = dfx[c].where(dfx[c].notna(), pd.to_numeric(dfx[crf], errors="coerce"))
            for c in [x for x in dfx.columns if x.endswith("_crf")]:
                dfx = dfx.drop(columns=[c])
            if frame_name == "hold":
                hold = dfx
            else:
                pred = dfx

    career = load_nba_career_lookup(args.nba_advanced_csv)
    if not career.empty:
        for frame_name in ("hold", "pred"):
            dfx = hold.copy() if frame_name == "hold" else pred.copy()
            dfx["name_key"] = dfx.get("name", pd.Series([""] * len(dfx))).map(_name_key)
            dfx = dfx.merge(career, how="left", on=["name_key"])
            if frame_name == "hold":
                hold = dfx
            else:
                pred = dfx

    nba_disp = load_nba_display_lookup(args.nba_display_csv)
    if not nba_disp.empty:
        for frame_name in ("hold", "pred"):
            dfx = hold.copy() if frame_name == "hold" else pred.copy()
            dfx["name_key"] = dfx.get("name", pd.Series([""] * len(dfx))).map(_name_key)
            dfx = dfx.merge(nba_disp, how="left", on=["name_key"], suffixes=("", "_nbd"))
            for c in [
                "nba_games",
                "nba_minutes",
                "nba_points_per75",
                "nba_assists_per100",
                "nba_steals_per100",
                "nba_blocks_per100",
                "nba_stocks_per100",
                "nba_ts_pct",
                "nba_three_point_pct",
                "nba_three_pa_per100",
                "nba_tov_per100",
            ]:
                nbd = f"{c}_nbd"
                if c not in dfx.columns:
                    dfx[c] = pd.Series([float("nan")] * len(dfx), index=dfx.index, dtype=float)
                dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
                if nbd in dfx.columns:
                    dfx[c] = dfx[c].where(dfx[c].notna(), pd.to_numeric(dfx[nbd], errors="coerce"))
            if "nba_team_display" not in dfx.columns:
                dfx["nba_team_display"] = ""
            if "nba_position" not in dfx.columns:
                dfx["nba_position"] = ""
            if "nba_team_display_nbd" in dfx.columns:
                dfx["nba_team_display"] = dfx["nba_team_display"].where(
                    dfx["nba_team_display"].astype(str).str.strip() != "",
                    dfx["nba_team_display_nbd"],
                )
            if "nba_position_nbd" in dfx.columns:
                dfx["nba_position"] = dfx["nba_position"].where(
                    dfx["nba_position"].astype(str).str.strip() != "",
                    dfx["nba_position_nbd"],
                )
            dfx["nba_team_display"] = (
                dfx.get("nba_team_display", pd.Series([""] * len(dfx), index=dfx.index))
                .astype(str)
                .str.strip()
            )
            dfx["nba_position"] = (
                dfx.get("nba_position", pd.Series([""] * len(dfx), index=dfx.index))
                .astype(str)
                .str.strip()
            )
            dfx["nba_team_display"] = dfx["nba_team_display"].where(
                dfx["nba_team_display"] != "",
                dfx.get("nba_team_abbr", pd.Series([""] * len(dfx), index=dfx.index)).astype(str).str.strip(),
            )
            dfx["nba_position"] = dfx["nba_position"].where(
                dfx["nba_position"] != "",
                dfx.get("position_group", pd.Series([""] * len(dfx), index=dfx.index)).astype(str).str.strip(),
            )
            for c in [x for x in dfx.columns if x.endswith("_nbd")]:
                dfx = dfx.drop(columns=[c])
            if frame_name == "hold":
                hold = dfx
            else:
                pred = dfx

    nba_adv = load_nba_advanced_season_lookup(args.nba_advanced_csv)
    if not nba_adv.empty:
        for frame_name in ("hold", "pred"):
            dfx = hold.copy() if frame_name == "hold" else pred.copy()
            dfx["name_key"] = dfx.get("name", pd.Series([""] * len(dfx))).map(_name_key)
            dfx = dfx.merge(nba_adv, how="left", on=["name_key"])

            for c in ["nba_games", "nba_minutes"]:
                adv = f"{c}_adv"
                if c not in dfx.columns:
                    dfx[c] = pd.Series([float("nan")] * len(dfx), index=dfx.index, dtype=float)
                dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
                if adv in dfx.columns:
                    dfx[c] = dfx[c].where(dfx[c].notna(), pd.to_numeric(dfx[adv], errors="coerce"))

            for c in ["nba_team_display", "nba_position", "nba_team_history", "nba_position_history"]:
                adv = f"{c}_adv"
                if c not in dfx.columns:
                    dfx[c] = ""
                dfx[c] = dfx[c].fillna("").astype(str).str.strip()
                if adv in dfx.columns:
                    dfx[c] = dfx[c].where(dfx[c] != "", dfx[adv].fillna("").astype(str).str.strip())

            if "nba_seasons_adv" in dfx.columns:
                if "nba_seasons" not in dfx.columns:
                    dfx["nba_seasons"] = [[] for _ in range(len(dfx))]
                dfx["nba_seasons"] = dfx.apply(
                    lambda r: _combine_nba_season_rows(r.get("nba_seasons"), r.get("nba_seasons_adv")),
                    axis=1,
                )

            for c in [x for x in dfx.columns if x.endswith("_adv")]:
                dfx = dfx.drop(columns=[c])
            if frame_name == "hold":
                hold = dfx
            else:
                pred = dfx

    college_hist = load_college_season_lookup(args.training_csv)
    if not college_hist.empty:
        for frame_name in ("hold", "pred"):
            dfx = hold.copy() if frame_name == "hold" else pred.copy()
            dfx["name_key"] = dfx.get("name", pd.Series([""] * len(dfx))).map(_name_key)
            dfx["season"] = pd.to_numeric(dfx.get("season"), errors="coerce")
            dfx = dfx.merge(college_hist, how="left", on=["name_key", "season"])
            if frame_name == "hold":
                hold = dfx
            else:
                pred = dfx

    if args.draft_history_csv.exists():
        dh = pd.read_csv(args.draft_history_csv, low_memory=False)
        if {"draft_year", "pick_overall", "team_abbr"}.issubset(set(dh.columns)):
            dm = dh[["draft_year", "pick_overall", "team_abbr"]].copy()
            dm["draft_year"] = pd.to_numeric(dm["draft_year"], errors="coerce")
            dm["pick_overall"] = pd.to_numeric(dm["pick_overall"], errors="coerce")
            dm["team_abbr"] = dm["team_abbr"].astype(str).str.strip()
            dm = dm.dropna(subset=["draft_year", "pick_overall"]).drop_duplicates(
                subset=["draft_year", "pick_overall"], keep="first"
            )
            dm = dm.rename(columns={"draft_year": "season", "pick_overall": "pick_number", "team_abbr": "nba_team_abbr"})
            hold["season"] = pd.to_numeric(hold.get("season"), errors="coerce")
            pred["season"] = pd.to_numeric(pred.get("season"), errors="coerce")
            hold = hold.merge(dm, how="left", on=["season", "pick_number"])
            pred = pred.merge(dm, how="left", on=["season", "pick_number"])

    # Fill missing season-level NBA team tabs (common in fallback advanced rows)
    # using draft-team abbreviation for the first NBA season when available.
    for frame_name in ("hold", "pred"):
        dfx = hold.copy() if frame_name == "hold" else pred.copy()
        if "nba_seasons" in dfx.columns:
            dfx["nba_seasons"] = dfx.apply(
                lambda r: _backfill_nba_season_team_rows(
                    r.get("nba_seasons"),
                    r.get("nba_team_abbr"),
                    r.get("nba_team_display"),
                    r.get("nba_start_year"),
                ),
                axis=1,
            )
        if frame_name == "hold":
            hold = dfx
        else:
            pred = dfx

    # Cohort guard for holdout: keep 2022-class entrants (drafted or signed in class window).
    # `nba_start_year` is season-end (e.g., 2023 corresponds to 2022-23 entry).
    hold_season_num = pd.to_numeric(hold.get("season"), errors="coerce")
    holdout_year_num = int(hold_season_num.dropna().iloc[0]) if hold_season_num.notna().any() else None
    if holdout_year_num is not None:
        pick_num = pd.to_numeric(hold.get("pick_number"), errors="coerce")
        drafted_num = pd.to_numeric(hold.get("drafted"), errors="coerce")
        nba_start_num = pd.to_numeric(hold.get("nba_start_year"), errors="coerce")
        nba_entry_num = nba_start_num - 1.0
        hold["nba_entry_year"] = nba_entry_num
        same_season = hold_season_num.eq(float(holdout_year_num))
        cohort_keep = same_season & (
            pick_num.notna()
            | drafted_num.eq(1.0)
            | nba_entry_num.eq(float(holdout_year_num))
        )
        if not bool(cohort_keep.any()):
            cohort_keep = same_season
        hold = hold.loc[cohort_keep].copy()

    best2_mode = {"actual_rapm_best2_mean", "pred_rapm_best2_mean"}.issubset(set(hold.columns))
    if best2_mode:
        hold["nba_rapm_current"] = _num(hold["actual_rapm_best2_mean"])
        hold["pred_rapm_current_est"] = _num(hold["pred_rapm_best2_mean"])
        if "pred_rapm_best2_mean" in pred.columns:
            pred["pred_rapm_current_est"] = _num(pred["pred_rapm_best2_mean"])
        elif "pred_nba_impact_z" in pred.columns:
            pred["pred_rapm_current_est"] = _num(pred["pred_nba_impact_z"])
        else:
            pred["pred_rapm_current_est"] = float("nan")
        # Recalibrate best2 predictions onto observed holdout RAPM distribution
        # so dashboard values are not over-compressed to one side of zero.
        calib_b2 = hold[["pred_rapm_current_est", "nba_rapm_current"]].copy().dropna()
        hold_q = _map_to_empirical_rapm(
            pd.to_numeric(hold.get("pred_rapm_current_est"), errors="coerce"),
            calib_b2["pred_rapm_current_est"] if not calib_b2.empty else pd.Series(dtype=float),
            calib_b2["nba_rapm_current"] if not calib_b2.empty else pd.Series(dtype=float),
        )
        pred_q = _map_to_empirical_rapm(
            pd.to_numeric(pred.get("pred_rapm_current_est"), errors="coerce"),
            calib_b2["pred_rapm_current_est"] if not calib_b2.empty else pd.Series(dtype=float),
            calib_b2["nba_rapm_current"] if not calib_b2.empty else pd.Series(dtype=float),
        )
        hold["pred_rapm_current_est"] = hold_q.where(hold_q.notna(), hold["pred_rapm_current_est"])
        pred["pred_rapm_current_est"] = pred_q.where(pred_q.notna(), pred["pred_rapm_current_est"])
    else:
        if args.rapm_csv.exists():
            rapm = pd.read_csv(args.rapm_csv, low_memory=False)
            if {"player_name", "rapm"}.issubset(set(rapm.columns)):
                rapm = rapm[["player_name", "rapm"]].copy()
                rapm["name_key"] = rapm["player_name"].map(_name_key)
                rapm["rapm"] = pd.to_numeric(rapm["rapm"], errors="coerce")
                rapm = rapm.sort_values(["rapm"], ascending=False, kind="stable").drop_duplicates(
                    subset=["name_key"], keep="first"
                )
                rapm = rapm.rename(columns={"rapm": "nba_rapm_current"})
                hold["name_key"] = hold["name"].map(_name_key)
                pred["name_key"] = pred["name"].map(_name_key)
                hold = hold.merge(rapm[["name_key", "nba_rapm_current"]], on="name_key", how="left")
                pred = pred.merge(rapm[["name_key", "nba_rapm_current"]], on="name_key", how="left")

        # Map model z-score to RAPM scale using holdout rows with known RAPM.
        calib = hold[["pred_impact_z", "nba_rapm_current"]].copy()
        calib["pred_impact_z"] = pd.to_numeric(calib["pred_impact_z"], errors="coerce")
        calib["nba_rapm_current"] = pd.to_numeric(calib["nba_rapm_current"], errors="coerce")
        calib = calib.dropna()
        if len(calib) >= 2:
            x = calib["pred_impact_z"]
            y = calib["nba_rapm_current"]
            x_mu = float(x.mean())
            y_mu = float(y.mean())
            x_var = float(((x - x_mu) ** 2).mean())
            if x_var > 1e-12:
                slope = float(((x - x_mu) * (y - y_mu)).mean() / x_var)
                intercept = float(y_mu - slope * x_mu)
            else:
                slope = 0.0
                intercept = y_mu
        else:
            slope = 1.0
            intercept = 0.0
        hold_lin = intercept + slope * pd.to_numeric(hold.get("pred_impact_z"), errors="coerce")
        pred_lin = intercept + slope * pd.to_numeric(pred.get("pred_nba_impact_z"), errors="coerce")
        hold_q = _map_to_empirical_rapm(
            pd.to_numeric(hold.get("pred_impact_z"), errors="coerce"),
            calib["pred_impact_z"] if not calib.empty else pd.Series(dtype=float),
            calib["nba_rapm_current"] if not calib.empty else pd.Series(dtype=float),
        )
        pred_q = _map_to_empirical_rapm(
            pd.to_numeric(pred.get("pred_nba_impact_z"), errors="coerce"),
            calib["pred_impact_z"] if not calib.empty else pd.Series(dtype=float),
            calib["nba_rapm_current"] if not calib.empty else pd.Series(dtype=float),
        )
        hold["pred_rapm_current_est"] = hold_q.where(hold_q.notna(), hold_lin)
        pred["pred_rapm_current_est"] = pred_q.where(pred_q.notna(), pred_lin)

    holdout_year = (
        str(int(pd.to_numeric(hold["season"], errors="coerce").dropna().iloc[0])) if not hold.empty else "n/a"
    )
    predict_year = (
        str(int(pd.to_numeric(pred["season"], errors="coerce").dropna().iloc[0])) if ("season" in pred.columns and not pred.empty) else "n/a"
    )
    rapm_eval = hold[["nba_rapm_current", "pred_rapm_current_est"]].copy()
    rapm_eval["nba_rapm_current"] = pd.to_numeric(rapm_eval["nba_rapm_current"], errors="coerce")
    rapm_eval["pred_rapm_current_est"] = pd.to_numeric(rapm_eval["pred_rapm_current_est"], errors="coerce")
    rapm_eval = rapm_eval.dropna()
    hold["actual_rank"] = pd.to_numeric(hold.get("nba_rapm_current"), errors="coerce").rank(
        method="min", ascending=False
    )
    hold["pred_rank"] = pd.to_numeric(hold.get("pred_rapm_current_est"), errors="coerce").rank(
        method="min", ascending=False
    )
    hold["actual_rapm_label"] = hold["nba_rapm_current"].map(rapm_label)
    hold["pred_rapm_label"] = hold["pred_rapm_current_est"].map(rapm_label)
    hold["label_distance"] = hold["pred_rapm_label"].map(label_to_ord) - hold["actual_rapm_label"].map(label_to_ord)
    hold["label_distance"] = pd.to_numeric(hold["label_distance"], errors="coerce").abs()
    hold["rank_error"] = pd.to_numeric(hold["pred_rank"], errors="coerce") - pd.to_numeric(
        hold["actual_rank"], errors="coerce"
    )
    pred["pred_rapm_label"] = pred.get("pred_rapm_current_est", pd.Series([float("nan")] * len(pred), index=pred.index)).map(rapm_label)
    for dfx in (hold, pred):
        if "stocks_per40" not in dfx.columns:
            dfx["stocks_per40"] = pd.to_numeric(dfx.get("blocks_per40"), errors="coerce").fillna(0.0) + pd.to_numeric(
                dfx.get("steals_per40"), errors="coerce"
            ).fillna(0.0)
        if "assist_to_turnover" not in dfx.columns:
            ast = pd.to_numeric(dfx.get("assists_per40"), errors="coerce")
            tov = pd.to_numeric(dfx.get("turnovers_per40"), errors="coerce")
            dfx["assist_to_turnover"] = ast / tov.replace(0, pd.NA)

        dfx["measurement_height_in"], dfx["measurement_height_source"] = _coalesce_numeric(
            dfx,
            [
                ("combine_height_wo_shoes_in", "NBA combine"),
                ("height_in", "bio"),
                ("crafted_height_in", "crafted"),
                ("ext_height_in_modeldb", "external model db"),
            ],
        )
        dfx["measurement_weight_lb"], dfx["measurement_weight_source"] = _coalesce_numeric(
            dfx,
            [
                ("combine_weight_lb", "NBA combine"),
                ("weight_lb", "bio"),
                ("ext_weight_lb_modeldb", "external model db"),
            ],
        )
        dfx["measurement_age"], dfx["measurement_age_source"] = _coalesce_numeric(
            dfx,
            [
                ("ext_draft_day_age", "external model db"),
                ("age", "bio"),
            ],
        )
        dfx["measurement_wingspan_in"], dfx["measurement_wingspan_source"] = _coalesce_numeric(
            dfx,
            [
                ("combine_wingspan_in", "NBA combine"),
                ("crafted_wingspan_in", "crafted"),
                ("wingspan_in", "other source"),
            ],
        )
        dfx["measurement_standing_reach_in"], dfx["measurement_standing_reach_source"] = _coalesce_numeric(
            dfx,
            [
                ("combine_standing_reach_in", "NBA combine"),
                ("standing_reach_in", "other source"),
            ],
        )

        # Estimate missing physical measurements from available size signals.
        dfx["measurement_wingspan_in"], ws_est_mask = _impute_linear(
            dfx,
            "measurement_wingspan_in",
            ["measurement_height_in"],
            min_fit_rows=4,
        )
        if "measurement_wingspan_source" in dfx.columns:
            dfx.loc[
                ws_est_mask & dfx["measurement_wingspan_source"].astype(str).eq(""),
                "measurement_wingspan_source",
            ] = "estimated"

        dfx["measurement_standing_reach_in"], reach_est_mask = _impute_linear(
            dfx,
            "measurement_standing_reach_in",
            ["measurement_height_in", "measurement_wingspan_in"],
            min_fit_rows=4,
        )
        if "measurement_standing_reach_source" in dfx.columns:
            dfx.loc[
                reach_est_mask & dfx["measurement_standing_reach_source"].astype(str).eq(""),
                "measurement_standing_reach_source",
            ] = "estimated"

        dfx["combine_hand_length_in"], _ = _impute_linear(
            dfx,
            "combine_hand_length_in",
            ["measurement_height_in", "measurement_wingspan_in"],
            min_fit_rows=4,
        )
        dfx["combine_hand_width_in"], _ = _impute_linear(
            dfx,
            "combine_hand_width_in",
            ["measurement_height_in", "measurement_wingspan_in"],
            min_fit_rows=4,
        )
        dfx["combine_body_fat_pct"], _ = _impute_linear(
            dfx,
            "combine_body_fat_pct",
            ["measurement_height_in", "measurement_weight_lb"],
            min_fit_rows=4,
        )

        dfx["measurement_wingspan_minus_height"] = (
            dfx["measurement_wingspan_in"] - dfx["measurement_height_in"]
        )
        dfx["measurement_reach_minus_height"] = (
            dfx["measurement_standing_reach_in"] - dfx["measurement_height_in"]
        )
        season_num = pd.to_numeric(
            dfx.get("season", pd.Series([float("nan")] * len(dfx), index=dfx.index)),
            errors="coerce",
        )
        draft_age = pd.to_numeric(dfx["measurement_age"], errors="coerce")
        dfx["age_draft_years"] = draft_age
        current_year_num = float(pd.Timestamp.utcnow().year)
        dfx["age_current_years_est"] = draft_age + (current_year_num - season_num)
        dfx["birth_year_est"] = season_num - draft_age
        dfx["abs_rank_error"] = (
            pd.to_numeric(
                dfx.get("rank_error", pd.Series([float("nan")] * len(dfx), index=dfx.index)),
                errors="coerce",
            )
            .abs()
        )
        yrs_since = pd.to_numeric(
            dfx.get("years_since_first_seen", pd.Series([float("nan")] * len(dfx), index=dfx.index)),
            errors="coerce",
        )
        first_seen = pd.to_numeric(
            dfx.get("first_seen_season", pd.Series([float("nan")] * len(dfx), index=dfx.index)),
            errors="coerce",
        )
        dfx["college_start_year"] = first_seen.where(first_seen.notna(), season_num - yrs_since)
        yrs_college = pd.to_numeric(
            dfx.get("years_in_college", pd.Series([float("nan")] * len(dfx), index=dfx.index)),
            errors="coerce",
        )
        dfx["college_years_played"] = yrs_college.where(yrs_college.notna(), yrs_since + 1.0)
        dfx["nba_start_year"] = pd.to_numeric(dfx.get("nba_start_year"), errors="coerce")
        dfx["nba_entry_year"] = dfx["nba_start_year"] - 1.0
        dfx["nba_years_played"] = pd.to_numeric(dfx.get("nba_years_played"), errors="coerce")
    if not rapm_eval.empty:
        abs_err = (rapm_eval["pred_rapm_current_est"] - rapm_eval["nba_rapm_current"]).abs()
        rapm_mae = float(abs_err.mean())
        rapm_rmse = float(((rapm_eval["pred_rapm_current_est"] - rapm_eval["nba_rapm_current"]) ** 2).mean() ** 0.5)
        rapm_corr = float(rapm_eval["pred_rapm_current_est"].corr(rapm_eval["nba_rapm_current"]))
        rapm_hits_075 = int((abs_err <= 0.75).sum())
        rapm_hit_rate_075 = float(rapm_hits_075 / len(abs_err))
        rapm_hits = int((abs_err <= 1.0).sum())
        rapm_hit_rate = float(rapm_hits / len(abs_err))
    else:
        rapm_mae = float("nan")
        rapm_rmse = float("nan")
        rapm_corr = float("nan")
        rapm_hits_075 = 0
        rapm_hit_rate_075 = float("nan")
        rapm_hits = 0
        rapm_hit_rate = float("nan")
    rank_eval = hold[["actual_rank", "pred_rank"]].copy()
    rank_eval["actual_rank"] = pd.to_numeric(rank_eval["actual_rank"], errors="coerce")
    rank_eval["pred_rank"] = pd.to_numeric(rank_eval["pred_rank"], errors="coerce")
    rank_eval = rank_eval.dropna()
    rank_mae = float((rank_eval["pred_rank"] - rank_eval["actual_rank"]).abs().mean()) if not rank_eval.empty else float("nan")
    if not rank_eval.empty:
        rank_abs = (rank_eval["pred_rank"] - rank_eval["actual_rank"]).abs()
        rank_hits_5 = int((rank_abs <= 5).sum())
        rank_hit_rate_5 = float(rank_hits_5 / len(rank_abs))
        rank_hits_10 = int((rank_abs <= 10).sum())
        rank_hit_rate_10 = float(rank_hits_10 / len(rank_abs))
    else:
        rank_hits_5 = 0
        rank_hit_rate_5 = float("nan")
        rank_hits_10 = 0
        rank_hit_rate_10 = float("nan")

    metrics = {
        "holdout_year": holdout_year,
        "n_holdout": f"{len(hold):,}",
        "n_rapm": f"{len(rapm_eval):,}",
        "rapm_mae": f"{rapm_mae:.3f}" if pd.notna(rapm_mae) else "n/a",
        "rapm_rmse": f"{rapm_rmse:.3f}" if pd.notna(rapm_rmse) else "n/a",
        "rapm_corr": f"{rapm_corr:.3f}" if pd.notna(rapm_corr) else "n/a",
        "rapm_hits_075": f"{rapm_hits_075:,}",
        "rapm_hit_rate_075": f"{100.0 * rapm_hit_rate_075:.1f}%" if pd.notna(rapm_hit_rate_075) else "n/a",
        "rapm_hits": f"{rapm_hits:,}",
        "rapm_hit_rate": f"{100.0 * rapm_hit_rate:.1f}%" if pd.notna(rapm_hit_rate) else "n/a",
        "rank_mae": f"{rank_mae:.2f}" if pd.notna(rank_mae) else "n/a",
        "rank_hits_5": f"{rank_hits_5:,}",
        "rank_hit_rate_5": f"{100.0 * rank_hit_rate_5:.1f}%" if pd.notna(rank_hit_rate_5) else "n/a",
        "rank_hits_10": f"{rank_hits_10:,}",
        "rank_hit_rate_10": f"{100.0 * rank_hit_rate_10:.1f}%" if pd.notna(rank_hit_rate_10) else "n/a",
    }

    hold = hold.join(_build_driver_and_miss_text(hold))

    hold_keep = [
        "season",
        "name",
        "team",
        "pick_number",
        "nba_team_abbr",
        "actual_rank",
        "pred_rank",
        "rank_error",
        "abs_rank_error",
        "nba_rapm_current",
        "pred_rapm_current_est",
        "actual_rapm_label",
        "pred_rapm_label",
        "label_distance",
        "actual_impact_z",
        "pred_impact_z",
        "driver_positive_short",
        "driver_negative_short",
        "miss_explain_short",
        "college_seasons",
        "college_team_history",
        "college_position_history",
        "nba_seasons",
        "nba_team_history",
        "nba_position_history",
        "age_draft_years",
        "age_current_years_est",
        "birth_year_est",
        "age_display_years",
        "college_start_year",
        "college_years_played",
        "nba_start_year",
        "nba_years_played",
        "age",
        "games",
        "minutes",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "steals_per40",
        "blocks_per40",
        "stocks_per40",
        "usage",
        "true_shooting_pct",
        "three_point_pct",
        "three_point_attempt_rate",
        "assist_to_turnover",
        "turnovers_per40",
        "net_rating",
        "height_in",
        "weight_lb",
        "ext_draft_day_age",
        "ext_height_in_modeldb",
        "ext_weight_lb_modeldb",
        "measurement_height_in",
        "measurement_height_source",
        "measurement_weight_lb",
        "measurement_weight_source",
        "measurement_age",
        "measurement_age_source",
        "measurement_wingspan_in",
        "measurement_wingspan_source",
        "measurement_standing_reach_in",
        "measurement_standing_reach_source",
        "measurement_wingspan_minus_height",
        "measurement_reach_minus_height",
        "crafted_height_in",
        "crafted_wingspan_in",
        "crafted_length_in",
        "crafted_wingspan_minus_height",
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "combine_body_fat_pct",
        "combine_hand_length_in",
        "combine_hand_width_in",
        "combine_wingspan_minus_height",
        "combine_standing_reach_minus_height",
        "first_seen_season",
        "years_since_first_seen",
        "years_in_college",
    ]
    hold_keep = [c for c in hold_keep if c in hold.columns]
    hold_rec = hold[hold_keep].copy().to_dict(orient="records")

    board_keep = [
        "name",
        "team",
        "pick_number",
        "nba_team_abbr",
        "nba_rapm_current",
        "pred_rapm_current_est",
        "pred_rapm_label",
        "pred_nba_impact_z",
        "abs_rank_error",
        "college_seasons",
        "college_team_history",
        "college_position_history",
        "nba_seasons",
        "nba_team_history",
        "nba_position_history",
        "age_draft_years",
        "age_current_years_est",
        "birth_year_est",
        "age_display_years",
        "college_start_year",
        "college_years_played",
        "nba_start_year",
        "nba_years_played",
        "age",
        "games",
        "minutes",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "steals_per40",
        "blocks_per40",
        "stocks_per40",
        "usage",
        "true_shooting_pct",
        "three_point_pct",
        "three_point_attempt_rate",
        "assist_to_turnover",
        "turnovers_per40",
        "net_rating",
        "height_in",
        "weight_lb",
        "ext_draft_day_age",
        "ext_height_in_modeldb",
        "ext_weight_lb_modeldb",
        "measurement_height_in",
        "measurement_height_source",
        "measurement_weight_lb",
        "measurement_weight_source",
        "measurement_age",
        "measurement_age_source",
        "measurement_wingspan_in",
        "measurement_wingspan_source",
        "measurement_standing_reach_in",
        "measurement_standing_reach_source",
        "measurement_wingspan_minus_height",
        "measurement_reach_minus_height",
        "crafted_height_in",
        "crafted_wingspan_in",
        "crafted_length_in",
        "crafted_wingspan_minus_height",
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "combine_body_fat_pct",
        "combine_hand_length_in",
        "combine_hand_width_in",
        "combine_wingspan_minus_height",
        "combine_standing_reach_minus_height",
        "first_seen_season",
        "years_since_first_seen",
        "years_in_college",
    ]
    board_keep = [c for c in board_keep if c in pred.columns]
    board_rec = pred[board_keep].copy().sort_values("pred_rapm_current_est", ascending=False, kind="stable").to_dict(orient="records")
    holdout_glob = re.sub(r"_(\d{4})\\.csv$", "_*.csv", args.holdout_csv.name)
    if holdout_glob == args.holdout_csv.name:
        holdout_glob = "nba_success_rapm_holdout_actual_vs_predicted_*.csv"

    html = build_html(
        holdout_json=json.dumps(hold_rec, ensure_ascii=True, separators=(",", ":")),
        board_json=json.dumps(board_rec, ensure_ascii=True, separators=(",", ":")),
        metrics_json=json.dumps(metrics, ensure_ascii=True, separators=(",", ":")),
        holdout_year=holdout_year,
        predict_year=predict_year,
        year_options_json=json.dumps(
            sorted(
                {
                    int(m.group(1))
                    for p in args.holdout_csv.parent.glob(holdout_glob)
                    for m in [re.search(r"_(\d{4})\.csv$", p.name)]
                    if m and int(m.group(1)) >= 2018
                }
            ),
            ensure_ascii=True,
            separators=(",", ":"),
        ),
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
