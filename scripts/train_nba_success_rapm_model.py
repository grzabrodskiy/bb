from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_nba_draft_predictors as draft_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train NBA success model: college profile -> best 2-year NBA RAPM peak."
    )
    p.add_argument(
        "--training-csv",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
    )
    p.add_argument(
        "--rapm-csv",
        type=Path,
        default=Path("data/raw/external/rapm_history/rapm_history_regular.csv"),
        help="Historical regular-season RAPM table from scripts/download_rapm_history.py",
    )
    p.add_argument(
        "--target-source",
        type=str,
        default="rapm_history",
        choices=["rapm_history", "darko_dpm_proxy", "multi_metric_proxy"],
        help=(
            "Target source for best-2-year NBA impact. "
            "'rapm_history' uses historical RAPM directly; "
            "'darko_dpm_proxy' uses DARKO dpm calibrated onto RAPM scale; "
            "'multi_metric_proxy' uses a season-level composite of multiple public metrics calibrated onto RAPM scale."
        ),
    )
    p.add_argument(
        "--darko-csv",
        type=Path,
        default=Path("data/raw/external/nbarapm/DARKO.csv"),
        help="Season-level DARKO CSV used when --target-source=darko_dpm_proxy.",
    )
    p.add_argument(
        "--lebron-csv",
        type=Path,
        default=Path("data/raw/external/nbarapm/lebron.csv"),
        help="Season-level LEBRON CSV used by --target-source=multi_metric_proxy.",
    )
    p.add_argument(
        "--mamba-csv",
        type=Path,
        default=Path("data/raw/external/nbarapm/mamba.csv"),
        help="Season-level MAMBA CSV used by --target-source=multi_metric_proxy.",
    )
    p.add_argument(
        "--raptor-csv",
        type=Path,
        default=Path("data/raw/external/nbarapm/raptor.csv"),
        help="Season-level RAPTOR CSV used by --target-source=multi_metric_proxy.",
    )
    p.add_argument(
        "--bref-csv",
        type=Path,
        default=Path("data/raw/nba/bref/player_advanced_2010_2026.csv"),
        help="Basketball-Reference advanced CSV used by --target-source=multi_metric_proxy.",
    )
    p.add_argument(
        "--proxy-min-metrics",
        type=int,
        default=2,
        help="Minimum number of available component metrics per player-season for multi-metric proxy.",
    )
    p.add_argument(
        "--combine-dir",
        type=Path,
        default=Path("data/raw/external/nba_stats_draft/antro/antro"),
    )
    p.add_argument(
        "--target-window-years",
        type=int,
        default=2,
        help="Use top N post-draft RAPM seasons (by RAPM value).",
    )
    p.add_argument(
        "--test-draft-year",
        type=int,
        default=2017,
        help="Preferred holdout exit year used for offline check (auto-falls back if unavailable).",
    )
    p.add_argument(
        "--model-start-year",
        type=int,
        default=None,
        help="Optional lower bound (inclusive) for entrant exit years used to train the model.",
    )
    p.add_argument(
        "--model-end-year",
        type=int,
        default=None,
        help="Optional upper bound (inclusive) for entrant exit years used to train the model.",
    )
    p.add_argument(
        "--predict-draft-year",
        type=int,
        default=2025,
        help="Single exit-year cohort to score when no predict-year range is provided.",
    )
    p.add_argument(
        "--predict-start-year",
        type=int,
        default=None,
        help="Optional lower bound (inclusive) for exit-year cohort scoring.",
    )
    p.add_argument(
        "--predict-end-year",
        type=int,
        default=None,
        help="Optional upper bound (inclusive) for exit-year cohort scoring.",
    )
    p.add_argument(
        "--selection-objective",
        type=str,
        default="hit075",
        choices=["hit075", "rapm_mae", "rapm_rmse", "spearman", "rank_mae"],
        help="Objective used to choose final model on holdout.",
    )
    p.add_argument(
        "--force-model",
        type=str,
        default="",
        help=(
            "Optional explicit final model key (e.g. ridge_a400). "
            "When set, overrides auto-selection and uses that fitted model."
        ),
    )
    p.add_argument(
        "--feature-profile",
        type=str,
        default="full",
        choices=["full", "no_ext", "core"],
        help="Feature profile to control bias/variance on small historical RAPM sample.",
    )
    p.add_argument(
        "--disable-linear-calibration",
        action="store_true",
        help="Disable post-model linear calibration on training predictions.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
    )
    return p.parse_args()


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return x * 0.0
    return (x - mu) / sd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = pd.Series(y_true).rank(method="average")
    yp = pd.Series(y_pred).rank(method="average")
    c = yt.corr(yp)
    return float(c) if pd.notna(c) else float("nan")


def _safe_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _name_key(v: object) -> str:
    return draft_model.normalize_name(v)


def fit_linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return 0.0, 1.0
    x = x[m]
    y = y[m]
    x_mu = float(np.mean(x))
    y_mu = float(np.mean(y))
    x_var = float(np.mean((x - x_mu) ** 2))
    if x_var <= 1e-12:
        return y_mu, 0.0
    b = float(np.mean((x - x_mu) * (y - y_mu)) / x_var)
    a = float(y_mu - b * x_mu)
    return a, b


def load_combine_anthro(antro_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not antro_dir.exists():
        return pd.DataFrame(columns=["draft_year", "name_key"])
    for p in sorted(antro_dir.glob("Draft_antro_*.csv")):
        m = re.search(r"Draft_antro_(\d{4})\.csv$", p.name)
        if not m:
            continue
        draft_year = int(m.group(1))
        d = pd.read_csv(p, low_memory=False)
        if "PLAYER_NAME" not in d.columns:
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
        cols = [c for c in keep if c in d.columns]
        if not cols:
            continue
        out = d[cols].rename(columns={c: keep[c] for c in cols}).copy()
        out["draft_year"] = draft_year
        out["name_key"] = out["combine_player_name"].map(_name_key)
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["draft_year", "name_key"])

    comb = pd.concat(rows, ignore_index=True)
    _safe_num(
        comb,
        [
            "combine_height_wo_shoes_in",
            "combine_weight_lb",
            "combine_wingspan_in",
            "combine_standing_reach_in",
            "combine_body_fat_pct",
            "combine_hand_length_in",
            "combine_hand_width_in",
        ],
    )

    # Keep only unique name+year rows to avoid accidental same-name mismatches.
    g = comb.groupby(["draft_year", "name_key"], dropna=False).size().rename("n").reset_index()
    uniq = g[g["n"] == 1][["draft_year", "name_key"]]
    comb = comb.merge(uniq, how="inner", on=["draft_year", "name_key"])
    comb = comb.drop_duplicates(subset=["draft_year", "name_key"], keep="first").copy()
    comb["combine_wingspan_minus_height"] = comb["combine_wingspan_in"] - comb["combine_height_wo_shoes_in"]
    comb["combine_standing_reach_minus_height"] = (
        comb["combine_standing_reach_in"] - comb["combine_height_wo_shoes_in"]
    )
    return comb


def load_rapm(rapm_csv: Path) -> pd.DataFrame:
    if not rapm_csv.exists():
        raise SystemExit(f"Missing RAPM CSV: {rapm_csv}")
    r = pd.read_csv(rapm_csv, low_memory=False)
    need = {"player_name", "season_end_year", "rapm", "possessions"}
    missing = [c for c in sorted(need) if c not in r.columns]
    if missing:
        raise SystemExit(f"RAPM CSV missing required columns: {missing}")
    r = r.copy()
    _safe_num(r, ["season_end_year", "rapm", "possessions"])
    r["name_key"] = r["player_name"].map(_name_key)
    r = r[r["name_key"] != ""].copy()

    # If a player changed teams, merge season rows by possession-weighted RAPM.
    r["rapm_x_poss"] = r["rapm"] * r["possessions"]
    g = (
        r.groupby(["name_key", "season_end_year"], dropna=False)
        .agg(
            rapm_x_poss=("rapm_x_poss", "sum"),
            possessions=("possessions", "sum"),
            rapm_mean=("rapm", "mean"),
        )
        .reset_index()
    )
    g["rapm"] = np.where(g["possessions"] > 1e-9, g["rapm_x_poss"] / g["possessions"], g["rapm_mean"])
    out = g[["name_key", "season_end_year", "rapm", "possessions"]].dropna(subset=["rapm", "season_end_year"])
    out["season_end_year"] = out["season_end_year"].astype(int)
    return out


def load_darko_proxy_rapm(darko_csv: Path, rapm_anchor_csv: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    if not darko_csv.exists():
        raise SystemExit(f"Missing DARKO CSV: {darko_csv}")
    d = pd.read_csv(darko_csv, low_memory=False)
    need = {"player_name", "season", "dpm"}
    missing = [c for c in sorted(need) if c not in d.columns]
    if missing:
        raise SystemExit(f"DARKO CSV missing required columns: {missing}")

    d = d.copy()
    _safe_num(d, ["season", "dpm"])
    d["name_key"] = d["player_name"].map(_name_key)
    d = d[(d["name_key"] != "") & d["season"].notna() & d["dpm"].notna()].copy()
    d["season_end_year"] = d["season"].astype(int)

    # Team-changed seasons are averaged at player-season level.
    d = d.groupby(["name_key", "season_end_year"], dropna=False, as_index=False)["dpm"].mean()

    a = 0.0
    b = 1.0
    overlap_n = 0
    overlap_corr = float("nan")

    if rapm_anchor_csv.exists():
        rapm_anchor = load_rapm(rapm_anchor_csv)
        ov = d.merge(
            rapm_anchor[["name_key", "season_end_year", "rapm"]],
            how="inner",
            on=["name_key", "season_end_year"],
        )
        ov = ov.dropna(subset=["dpm", "rapm"]).copy()
        overlap_n = int(len(ov))
        if overlap_n >= 30:
            y_true = pd.to_numeric(ov["rapm"], errors="coerce").to_numpy(dtype=float)
            x_pred = pd.to_numeric(ov["dpm"], errors="coerce").to_numpy(dtype=float)
            a, b = fit_linear_calibration(y_true=y_true, y_pred=x_pred)
            c = pd.Series(x_pred).corr(pd.Series(y_true))
            overlap_corr = float(c) if pd.notna(c) else float("nan")

    out = d[["name_key", "season_end_year"]].copy()
    out["rapm"] = float(a) + float(b) * pd.to_numeric(d["dpm"], errors="coerce")
    # Dummy positive weight for compatibility with RAPM loader interface.
    out["possessions"] = 1.0
    meta = {
        "calib_intercept": float(a),
        "calib_slope": float(b),
        "overlap_n": float(overlap_n),
        "overlap_corr_dpm_vs_rapm": float(overlap_corr) if np.isfinite(overlap_corr) else float("nan"),
    }
    return out, meta


def _load_season_metric(path: Path, name_col: str, year_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["name_key", "season_end_year", out_col])
    df = pd.read_csv(path, low_memory=False)
    need = {name_col, year_col, value_col}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(columns=["name_key", "season_end_year", out_col])
    out = df[[name_col, year_col, value_col]].copy()
    out[name_col] = out[name_col].astype(str)
    out["name_key"] = out[name_col].map(_name_key)
    out["season_end_year"] = pd.to_numeric(out[year_col], errors="coerce")
    out[out_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out[(out["name_key"] != "") & out["season_end_year"].notna() & out[out_col].notna()].copy()
    if out.empty:
        return pd.DataFrame(columns=["name_key", "season_end_year", out_col])
    out["season_end_year"] = out["season_end_year"].astype(int)
    out = (
        out.groupby(["name_key", "season_end_year"], dropna=False, as_index=False)[out_col]
        .mean()
        .reset_index(drop=True)
    )
    return out[["name_key", "season_end_year", out_col]]


def _season_zscore(series: pd.Series, seasons: pd.Series) -> pd.Series:
    out = pd.Series([np.nan] * len(series), index=series.index, dtype=float)
    x = pd.to_numeric(series, errors="coerce")
    y = pd.to_numeric(seasons, errors="coerce")
    for season, idx in y.dropna().astype(int).groupby(y.dropna().astype(int)).groups.items():
        loc = list(idx)
        s = x.loc[loc]
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-12:
            out.loc[loc] = 0.0
        else:
            out.loc[loc] = (s - mu) / sd
    return out


def load_multi_metric_proxy_rapm(
    *,
    darko_csv: Path,
    lebron_csv: Path,
    mamba_csv: Path,
    raptor_csv: Path,
    bref_csv: Path,
    rapm_anchor_csv: Path,
    min_metrics: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    parts: list[pd.DataFrame] = []
    parts.append(_load_season_metric(darko_csv, "player_name", "season", "dpm", "darko_dpm"))
    parts.append(_load_season_metric(lebron_csv, "player_name", "year", "LEBRON", "lebron"))
    parts.append(_load_season_metric(mamba_csv, "player_name", "year", "MAMBA", "mamba"))
    parts.append(_load_season_metric(raptor_csv, "player_name", "season", "raptor_total", "raptor"))
    parts.append(_load_season_metric(bref_csv, "Player", "year", "BPM", "bref_bpm"))
    parts.append(_load_season_metric(bref_csv, "Player", "year", "WS/48", "bref_ws48"))

    base = None
    for p in parts:
        if p.empty:
            continue
        if base is None:
            base = p.copy()
        else:
            base = base.merge(p, how="outer", on=["name_key", "season_end_year"])
    if base is None or base.empty:
        raise SystemExit("No metric rows available for multi-metric proxy.")

    metric_cols = [c for c in ["darko_dpm", "lebron", "mamba", "raptor", "bref_bpm", "bref_ws48"] if c in base.columns]
    if not metric_cols:
        raise SystemExit("No valid metric columns found for multi-metric proxy.")
    for c in metric_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
        base[f"{c}_z"] = _season_zscore(base[c], base["season_end_year"])
    z_cols = [f"{c}_z" for c in metric_cols if f"{c}_z" in base.columns]
    base["proxy_metric_n"] = base[z_cols].notna().sum(axis=1)
    base["proxy_z"] = base[z_cols].mean(axis=1, skipna=True)
    base = base[base["proxy_metric_n"] >= int(max(1, min_metrics))].copy()
    base = base.dropna(subset=["proxy_z"]).copy()
    if base.empty:
        raise SystemExit("No rows left after multi-metric proxy coverage filter.")

    a = 0.0
    b = 1.0
    overlap_n = 0
    overlap_corr = float("nan")
    if rapm_anchor_csv.exists():
        rapm_anchor = load_rapm(rapm_anchor_csv)
        ov = base.merge(
            rapm_anchor[["name_key", "season_end_year", "rapm"]],
            how="inner",
            on=["name_key", "season_end_year"],
        )
        ov = ov.dropna(subset=["proxy_z", "rapm"]).copy()
        overlap_n = int(len(ov))
        if overlap_n >= 30:
            y_true = pd.to_numeric(ov["rapm"], errors="coerce").to_numpy(dtype=float)
            x_pred = pd.to_numeric(ov["proxy_z"], errors="coerce").to_numpy(dtype=float)
            a, b = fit_linear_calibration(y_true=y_true, y_pred=x_pred)
            c = pd.Series(x_pred).corr(pd.Series(y_true))
            overlap_corr = float(c) if pd.notna(c) else float("nan")

    out = base[["name_key", "season_end_year"]].copy()
    out["rapm"] = float(a) + float(b) * pd.to_numeric(base["proxy_z"], errors="coerce")
    out["possessions"] = 1.0
    meta = {
        "calib_intercept": float(a),
        "calib_slope": float(b),
        "overlap_n": float(overlap_n),
        "overlap_corr_proxy_vs_rapm": float(overlap_corr) if np.isfinite(overlap_corr) else float("nan"),
        "proxy_min_metrics": float(min_metrics),
        "proxy_rows": float(len(base)),
    }
    for c in metric_cols:
        meta[f"coverage_rows_{c}"] = float(base[c].notna().sum())
    return out, meta


def build_targets(drafted: pd.DataFrame, rapm: pd.DataFrame, window_years: int) -> pd.DataFrame:
    vals: list[float] = []
    n_obs: list[int] = []
    for _, row in drafted.iterrows():
        y = int(row["season"])
        nk = row["name_key"]
        sub = rapm[(rapm["name_key"] == nk) & (rapm["season_end_year"] >= y + 1)].copy()
        if sub.empty:
            vals.append(np.nan)
            n_obs.append(0)
            continue
        top = sub.sort_values("rapm", ascending=False, kind="stable").head(window_years)
        vals.append(float(top["rapm"].mean()))
        n_obs.append(int(len(top)))
    t = drafted[["row_id"]].copy()
    t["rapm_best2_mean"] = vals
    t["rapm_n_obs"] = n_obs
    t["rapm_best2_mean_z"] = zscore(t["rapm_best2_mean"])
    t = t.rename(columns={"rapm_best2_mean_z": "nba_impact_target_z", "rapm_n_obs": "target_metrics_n"})
    return t


def choose_success_features(df: pd.DataFrame, model_df: pd.DataFrame, profile: str) -> tuple[list[str], list[str]]:
    base_num, base_cat = draft_model.choose_feature_columns(df)
    combine_num = [
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "combine_body_fat_pct",
        "combine_hand_length_in",
        "combine_hand_width_in",
        "combine_standing_reach_minus_height",
    ]
    combine_num.extend([c for c in model_df.columns if c.startswith("missing_combine_")])

    core_allow = {
        "minutes",
        "minutes_per_game",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "steals_per40",
        "blocks_per40",
        "turnovers_per40",
        "usage",
        "true_shooting_pct",
        "effective_field_goal_pct",
        "three_point_pct",
        "free_throw_pct",
        "net_rating",
        "offensive_rating",
        "defensive_rating",
        "win_shares_total_per40",
        "three_point_attempt_rate",
        "stocks_per40",
        "log_minutes",
        "assist_to_turnover",
        "scoring_load_x_usage",
        "playmaking_load_x_usage",
        "efficiency_x_usage",
        "rebounding_load_x_usage",
        "net_x_minutes",
        "age",
        "height_in",
        "weight_lb",
        "years_since_first_seen",
        "years_since_first_seen_sq",
        "is_first_year",
        "is_upperclass",
        "age_u20",
        "age_u22",
        "years_in_college",
        "team_win_pct",
        "team_avg_margin",
        "team_avg_elo_start",
        "team_avg_opp_elo_start",
        "team_avg_elo_delta",
        "team_conference_game_rate",
    }

    if profile == "full":
        num_cols = list(base_num)
    elif profile == "no_ext":
        num_cols = [c for c in base_num if not (c.startswith("ext_") or c.startswith("missing_ext_"))]
    else:
        num_cols = [c for c in base_num if c in core_allow]
        num_cols.extend(
            [
                c
                for c in model_df.columns
                if c.startswith("missing_")
                and (
                    "combine_" in c
                    or c
                    in {
                        "missing_age",
                        "missing_height_in",
                        "missing_weight_lb",
                        "missing_usage",
                        "missing_true_shooting_pct",
                        "missing_net_rating",
                    }
                )
            ]
        )

    for c in combine_num:
        if c in model_df.columns and c not in num_cols:
            num_cols.append(c)
    for c in model_df.columns:
        if c.startswith("best_") and c not in num_cols:
            num_cols.append(c)
    for c in model_df.columns:
        if (
            c.startswith("career_avg_")
            or c.startswith("career_trim1_avg_")
        ) and c not in num_cols:
            num_cols.append(c)

    tenure_penalty_cols = {
        "first_seen_season",
        "years_since_first_seen",
        "years_since_first_seen_sq",
        "years_in_college",
        "n_college_seasons_seen",
        "is_first_year",
        "is_upperclass",
    }
    num_cols = [c for c in num_cols if c not in tenure_penalty_cols]
    # Drop trend family (YOY deltas) and weak/noisy length-delta family.
    num_cols = [c for c in num_cols if not c.startswith("trend_")]
    # Keep peak and average profiles; drop explicit peak-minus-average deltas.
    num_cols = [c for c in num_cols if not c.startswith("peak_gap_")]
    # Drop weak/noisy delta-length feature family from model fitting.
    num_cols = [c for c in num_cols if "wingspan_minus_height" not in c]

    cat_cols = [c for c in base_cat if c in model_df.columns and (profile == "full" or c != "ext_match_type")]
    return num_cols, cat_cols


def add_best_college_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "name_key" not in out.columns:
        out["name_key"] = out.get("name", pd.Series([""] * len(out), index=out.index)).map(_name_key)
    out["season_num"] = pd.to_numeric(out.get("season"), errors="coerce")
    w = out.sort_values(["name_key", "season_num"], kind="stable").copy()

    max_metrics = [
        "minutes",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "steals_per40",
        "blocks_per40",
        "stocks_per40",
        "usage",
        "true_shooting_pct",
        "effective_field_goal_pct",
        "three_point_pct",
        "three_point_attempt_rate",
        "win_shares_total_per40",
        "net_rating",
        "offensive_rating",
        "defensive_rating",
        "assist_to_turnover",
        "scoring_load_x_usage",
        "playmaking_load_x_usage",
        "efficiency_x_usage",
    ]
    min_metrics = [
        "turnovers_per40",
    ]

    for c in max_metrics:
        if c in w.columns:
            s = pd.to_numeric(w[c], errors="coerce")
            w[f"best_{c}"] = s.groupby(w["name_key"], dropna=False).cummax()
    for c in min_metrics:
        if c in w.columns:
            s = pd.to_numeric(w[c], errors="coerce")
            w[f"best_{c}"] = s.groupby(w["name_key"], dropna=False).cummin()

    w = w.sort_index(kind="stable")
    return w


def add_multi_year_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "name_key" not in out.columns:
        out["name_key"] = out.get("name", pd.Series([""] * len(out), index=out.index)).map(_name_key)
    out["season_num"] = pd.to_numeric(out.get("season"), errors="coerce")
    w = out.sort_values(["name_key", "season_num"], kind="stable").copy()
    mins = pd.to_numeric(w.get("minutes"), errors="coerce").fillna(0.0)
    g = w["name_key"]

    metrics = [
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "steals_per40",
        "blocks_per40",
        "stocks_per40",
        "usage",
        "true_shooting_pct",
        "effective_field_goal_pct",
        "three_point_pct",
        "three_point_attempt_rate",
        "turnovers_per40",
        "net_rating",
        "offensive_rating",
        "defensive_rating",
        "assist_to_turnover",
    ]
    w["career_minutes_total"] = mins.groupby(g, dropna=False).cumsum()
    w["n_college_seasons_seen"] = w.groupby("name_key", dropna=False).cumcount() + 1

    for c in metrics:
        if c not in w.columns:
            continue
        s = pd.to_numeric(w[c], errors="coerce")
        num = (s.fillna(0.0) * mins).groupby(g, dropna=False).cumsum()
        den = mins.groupby(g, dropna=False).cumsum()
        w[f"career_avg_{c}"] = np.where(den > 1e-9, num / den, np.nan)
        first = s.groupby(g, dropna=False).transform("first")
        w[f"trend_{c}"] = s - first
        best_col = f"best_{c}"
        if best_col in w.columns:
            w[f"peak_gap_{c}"] = pd.to_numeric(w[best_col], errors="coerce") - pd.to_numeric(
                w[f"career_avg_{c}"], errors="coerce"
            )

    quality_hi = [
        "true_shooting_pct",
        "three_point_pct",
        "three_point_attempt_rate",
        "assist_to_turnover",
        "assists_per40",
        "stocks_per40",
        "rebounds_total_per40",
        "net_rating",
        "usage",
    ]
    quality_lo = [
        "turnovers_per40",
    ]
    trim_metrics = list(metrics)
    w["trim_worst_year_dropped"] = 0
    for _, idx in w.groupby("name_key", sort=False).groups.items():
        idxs = list(idx)
        idxs.sort(key=lambda i: (float(w.at[i, "season_num"]) if pd.notna(w.at[i, "season_num"]) else float("inf")))
        for j, ridx in enumerate(idxs):
            sub_idx = idxs[: j + 1]
            sub = w.loc[sub_idx].copy()
            drop_idx: int | None = None
            if len(sub) >= 3:
                quality_parts: list[pd.Series] = []
                for c in quality_hi:
                    if c not in sub.columns:
                        continue
                    s = pd.to_numeric(sub[c], errors="coerce")
                    if s.notna().sum() >= 2:
                        quality_parts.append(s.rank(pct=True, method="average"))
                for c in quality_lo:
                    if c not in sub.columns:
                        continue
                    s = pd.to_numeric(sub[c], errors="coerce")
                    if s.notna().sum() >= 2:
                        quality_parts.append((-s).rank(pct=True, method="average"))
                if quality_parts:
                    q = pd.concat(quality_parts, axis=1).mean(axis=1, skipna=True)
                    if q.notna().any():
                        drop_idx = int(q.idxmin())
            if drop_idx is not None:
                sub_eff = sub.drop(index=[drop_idx]).copy()
                w.at[ridx, "trim_worst_year_dropped"] = 1
            else:
                sub_eff = sub

            mins_eff = pd.to_numeric(sub_eff.get("minutes"), errors="coerce").fillna(0.0)
            for c in trim_metrics:
                if c not in sub_eff.columns:
                    continue
                s_eff = pd.to_numeric(sub_eff[c], errors="coerce")
                mask = s_eff.notna() & mins_eff.notna()
                den = float(mins_eff[mask].sum()) if mask.any() else 0.0
                val = float((s_eff[mask] * mins_eff[mask]).sum() / den) if den > 1e-9 else float("nan")
                w.at[ridx, f"career_trim1_avg_{c}"] = val

    return w.sort_index(kind="stable")


def apply_feature_weights(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    if X.size == 0 or not feature_names:
        return X
    weighted = X.copy()
    weight_rules = [
        ("three_point_attempt_rate", 1.60),
        ("three_point_pct", 1.50),
        ("true_shooting_pct", 1.35),
        ("effective_field_goal_pct", 1.25),
        ("assist_to_turnover", 1.30),
        ("assists_per40", 1.20),
        ("stocks_per40", 1.20),
        ("steals_per40", 1.15),
        ("blocks_per40", 1.15),
        ("usage", 1.10),
        ("turnovers_per40", 1.10),
        ("age", 1.25),
        ("combine_wingspan_in", 1.25),
        ("combine_standing_reach_in", 1.20),
        ("combine_height_wo_shoes_in", 1.15),
        ("best_", 1.20),
        ("career_avg_", 1.10),
        ("career_trim1_avg_", 1.22),
    ]
    for j, fname in enumerate(feature_names):
        w = 1.0
        for patt, mult in weight_rules:
            if patt in fname:
                w = max(w, mult)
        if w != 1.0:
            weighted[:, j] *= w
    return weighted


def main() -> None:
    args = parse_args()
    if not args.training_csv.exists():
        raise SystemExit(f"Missing training table: {args.training_csv}")

    raw = pd.read_csv(args.training_csv, low_memory=False)
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce").astype("Int64")
    raw["label_known"] = draft_model.as_bool_series(raw["label_known"])
    raw["drafted"] = pd.to_numeric(raw["drafted"], errors="coerce")
    raw["pick_number"] = pd.to_numeric(raw["pick_number"], errors="coerce")

    df = draft_model.add_engineered_features(raw)
    df["name_key"] = df.get("name", pd.Series([""] * len(df), index=df.index)).map(_name_key)
    df = add_best_college_features(df)
    df = add_multi_year_profile_features(df)
    df["draft_year"] = pd.to_numeric(df.get("season"), errors="coerce")
    df["years_in_college"] = pd.to_numeric(df.get("years_since_first_seen"), errors="coerce") + 1.0

    comb = load_combine_anthro(args.combine_dir)
    if not comb.empty:
        df = df.merge(
            comb,
            how="left",
            left_on=["draft_year", "name_key"],
            right_on=["draft_year", "name_key"],
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
            if c in df.columns:
                df[f"missing_{c}"] = pd.to_numeric(df[c], errors="coerce").isna().astype(int)

    # Entrant cohort: one row per player at their last observed college season.
    entrants = df[df["label_known"]].copy().reset_index(drop=True)
    entrants["season"] = pd.to_numeric(entrants.get("season"), errors="coerce")
    entrants["name_key"] = entrants.get("name", pd.Series([""] * len(entrants), index=entrants.index)).map(_name_key)
    entrants["minutes_num"] = pd.to_numeric(entrants.get("minutes"), errors="coerce").fillna(0.0)
    entrants["athlete_id"] = pd.to_numeric(entrants.get("athlete_id"), errors="coerce").astype("Int64")
    entrants["pick_number"] = pd.to_numeric(entrants.get("pick_number"), errors="coerce")

    # Keep one row per player-season (transfer-safe), then keep the player's final season.
    entrants = entrants.sort_values(["athlete_id", "season", "minutes_num"], ascending=[True, True, False], kind="stable")
    entrants = entrants.drop_duplicates(subset=["athlete_id", "season"], keep="first").copy()
    entrants["college_exit_year"] = entrants.groupby("athlete_id", dropna=False)["season"].transform("max")
    entrants = entrants[entrants["season"] == entrants["college_exit_year"]].copy()
    entrants = entrants.drop(columns=["minutes_num"]).reset_index(drop=True)
    entrants["row_id"] = np.arange(len(entrants), dtype=int)

    proxy_meta: dict[str, float] = {}
    if args.target_source == "multi_metric_proxy":
        rapm, proxy_meta = load_multi_metric_proxy_rapm(
            darko_csv=args.darko_csv,
            lebron_csv=args.lebron_csv,
            mamba_csv=args.mamba_csv,
            raptor_csv=args.raptor_csv,
            bref_csv=args.bref_csv,
            rapm_anchor_csv=args.rapm_csv,
            min_metrics=args.proxy_min_metrics,
        )
        target_source_desc = "Multi-metric proxy (DARKO+LEBRON+MAMBA+RAPTOR+BPM+WS/48, season-z averaged, calibrated)"
    elif args.target_source == "darko_dpm_proxy":
        rapm, proxy_meta = load_darko_proxy_rapm(args.darko_csv, args.rapm_csv)
        target_source_desc = f"DARKO dpm proxy (calibrated) from {args.darko_csv}"
    else:
        rapm = load_rapm(args.rapm_csv)
        target_source_desc = f"Historical RAPM from {args.rapm_csv}"
    target = build_targets(entrants, rapm, window_years=args.target_window_years)
    entrants = entrants.merge(target, how="left", on="row_id")

    model_df = entrants[entrants["target_metrics_n"] >= args.target_window_years].copy()
    if args.model_start_year is not None:
        model_df = model_df[pd.to_numeric(model_df["college_exit_year"], errors="coerce") >= float(args.model_start_year)].copy()
    if args.model_end_year is not None:
        model_df = model_df[pd.to_numeric(model_df["college_exit_year"], errors="coerce") <= float(args.model_end_year)].copy()
    if model_df.empty:
        raise SystemExit("No entrant rows with enough RAPM history in the chosen model-year window.")

    num_cols, cat_cols = choose_success_features(df=df, model_df=model_df, profile=args.feature_profile)
    prep = draft_model.FeaturePrep(num_cols, cat_cols).fit(model_df)
    season_vals = pd.to_numeric(model_df["college_exit_year"], errors="coerce")
    avail_years = sorted(season_vals.dropna().astype(int).unique().tolist())
    if len(avail_years) < 2:
        raise SystemExit("Need at least two exit years with target coverage to run holdout model selection.")

    preferred_holdout = int(args.test_draft_year)

    def _split_for_year(y: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr = model_df[season_vals < float(y)].copy()
        te = model_df[season_vals == float(y)].copy()
        return tr, te

    holdout_year: int | None = None
    if preferred_holdout in avail_years:
        cand_train, cand_test = _split_for_year(preferred_holdout)
        if (not cand_train.empty) and (not cand_test.empty):
            holdout_year = preferred_holdout
    if holdout_year is None:
        for y in reversed(avail_years):
            cand_train, cand_test = _split_for_year(y)
            if (not cand_train.empty) and (not cand_test.empty):
                holdout_year = y
                break
    if holdout_year is None:
        raise SystemExit("Could not find a valid holdout year with non-empty train and test splits.")
    if holdout_year != preferred_holdout:
        print(f"[info] Preferred holdout year {preferred_holdout} unavailable in cohort; using {holdout_year} instead.")

    train_df, test_df = _split_for_year(holdout_year)

    X_train = prep.transform(train_df)
    X_train = apply_feature_weights(X_train, prep.feature_names)
    y_train = pd.to_numeric(train_df["rapm_best2_mean"], errors="coerce").to_numpy(dtype=float)
    X_test = prep.transform(test_df)
    X_test = apply_feature_weights(X_test, prep.feature_names)
    y_test = pd.to_numeric(test_df["rapm_best2_mean"], errors="coerce").to_numpy(dtype=float)
    rapm_mu = float(pd.to_numeric(model_df["rapm_best2_mean"], errors="coerce").mean())
    rapm_sd = float(pd.to_numeric(model_df["rapm_best2_mean"], errors="coerce").std(ddof=0))
    if (not np.isfinite(rapm_sd)) or (rapm_sd <= 1e-12):
        rapm_sd = 1.0
    y_test_rapm = y_test.copy()

    models = {
        "ridge_a20": Ridge(alpha=20.0, random_state=args.seed),
        "ridge_a80": Ridge(alpha=80.0, random_state=args.seed),
        "ridge_a200": Ridge(alpha=200.0, random_state=args.seed),
        "ridge_a400": Ridge(alpha=400.0, random_state=args.seed),
        "enet_a005_l15": ElasticNet(alpha=0.05, l1_ratio=0.15, max_iter=20000, random_state=args.seed),
        "enet_a01_l20": ElasticNet(alpha=0.10, l1_ratio=0.20, max_iter=20000, random_state=args.seed),
        "rf_d8_l3": RandomForestRegressor(
            n_estimators=700,
            max_depth=8,
            min_samples_leaf=3,
            random_state=args.seed,
            n_jobs=-1,
        ),
        "rf_d12_l2": RandomForestRegressor(
            n_estimators=1000,
            max_depth=12,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        ),
        "extra_d10_l2": ExtraTreesRegressor(
            n_estimators=1000,
            max_depth=10,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        ),
        "hgb_03_d3": HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=3,
            max_iter=700,
            l2_regularization=0.2,
            random_state=args.seed,
        ),
        "hgb_05_d4": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=500,
            l2_regularization=0.3,
            random_state=args.seed,
        ),
        "gbr_03_d2": GradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=2,
            n_estimators=1000,
            subsample=0.8,
            random_state=args.seed,
        ),
    }

    rows: list[dict[str, object]] = []
    fit_models: dict[str, object] = {}
    test_preds: dict[str, np.ndarray] = {}
    calib_map: dict[str, tuple[float, float]] = {}
    y_test_rank = pd.Series(y_test_rapm).rank(method="min", ascending=False)
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        p_train = mdl.predict(X_train)
        p = mdl.predict(X_test)
        if args.disable_linear_calibration:
            a, b = 0.0, 1.0
            p_cal = p
        else:
            a, b = fit_linear_calibration(y_train, p_train)
            p_cal = a + b * p
        p_cal_rapm = p_cal
        p_rank = pd.Series(p_cal_rapm).rank(method="min", ascending=False)
        rank_err = p_rank - y_test_rank
        abs_err = np.abs(p_cal_rapm - y_test_rapm)
        test_preds[name] = p_cal_rapm
        calib_map[name] = (a, b)
        rows.append(
            {
                "model": name,
                "rmse": rmse(y_test_rapm, p_cal_rapm),
                "mae": mae(y_test_rapm, p_cal_rapm),
                "spearman": spearman_corr(y_test_rapm, p_cal_rapm),
                "rapm_rmse": rmse(y_test_rapm, p_cal_rapm),
                "rapm_mae": mae(y_test_rapm, p_cal_rapm),
                "hit075": int((abs_err <= 0.75).sum()),
                "hit100": int((abs_err <= 1.0).sum()),
                "hit075_rate": float((abs_err <= 0.75).mean()),
                "hit100_rate": float((abs_err <= 1.0).mean()),
                "mae_rank": float(np.mean(np.abs(rank_err))),
                "within5": int((np.abs(rank_err) <= 5).sum()),
                "within10": int((np.abs(rank_err) <= 10).sum()),
                "calib_intercept": float(a),
                "calib_slope": float(b),
                "n_test": int(len(test_df)),
            }
        )
        fit_models[name] = mdl

    metrics_df = pd.DataFrame(rows)
    if args.selection_objective == "hit075":
        metrics_df = metrics_df.sort_values(
            ["hit075_rate", "rapm_mae", "rapm_rmse"], ascending=[False, True, True], kind="stable"
        )
    elif args.selection_objective == "rapm_mae":
        metrics_df = metrics_df.sort_values(
            ["rapm_mae", "rapm_rmse", "hit075_rate"], ascending=[True, True, False], kind="stable"
        )
    elif args.selection_objective == "rapm_rmse":
        metrics_df = metrics_df.sort_values(
            ["rapm_rmse", "rapm_mae", "hit075_rate"], ascending=[True, True, False], kind="stable"
        )
    elif args.selection_objective == "spearman":
        metrics_df = metrics_df.sort_values(
            ["spearman", "rapm_rmse", "hit075_rate"], ascending=[False, True, False], kind="stable"
        )
    else:
        metrics_df = metrics_df.sort_values(
            ["mae_rank", "rapm_rmse", "hit075_rate"], ascending=[True, True, False], kind="stable"
        )
    metrics_df = metrics_df.reset_index(drop=True)
    forced_model = str(args.force_model or "").strip()
    if forced_model:
        if forced_model not in models:
            raise SystemExit(f"--force-model '{forced_model}' is not a known model key.")
        best_name = forced_model
    else:
        best_name = str(metrics_df.iloc[0]["model"])
    best_test_pred = test_preds[best_name]

    holdout = test_df.copy()
    holdout["actual_rapm_best2_mean"] = pd.to_numeric(holdout["rapm_best2_mean"], errors="coerce")
    holdout["pred_rapm_best2_mean"] = best_test_pred
    holdout["actual_impact_z"] = (holdout["actual_rapm_best2_mean"] - rapm_mu) / rapm_sd
    holdout["pred_impact_z"] = (holdout["pred_rapm_best2_mean"] - rapm_mu) / rapm_sd
    holdout["actual_rank"] = holdout["actual_rapm_best2_mean"].rank(method="min", ascending=False).astype(int)
    holdout["pred_rank"] = holdout["pred_rapm_best2_mean"].rank(method="min", ascending=False).astype(int)
    holdout["rank_error"] = holdout["pred_rank"] - holdout["actual_rank"]
    holdout["abs_rank_error"] = holdout["rank_error"].abs()
    holdout = holdout.sort_values("actual_rank", kind="stable")

    final_prep = draft_model.FeaturePrep(num_cols, cat_cols).fit(model_df)
    final_model = models[best_name]
    X_all = final_prep.transform(model_df)
    X_all = apply_feature_weights(X_all, final_prep.feature_names)
    y_all = pd.to_numeric(model_df["rapm_best2_mean"], errors="coerce").to_numpy(dtype=float)
    final_model.fit(X_all, y_all)

    pred_year_start = args.predict_start_year
    pred_year_end = args.predict_end_year
    if pred_year_start is None and pred_year_end is not None:
        pred_year_start = pred_year_end
    if pred_year_end is None and pred_year_start is not None:
        pred_year_end = pred_year_start

    if pred_year_start is not None and pred_year_end is not None:
        if pred_year_end < pred_year_start:
            raise SystemExit("--predict-end-year cannot be smaller than --predict-start-year.")
        pred_mask = pd.to_numeric(entrants["college_exit_year"], errors="coerce").between(
            float(pred_year_start), float(pred_year_end), inclusive="both"
        )
        pred_df = entrants[pred_mask].copy()
        pred_scope_label = f"exit years {pred_year_start}-{pred_year_end}"
        pred_out_name = f"nba_success_rapm_predictions_exit_{pred_year_start}_{pred_year_end}.csv"
    else:
        pred_year = int(args.predict_draft_year)
        pred_df = entrants[pd.to_numeric(entrants["college_exit_year"], errors="coerce") == float(pred_year)].copy()
        pred_scope_label = f"exit year {pred_year}"
        pred_out_name = f"nba_success_rapm_predictions_draft_{pred_year}.csv"
    if pred_df.empty:
        raise SystemExit(f"No entrant rows found for prediction scope: {pred_scope_label}.")

    X_pred = final_prep.transform(pred_df)
    X_pred = apply_feature_weights(X_pred, final_prep.feature_names)
    pred_df["pred_rapm_best2_raw"] = final_model.predict(X_pred)
    if args.disable_linear_calibration:
        a_all, b_all = 0.0, 1.0
    else:
        pred_all_raw = final_model.predict(X_all)
        a_all, b_all = fit_linear_calibration(y_all, pred_all_raw)
    pred_df["pred_rapm_best2_mean"] = a_all + b_all * pred_df["pred_rapm_best2_raw"]
    pred_df["pred_nba_impact_z"] = (pred_df["pred_rapm_best2_mean"] - rapm_mu) / rapm_sd
    pred_df["actual_rapm_best2_mean"] = pd.to_numeric(pred_df["rapm_best2_mean"], errors="coerce")
    pred_df["actual_nba_impact_z"] = (pred_df["actual_rapm_best2_mean"] - rapm_mu) / rapm_sd
    pred_df = pred_df.sort_values("pred_rapm_best2_mean", ascending=False, kind="stable")
    pred_df["pred_impact_rank"] = np.arange(1, len(pred_df) + 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets_out = args.out_dir / "nba_success_rapm_targets_best2.csv"
    metrics_out = args.out_dir / "nba_success_rapm_model_metrics.csv"
    pred_out = args.out_dir / pred_out_name
    holdout_out = args.out_dir / f"nba_success_rapm_holdout_actual_vs_predicted_{holdout_year}.csv"
    report_out = args.out_dir / "nba_success_rapm_model_report.txt"

    entrants.to_csv(targets_out, index=False)
    metrics_df.to_csv(metrics_out, index=False)

    pred_keep = [
        "season",
        "college_exit_year",
        "name",
        "team",
        "conference",
        "pick_number",
        "minutes",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "usage",
        "true_shooting_pct",
        "net_rating",
        "years_since_first_seen",
        "years_in_college",
        "is_first_year",
        "is_upperclass",
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "target_metrics_n",
        "actual_nba_impact_z",
        "actual_rapm_best2_mean",
        "pred_nba_impact_z",
        "pred_rapm_best2_mean",
        "pred_impact_rank",
    ]
    pred_keep = [c for c in pred_keep if c in pred_df.columns]
    pred_df[pred_keep + [c for c in pred_df.columns if c not in pred_keep]].to_csv(pred_out, index=False)

    hold_keep = [
        "season",
        "name",
        "team",
        "conference",
        "pick_number",
        "minutes",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "usage",
        "true_shooting_pct",
        "net_rating",
        "years_since_first_seen",
        "years_in_college",
        "is_first_year",
        "is_upperclass",
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "target_metrics_n",
        "actual_rapm_best2_mean",
        "pred_rapm_best2_mean",
        "actual_impact_z",
        "pred_impact_z",
        "actual_rank",
        "pred_rank",
        "rank_error",
        "abs_rank_error",
    ]
    hold_keep = [c for c in hold_keep if c in holdout.columns] + [c for c in holdout.columns if c not in hold_keep]
    holdout[hold_keep].to_csv(holdout_out, index=False)

    lines: list[str] = []
    lines.append("NBA Success Model (College -> Best 2-Year NBA RAPM Peak)")
    lines.append(f"Target source: {target_source_desc}")
    if args.target_source in {"darko_dpm_proxy", "multi_metric_proxy"}:
        lines.append(f"RAPM anchor for calibration: {args.rapm_csv}")
        if args.target_source == "darko_dpm_proxy":
            lines.append(
                "DARKO->RAPM calibration: "
                f"intercept={proxy_meta.get('calib_intercept', float('nan')):.4f}, "
                f"slope={proxy_meta.get('calib_slope', float('nan')):.4f}, "
                f"overlap_n={int(proxy_meta.get('overlap_n', 0.0)):,}, "
                f"overlap_corr={proxy_meta.get('overlap_corr_dpm_vs_rapm', float('nan')):.4f}"
            )
        else:
            lines.append(
                "Proxy->RAPM calibration: "
                f"intercept={proxy_meta.get('calib_intercept', float('nan')):.4f}, "
                f"slope={proxy_meta.get('calib_slope', float('nan')):.4f}, "
                f"overlap_n={int(proxy_meta.get('overlap_n', 0.0)):,}, "
                f"overlap_corr={proxy_meta.get('overlap_corr_proxy_vs_rapm', float('nan')):.4f}"
            )
            lines.append(
                f"Multi-metric proxy rows={int(proxy_meta.get('proxy_rows', 0.0)):,}, "
                f"min_metrics_per_player_season={int(proxy_meta.get('proxy_min_metrics', 0.0))}"
            )
    lines.append(f"RAPM target: mean of top {args.target_window_years} post-draft RAPM seasons")
    lines.append(f"Combine source dir: {args.combine_dir}")
    if not comb.empty:
        lines.append(f"Combine unique rows merged: {len(comb):,}")
    else:
        lines.append("Combine unique rows merged: 0")
    lines.append("")
    if (args.model_start_year is not None) or (args.model_end_year is not None):
        lo = args.model_start_year if args.model_start_year is not None else min(avail_years)
        hi = args.model_end_year if args.model_end_year is not None else max(avail_years)
        lines.append(f"Model cohort exit years: {lo}..{hi}")
    else:
        lines.append("Model cohort exit years: all available")
    lines.append(f"Target-known entrant rows used for model: {len(model_df):,}")
    lines.append(f"Holdout exit year: {holdout_year} (n={len(test_df):,})")
    lines.append(f"Feature profile: {args.feature_profile}")
    lines.append(f"Selection objective: {args.selection_objective}")
    lines.append(f"Linear calibration: {'off' if args.disable_linear_calibration else 'on'}")
    lines.append("")
    lines.append("Holdout metrics by model:")
    for _, r in metrics_df.iterrows():
        lines.append(
            f"- {r['model']}: RAPM_RMSE={float(r['rapm_rmse']):.4f}, RAPM_MAE={float(r['rapm_mae']):.4f}, "
            f"Hit<=0.75={int(r['hit075'])}/{int(r['n_test'])} ({100.0*float(r['hit075_rate']):.1f}%), "
            f"Hit<=1.0={int(r['hit100'])}/{int(r['n_test'])} ({100.0*float(r['hit100_rate']):.1f}%), "
            f"Spearman={float(r['spearman']):.4f}, MAE_rank={float(r['mae_rank']):.2f}"
        )
    lines.append("")
    lines.append(f"Selected final model: {best_name}")
    hold_abs_rapm_err = (holdout["pred_rapm_best2_mean"] - holdout["actual_rapm_best2_mean"]).abs()
    lines.append(
        f"Holdout RAPM quality ({holdout_year}, selected model): "
        f"MAE={float(hold_abs_rapm_err.mean()):.4f}, "
        f"RMSE={float(np.sqrt(np.mean((holdout['pred_rapm_best2_mean'] - holdout['actual_rapm_best2_mean']) ** 2))):.4f}, "
        f"Hit<=0.75={(hold_abs_rapm_err <= 0.75).sum():,}/{len(holdout):,}, "
        f"Hit<=1.0={(hold_abs_rapm_err <= 1.0).sum():,}/{len(holdout):,}"
    )
    lines.append(
        f"Holdout rank quality ({holdout_year}, selected model): "
        f"MAE_rank={float(np.mean(np.abs(holdout['rank_error']))):.2f}, "
        f"within5={(holdout['abs_rank_error'] <= 5).sum():,}/{len(holdout):,}, "
        f"within10={(holdout['abs_rank_error'] <= 10).sum():,}/{len(holdout):,}"
    )
    lines.append(f"Scored entrant cohort: {pred_scope_label} (n={len(pred_df):,})")
    lines.append("Top 20 predicted NBA success (best-2 RAPM):")
    top = pred_df[
        ["name", "team", "pick_number", "pred_rapm_best2_mean", "pred_nba_impact_z", "pred_impact_rank"]
    ].head(20)
    for _, r in top.iterrows():
        pick = "n/a" if pd.isna(r.get("pick_number")) else int(float(r["pick_number"]))
        lines.append(
            f"- #{int(r['pred_impact_rank'])}: {r['name']} ({r['team']}), "
            f"pred_rapm_best2={float(r['pred_rapm_best2_mean']):.3f}, pred_z={float(r['pred_nba_impact_z']):.3f}, draft_pick={pick}"
        )

    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("")
    print(f"Wrote targets: {targets_out}")
    print(f"Wrote model metrics: {metrics_out}")
    print(f"Wrote predictions: {pred_out}")
    print(f"Wrote holdout detail: {holdout_out}")
    print(f"Wrote report: {report_out}")


if __name__ == "__main__":
    main()
