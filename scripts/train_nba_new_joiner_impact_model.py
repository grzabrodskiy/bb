from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_nba_draft_predictors as draft_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a college-to-NBA impact model for new joiners (RAPM-family proxy target)."
    )
    p.add_argument(
        "--training-csv",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
    )
    p.add_argument(
        "--rapm-csv",
        type=Path,
        default=Path("data/raw/external/nbarapm/TimedecayRAPM.csv"),
        help="Optional RAPM table used for holdout-aligned model selection.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
    )
    p.add_argument(
        "--target-window-years",
        type=int,
        default=2,
        help="Use top N best NBA seasons after draft when constructing target metrics.",
    )
    p.add_argument(
        "--target-season-mode",
        type=str,
        default="best",
        choices=["best", "first"],
        help="How to pick seasons after draft for each metric before averaging.",
    )
    p.add_argument(
        "--min-target-metrics",
        type=int,
        default=2,
        help="Minimum number of RAPM-family metrics required to keep a training target row.",
    )
    p.add_argument(
        "--test-draft-year",
        type=int,
        default=2022,
        help="Holdout draft year for offline model quality check.",
    )
    p.add_argument(
        "--predict-draft-year",
        type=int,
        default=2025,
        help="Draft year to score for new joiners.",
    )
    p.add_argument(
        "--combine-dir",
        type=Path,
        default=Path("data/raw/external/nba_stats_draft/antro/antro"),
        help="Optional NBA combine measurements directory (Draft_antro_YYYY.csv files).",
    )
    p.add_argument(
        "--crafted-measurements-csv",
        type=Path,
        default=Path("data/raw/external/craftednba/player_traits_length.csv"),
        help="Optional CraftedNBA measurements CSV (height/wingspan fallback).",
    )
    p.add_argument(
        "--selection-objective",
        type=str,
        default="hit_rank1",
        choices=["hit_rank1", "hit_rank5", "rank_mae", "spearman", "rmse", "rapm_mae", "rapm_rmse", "rapm_hit100"],
        help="Metric used to pick final model from holdout candidates.",
    )
    p.add_argument(
        "--feature-profile",
        type=str,
        default="full",
        choices=["core", "no_ext", "full"],
        help="Feature profile: 'core' keeps robust signals, 'no_ext' drops external db columns, 'full' keeps all.",
    )
    p.add_argument(
        "--timedecay-in-target",
        action="store_true",
        help=(
            "Include current Timedecay RAPM as an additional target metric component. "
            "Useful for a peak-recency objective alongside best-2 seasonal metrics."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def load_metric(
    path: Path,
    name_col: str,
    year_col: str | None,
    value_col: str,
    metric_name: str,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["name_key", "season_end_year", metric_name])
    df = pd.read_csv(path, low_memory=False)
    keep = [c for c in [name_col, value_col] if c in df.columns]
    if year_col and year_col in df.columns:
        keep.append(year_col)
    if len(keep) < 3:
        if len(keep) < 2:
            return pd.DataFrame(columns=["name_key", "season_end_year", metric_name])
    df = df[keep].copy()
    df[name_col] = df[name_col].astype(str)
    if year_col and year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        year_valid = df[year_col].notna()
        df = df[year_valid].copy()
        if df.empty:
            return pd.DataFrame(columns=["name_key", "season_end_year", metric_name])
        df["season_end_year"] = df[year_col].astype(int)
    else:
        # Timedecay/current-type metrics do not carry per-season values.
        # Assign a far-future season so they qualify as post-draft signal.
        df["season_end_year"] = 9999
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=["name_key", "season_end_year", metric_name])
    df["name_key"] = df[name_col].map(normalize_text)
    out = (
        df.groupby(["name_key", "season_end_year"], dropna=False)[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: metric_name})
    )
    return out


def aggregate_metric_window(
    base: pd.DataFrame,
    metric_df: pd.DataFrame,
    metric_name: str,
    window_years: int,
    season_mode: str,
) -> pd.DataFrame:
    vals: list[float] = []
    n_obs: list[int] = []
    if metric_df.empty:
        out = base[["row_id"]].copy()
        out[f"{metric_name}_{season_mode}{window_years}"] = np.nan
        out[f"{metric_name}_n_obs"] = 0
        return out

    for _, r in base.iterrows():
        y = int(r["season"])
        sub = metric_df[
            (metric_df["name_key"] == r["name_key"]) & (metric_df["season_end_year"] >= y + 1)
        ].copy()
        if sub.empty:
            vals.append(np.nan)
            n_obs.append(0)
        else:
            if season_mode == "first":
                picked = sub.sort_values("season_end_year", kind="stable").head(window_years)
            else:
                picked = sub.sort_values(metric_name, ascending=False, kind="stable").head(window_years)
            vals.append(float(picked[metric_name].mean()))
            n_obs.append(int(len(picked)))

    out = base[["row_id"]].copy()
    out[f"{metric_name}_{season_mode}{window_years}"] = vals
    out[f"{metric_name}_n_obs"] = n_obs
    return out


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
        out["name_key"] = out["combine_player_name"].map(normalize_text)
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["draft_year", "name_key"])

    comb = pd.concat(rows, ignore_index=True)
    for c in [
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "combine_body_fat_pct",
        "combine_hand_length_in",
        "combine_hand_width_in",
    ]:
        if c in comb.columns:
            comb[c] = pd.to_numeric(comb[c], errors="coerce")

    g = comb.groupby(["draft_year", "name_key"], dropna=False).size().rename("n").reset_index()
    uniq = g[g["n"] == 1][["draft_year", "name_key"]]
    comb = comb.merge(uniq, how="inner", on=["draft_year", "name_key"])
    comb = comb.drop_duplicates(subset=["draft_year", "name_key"], keep="first").copy()
    comb["combine_wingspan_minus_height"] = comb["combine_wingspan_in"] - comb["combine_height_wo_shoes_in"]
    comb["combine_standing_reach_minus_height"] = (
        comb["combine_standing_reach_in"] - comb["combine_height_wo_shoes_in"]
    )
    return comb


def load_crafted_measurements(crafted_csv: Path) -> pd.DataFrame:
    if not crafted_csv.exists():
        return pd.DataFrame(columns=["name_key"])
    df = pd.read_csv(crafted_csv, low_memory=False)
    if "name_key" not in df.columns:
        if "player_name" in df.columns:
            df["name_key"] = df["player_name"].map(normalize_text)
        else:
            return pd.DataFrame(columns=["name_key"])
    keep = ["name_key"]
    for c in ["crafted_height_in", "crafted_wingspan_in", "crafted_length_in"]:
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


def coalesce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        out = out.where(out.notna(), x)
    return out


def add_best_college_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "name_key" not in out.columns:
        out["name_key"] = out.get("name", pd.Series([""] * len(out), index=out.index)).map(normalize_text)
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
        "combine_wingspan_minus_height",
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
    return w.sort_index(kind="stable")


def add_multi_year_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "name_key" not in out.columns:
        out["name_key"] = out.get("name", pd.Series([""] * len(out), index=out.index)).map(normalize_text)
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
        # Minutes-weighted cumulative average across all college years.
        num = (s.fillna(0.0) * mins).groupby(g, dropna=False).cumsum()
        den = mins.groupby(g, dropna=False).cumsum()
        w[f"career_avg_{c}"] = np.where(den > 1e-9, num / den, np.nan)

        # Development trend from first observed season to draft season.
        first = s.groupby(g, dropna=False).transform("first")
        w[f"trend_{c}"] = s - first

        # Peak premium over the multi-year average.
        best_col = f"best_{c}"
        if best_col in w.columns:
            w[f"peak_gap_{c}"] = pd.to_numeric(w[best_col], errors="coerce") - pd.to_numeric(
                w[f"career_avg_{c}"], errors="coerce"
            )

    # Robust multi-year profile: for players with >=3 seasons, drop the lowest-quality
    # season from cumulative profile averages to reduce one-off bad-year noise.
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
    trim_metrics = [
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

    w["trim_worst_year_dropped"] = 0
    for key, idx in w.groupby("name_key", sort=False).groups.items():
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
                if den > 1e-9:
                    val = float((s_eff[mask] * mins_eff[mask]).sum() / den)
                else:
                    val = float("nan")
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
        ("combine_wingspan_in", 1.25),
        ("combine_standing_reach_in", 1.20),
        ("combine_height_wo_shoes_in", 1.15),
        ("combine_wingspan_minus_height", 1.30),
        ("measurement_wingspan_in", 1.30),
        ("measurement_wingspan_minus_height", 1.35),
        ("measurement_standing_reach_in", 1.20),
        ("measurement_height_in", 1.15),
        ("best_", 1.20),
        ("career_avg_", 1.10),
        ("career_trim1_avg_", 1.22),
        ("peak_gap_", 1.15),
        ("trend_", 1.10),
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

    # Keep all features; do not limit to single-season stats only.
    df = draft_model.add_engineered_features(raw)
    df["name_key"] = df.get("name", pd.Series([""] * len(df), index=df.index)).map(normalize_text)
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

    crafted = load_crafted_measurements(args.crafted_measurements_csv)
    if not crafted.empty:
        df = df.merge(crafted, how="left", on=["name_key"])

    # Merge measurement fallbacks so physical tools are available even when combine fields are missing.
    df["measurement_height_in"] = coalesce_numeric(
        df,
        ["combine_height_wo_shoes_in", "height_in", "crafted_height_in", "ext_height_in_modeldb"],
    )
    df["measurement_weight_lb"] = coalesce_numeric(
        df,
        ["combine_weight_lb", "weight_lb", "ext_weight_lb_modeldb"],
    )
    df["measurement_wingspan_in"] = coalesce_numeric(
        df,
        ["combine_wingspan_in", "wingspan_in", "crafted_wingspan_in"],
    )
    df["measurement_standing_reach_in"] = coalesce_numeric(
        df,
        ["combine_standing_reach_in", "standing_reach_in"],
    )
    df["measurement_wingspan_minus_height"] = df["measurement_wingspan_in"] - df["measurement_height_in"]
    df["measurement_reach_minus_height"] = df["measurement_standing_reach_in"] - df["measurement_height_in"]
    for c in [
        "measurement_height_in",
        "measurement_weight_lb",
        "measurement_wingspan_in",
        "measurement_standing_reach_in",
        "measurement_wingspan_minus_height",
        "measurement_reach_minus_height",
    ]:
        df[f"missing_{c}"] = pd.to_numeric(df[c], errors="coerce").isna().astype(int)

    # Model/evaluation universe: all label-known players (drafted and undrafted) with measurable NBA outcomes.
    entrants = df[df["label_known"]].copy()
    entrants["source_row_id"] = entrants.index
    entrants = entrants.reset_index(drop=True)
    entrants["row_id"] = np.arange(len(entrants), dtype=int)
    entrants["name_key"] = entrants.get("name", pd.Series([""] * len(entrants), index=entrants.index)).map(normalize_text)
    entrants["minutes_num"] = pd.to_numeric(entrants.get("minutes"), errors="coerce").fillna(0.0)
    entrants["pick_number"] = pd.to_numeric(entrants.get("pick_number"), errors="coerce")
    entrants = entrants.sort_values(
        ["season", "name_key", "pick_number", "minutes_num"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    entrants = entrants.drop_duplicates(subset=["season", "name_key", "pick_number"], keep="first").copy()
    entrants = entrants.drop(columns=["minutes_num"]).reset_index(drop=True)
    entrants["row_id"] = np.arange(len(entrants), dtype=int)

    metric_defs = [
        ("lebron", Path("data/raw/external/nbarapm/lebron.csv"), "player_name", "year", "LEBRON"),
        ("darko_dpm", Path("data/raw/external/nbarapm/DARKO.csv"), "player_name", "season", "dpm"),
        ("mamba", Path("data/raw/external/nbarapm/mamba.csv"), "player_name", "year", "MAMBA"),
        ("raptor", Path("data/raw/external/nbarapm/raptor.csv"), "player_name", "season", "raptor_total"),
        ("bref_bpm", Path("data/raw/nba/bref/player_advanced_2010_2026.csv"), "Player", "year", "BPM"),
        ("bref_ws48", Path("data/raw/nba/bref/player_advanced_2010_2026.csv"), "Player", "year", "WS/48"),
    ]
    if args.timedecay_in_target:
        metric_defs.append(("timedecay_rapm", args.rapm_csv, "player_name", None, "rapm"))

    target = entrants[["row_id", "season", "name_key"]].copy()
    metric_cols: list[str] = []
    coverage_lines: list[str] = []
    for metric_name, path, name_col, year_col, value_col in metric_defs:
        metric_df = load_metric(path, name_col, year_col, value_col, metric_name)
        agg = aggregate_metric_window(
            base=entrants[["row_id", "season", "name_key"]],
            metric_df=metric_df,
            metric_name=metric_name,
            window_years=args.target_window_years,
            season_mode=args.target_season_mode,
        )
        col = f"{metric_name}_{args.target_season_mode}{args.target_window_years}"
        ncol = f"{metric_name}_n_obs"
        target = target.merge(agg, how="left", on=["row_id"])
        target[f"{col}_z"] = zscore(target[col])
        metric_cols.append(f"{col}_z")
        coverage_lines.append(
            f"{metric_name}: matched={(target[ncol] > 0).sum():,}/{len(target):,}, "
            f">=2seasons={(target[ncol] >= 2).sum():,}"
        )

    target["target_metrics_n"] = target[metric_cols].notna().sum(axis=1)
    target["nba_impact_target_z"] = target[metric_cols].mean(axis=1, skipna=True)
    entrants = entrants.merge(
        target[["row_id", "nba_impact_target_z", "target_metrics_n"] + metric_cols],
        how="left",
        on=["row_id"],
    )

    model_df = entrants[entrants["target_metrics_n"] >= args.min_target_metrics].copy()
    if model_df.empty:
        raise SystemExit("No rows with enough target metric coverage. Try --min-target-metrics 1.")

    base_num, cat_cols = draft_model.choose_feature_columns(df)
    if args.feature_profile == "core":
        # Keep only stable, observable basketball/bio signals and engineered counterparts.
        allow = {
            "season",
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
            "effective_field_goal_pct",
            "true_shooting_pct",
            "field_goal_pct",
            "two_point_pct",
            "three_point_pct",
            "free_throw_pct",
            "win_shares_total",
            "win_shares_total_per40",
            "win_shares_defensive",
            "win_shares_offensive",
            "height_in",
            "weight_lb",
            "age",
            "first_seen_season",
            "years_since_first_seen",
            "years_since_first_seen_sq",
            "years_in_college",
            "is_first_year",
            "is_upperclass",
            "log_minutes",
            "log_points",
            "log_assists",
            "log_rebounds",
            "assist_to_turnover",
            "size_bmi_proxy",
            "scoring_load_x_usage",
            "playmaking_load_x_usage",
            "rebounding_load_x_usage",
            "efficiency_x_usage",
            "net_x_minutes",
            "stocks_per40",
            "three_point_attempt_rate",
            "age_known",
            "age_u20",
            "age_u22",
            "is_high_tier",
            "is_mid_tier",
            "is_other_tier",
            "team_games",
            "team_win_pct",
            "team_avg_margin",
            "team_avg_elo_start",
            "team_avg_opp_elo_start",
            "team_avg_elo_delta",
            "team_conference_game_rate",
        }
        base_num = [c for c in base_num if c in allow or c.startswith("missing_") or c.endswith("_season_pct")]
        cat_cols = [c for c in cat_cols if c != "ext_match_type"]
    elif args.feature_profile == "no_ext":
        base_num = [c for c in base_num if not (c.startswith("ext_") or c.startswith("missing_ext_"))]
        cat_cols = [c for c in cat_cols if c != "ext_match_type"]
    extra_num = [
        "years_in_college",
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "combine_body_fat_pct",
        "combine_hand_length_in",
        "combine_hand_width_in",
        "combine_wingspan_minus_height",
        "combine_standing_reach_minus_height",
        "measurement_height_in",
        "measurement_weight_lb",
        "measurement_wingspan_in",
        "measurement_standing_reach_in",
        "measurement_wingspan_minus_height",
        "measurement_reach_minus_height",
        "crafted_height_in",
        "crafted_wingspan_in",
        "crafted_length_in",
        "crafted_wingspan_minus_height",
    ]
    extra_num.extend([c for c in model_df.columns if c.startswith("missing_combine_")])
    extra_num.extend([c for c in model_df.columns if c.startswith("missing_measurement_")])
    num_cols = base_num + [c for c in extra_num if c in model_df.columns and c not in base_num]
    num_cols += [c for c in model_df.columns if c.startswith("best_") and c not in num_cols]
    num_cols += [
        c
        for c in model_df.columns
        if (
            c.startswith("career_avg_")
            or c.startswith("career_trim1_avg_")
            or c.startswith("trend_")
            or c.startswith("peak_gap_")
        )
        and c not in num_cols
    ]

    # Tenure-neutral policy: do not let years-in-college/class-year directly penalize players.
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
    prep = draft_model.FeaturePrep(num_cols, cat_cols).fit(model_df)

    train_df = model_df[model_df["season"] < args.test_draft_year].copy()
    test_df = model_df[model_df["season"] == args.test_draft_year].copy()
    if train_df.empty:
        raise SystemExit("No train rows for chosen test year.")
    if test_df.empty:
        raise SystemExit("No test rows for chosen test year.")

    X_train = prep.transform(train_df)
    X_train = apply_feature_weights(X_train, prep.feature_names)
    y_train = train_df["nba_impact_target_z"].to_numpy(dtype=float)
    X_test = prep.transform(test_df)
    X_test = apply_feature_weights(X_test, prep.feature_names)
    y_test = test_df["nba_impact_target_z"].to_numpy(dtype=float)

    model_specs: list[tuple[str, object]] = [
        ("ridge_a8", Ridge(alpha=8.0, random_state=args.seed)),
        ("ridge_a20", Ridge(alpha=20.0, random_state=args.seed)),
        ("ridge_a60", Ridge(alpha=60.0, random_state=args.seed)),
        (
            "rf_d10_l4",
            RandomForestRegressor(
                n_estimators=900,
                max_depth=10,
                min_samples_leaf=4,
                random_state=args.seed,
                n_jobs=-1,
            ),
        ),
        (
            "rf_d14_l2",
            RandomForestRegressor(
                n_estimators=1200,
                max_depth=14,
                min_samples_leaf=2,
                random_state=args.seed,
                n_jobs=-1,
            ),
        ),
        (
            "rf_d18_l1",
            RandomForestRegressor(
                n_estimators=1400,
                max_depth=18,
                min_samples_leaf=1,
                random_state=args.seed,
                n_jobs=-1,
            ),
        ),
        (
            "extra_d14_l2",
            ExtraTreesRegressor(
                n_estimators=1200,
                max_depth=14,
                min_samples_leaf=2,
                random_state=args.seed,
                n_jobs=-1,
            ),
        ),
        (
            "extra_d18_l1",
            ExtraTreesRegressor(
                n_estimators=1400,
                max_depth=18,
                min_samples_leaf=1,
                random_state=args.seed,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gbrt_d5",
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=5,
                max_iter=700,
                l2_regularization=0.2,
                random_state=args.seed,
            ),
        ),
        (
            "hist_gbrt_d3",
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_depth=3,
                max_iter=1000,
                l2_regularization=0.4,
                random_state=args.seed,
            ),
        ),
        (
            "gbrt_d3",
            GradientBoostingRegressor(
                n_estimators=900,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                random_state=args.seed,
            ),
        ),
    ]

    rows: list[dict[str, object]] = []
    fit_models: dict[str, object] = {}
    test_preds: dict[str, np.ndarray] = {}
    holdout_rapm = pd.Series([np.nan] * len(test_df), index=test_df.index, dtype=float)
    if args.rapm_csv.exists():
        rapm_df = pd.read_csv(args.rapm_csv, low_memory=False)
        if {"player_name", "rapm"}.issubset(set(rapm_df.columns)):
            rapm_df = rapm_df[["player_name", "rapm"]].copy()
            rapm_df["name_key"] = rapm_df["player_name"].map(normalize_text)
            rapm_df["rapm"] = pd.to_numeric(rapm_df["rapm"], errors="coerce")
            rapm_df = rapm_df.sort_values("rapm", ascending=False, kind="stable").drop_duplicates(
                subset=["name_key"], keep="first"
            )
            rapm_map = dict(zip(rapm_df["name_key"], rapm_df["rapm"]))
            holdout_rapm = pd.to_numeric(test_df.get("name_key").map(rapm_map), errors="coerce")

    y_test_rank = pd.Series(y_test).rank(method="min", ascending=False)
    for name, mdl in model_specs:
        mdl.fit(X_train, y_train)
        p = mdl.predict(X_test)
        p_rank = pd.Series(p).rank(method="min", ascending=False)
        rank_err = p_rank - y_test_rank
        a_rapm, b_rapm = fit_linear_calibration(
            y_true=holdout_rapm.to_numpy(dtype=float),
            y_pred=np.asarray(p, dtype=float),
        )
        p_rapm = a_rapm + b_rapm * np.asarray(p, dtype=float)
        rapm_mask = np.isfinite(holdout_rapm.to_numpy(dtype=float)) & np.isfinite(p_rapm)
        if int(rapm_mask.sum()) >= 2:
            rapm_abs = np.abs(p_rapm[rapm_mask] - holdout_rapm.to_numpy(dtype=float)[rapm_mask])
            rapm_mae = float(np.mean(rapm_abs))
            rapm_rmse = float(np.sqrt(np.mean((p_rapm[rapm_mask] - holdout_rapm.to_numpy(dtype=float)[rapm_mask]) ** 2)))
            rapm_hit100 = int((rapm_abs <= 1.0).sum())
            rapm_hit100_rate = float((rapm_abs <= 1.0).mean())
            rapm_n = int(rapm_mask.sum())
        else:
            rapm_mae = float("nan")
            rapm_rmse = float("nan")
            rapm_hit100 = 0
            rapm_hit100_rate = float("nan")
            rapm_n = 0
        test_preds[name] = p
        rows.append(
            {
                "model": name,
                "rmse": rmse(y_test, p),
                "mae": mae(y_test, p),
                "spearman": spearman_corr(y_test, p),
                "mae_rank": float(np.mean(np.abs(rank_err))),
                "within1": int((np.abs(rank_err) <= 1).sum()),
                "within2": int((np.abs(rank_err) <= 2).sum()),
                "within5": int((np.abs(rank_err) <= 5).sum()),
                "within10": int((np.abs(rank_err) <= 10).sum()),
                "rapm_mae": rapm_mae,
                "rapm_rmse": rapm_rmse,
                "rapm_hit100": rapm_hit100,
                "rapm_hit100_rate": rapm_hit100_rate,
                "rapm_n": rapm_n,
                "n_test": int(len(test_df)),
            }
        )
        fit_models[name] = mdl

    # Blend top tree + linear models for stability.
    if {"ridge_a20", "rf_d14_l2", "extra_d14_l2"}.issubset(set(test_preds.keys())):
        p_blend = (
            test_preds["ridge_a20"] + test_preds["rf_d14_l2"] + test_preds["extra_d14_l2"]
        ) / 3.0
        p_rank = pd.Series(p_blend).rank(method="min", ascending=False)
        rank_err = p_rank - y_test_rank
        a_rapm, b_rapm = fit_linear_calibration(
            y_true=holdout_rapm.to_numpy(dtype=float),
            y_pred=np.asarray(p_blend, dtype=float),
        )
        p_rapm = a_rapm + b_rapm * np.asarray(p_blend, dtype=float)
        rapm_mask = np.isfinite(holdout_rapm.to_numpy(dtype=float)) & np.isfinite(p_rapm)
        if int(rapm_mask.sum()) >= 2:
            rapm_abs = np.abs(p_rapm[rapm_mask] - holdout_rapm.to_numpy(dtype=float)[rapm_mask])
            rapm_mae = float(np.mean(rapm_abs))
            rapm_rmse = float(np.sqrt(np.mean((p_rapm[rapm_mask] - holdout_rapm.to_numpy(dtype=float)[rapm_mask]) ** 2)))
            rapm_hit100 = int((rapm_abs <= 1.0).sum())
            rapm_hit100_rate = float((rapm_abs <= 1.0).mean())
            rapm_n = int(rapm_mask.sum())
        else:
            rapm_mae = float("nan")
            rapm_rmse = float("nan")
            rapm_hit100 = 0
            rapm_hit100_rate = float("nan")
            rapm_n = 0
        rows.append(
            {
                "model": "blend_ridge_rf_extra",
                "rmse": rmse(y_test, p_blend),
                "mae": mae(y_test, p_blend),
                "spearman": spearman_corr(y_test, p_blend),
                "mae_rank": float(np.mean(np.abs(rank_err))),
                "within1": int((np.abs(rank_err) <= 1).sum()),
                "within2": int((np.abs(rank_err) <= 2).sum()),
                "within5": int((np.abs(rank_err) <= 5).sum()),
                "within10": int((np.abs(rank_err) <= 10).sum()),
                "rapm_mae": rapm_mae,
                "rapm_rmse": rapm_rmse,
                "rapm_hit100": rapm_hit100,
                "rapm_hit100_rate": rapm_hit100_rate,
                "rapm_n": rapm_n,
                "n_test": int(len(test_df)),
            }
        )
        test_preds["blend_ridge_rf_extra"] = p_blend

    metrics_df = pd.DataFrame(rows)
    if args.selection_objective == "hit_rank1":
        metrics_df = metrics_df.sort_values(
            ["within1", "within2", "mae_rank", "rmse"], ascending=[False, False, True, True], kind="stable"
        )
    elif args.selection_objective == "hit_rank5":
        metrics_df = metrics_df.sort_values(
            ["within5", "within10", "mae_rank", "rmse"], ascending=[False, False, True, True], kind="stable"
        )
    elif args.selection_objective == "rank_mae":
        metrics_df = metrics_df.sort_values(["mae_rank", "spearman", "rmse"], ascending=[True, False, True], kind="stable")
    elif args.selection_objective == "spearman":
        metrics_df = metrics_df.sort_values(["spearman", "rmse", "mae"], ascending=[False, True, True], kind="stable")
    elif args.selection_objective == "rapm_mae":
        metrics_df = metrics_df.sort_values(
            ["rapm_mae", "rapm_rmse", "rapm_hit100_rate"], ascending=[True, True, False], kind="stable"
        )
    elif args.selection_objective == "rapm_rmse":
        metrics_df = metrics_df.sort_values(
            ["rapm_rmse", "rapm_mae", "rapm_hit100_rate"], ascending=[True, True, False], kind="stable"
        )
    elif args.selection_objective == "rapm_hit100":
        metrics_df = metrics_df.sort_values(
            ["rapm_hit100_rate", "rapm_mae", "rapm_rmse"], ascending=[False, True, True], kind="stable"
        )
    else:
        metrics_df = metrics_df.sort_values(["rmse", "mae", "mae_rank"], ascending=[True, True, True], kind="stable")
    metrics_df = metrics_df.reset_index(drop=True)
    best_name = str(metrics_df.iloc[0]["model"])
    best_test_pred = test_preds[best_name]

    # Holdout detail export for dashboard/debugging.
    holdout = test_df.copy()
    holdout["actual_impact_z"] = holdout["nba_impact_target_z"]
    holdout["pred_impact_z"] = best_test_pred
    holdout["actual_rank"] = holdout["actual_impact_z"].rank(method="min", ascending=False).astype(int)
    holdout["pred_rank_drafted"] = holdout["pred_impact_z"].rank(method="min", ascending=False).astype(int)
    holdout["pred_rank"] = holdout["pred_rank_drafted"]
    holdout["rank_error"] = holdout["pred_rank"] - holdout["actual_rank"]
    holdout["abs_rank_error"] = holdout["rank_error"].abs()

    # Also score full holdout-season pool so rank can be read against all players.
    holdout_pool = model_df[model_df["season"] == args.test_draft_year].copy()
    holdout_pool["source_row_id"] = holdout_pool.index
    if not holdout_pool.empty:
        X_pool = prep.transform(holdout_pool)
        X_pool = apply_feature_weights(X_pool, prep.feature_names)
        if best_name == "blend_ridge_rf_extra":
            holdout_pool["pred_impact_z_all_players"] = (
                fit_models["ridge_a20"].predict(X_pool)
                + fit_models["rf_d14_l2"].predict(X_pool)
                + fit_models["extra_d14_l2"].predict(X_pool)
            ) / 3.0
        else:
            holdout_pool["pred_impact_z_all_players"] = fit_models[best_name].predict(X_pool)
        holdout_pool["pred_rank_all_players"] = (
            holdout_pool["pred_impact_z_all_players"].rank(method="min", ascending=False).astype(int)
        )
        holdout = holdout.merge(
            holdout_pool[["source_row_id", "pred_impact_z_all_players", "pred_rank_all_players"]],
            how="left",
            on=["source_row_id"],
        )

    holdout = holdout.sort_values("actual_rank", kind="stable")

    # Refit best model on all target-known rows and score new joiners.
    final_prep = draft_model.FeaturePrep(num_cols, cat_cols).fit(model_df)
    X_all = final_prep.transform(model_df)
    X_all = apply_feature_weights(X_all, final_prep.feature_names)
    y_all = model_df["nba_impact_target_z"].to_numpy(dtype=float)

    # Future predictions remain draft-class focused (known entrants for the target draft year).
    predict_df = df[(df["season"] == args.predict_draft_year) & (df["drafted"] == 1)].copy()
    X_pred = final_prep.transform(predict_df)
    X_pred = apply_feature_weights(X_pred, final_prep.feature_names)
    if best_name == "blend_ridge_rf_extra":
        mdl_ridge = Ridge(alpha=20.0, random_state=args.seed)
        mdl_rf = RandomForestRegressor(
            n_estimators=1200,
            max_depth=14,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        )
        mdl_extra = ExtraTreesRegressor(
            n_estimators=1200,
            max_depth=14,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        )
        mdl_ridge.fit(X_all, y_all)
        mdl_rf.fit(X_all, y_all)
        mdl_extra.fit(X_all, y_all)
        predict_df["pred_nba_impact_z"] = (
            mdl_ridge.predict(X_pred) + mdl_rf.predict(X_pred) + mdl_extra.predict(X_pred)
        ) / 3.0
    else:
        final_model = dict(model_specs)[best_name]
        final_model.fit(X_all, y_all)
        predict_df["pred_nba_impact_z"] = final_model.predict(X_pred)
    predict_df = predict_df.sort_values("pred_nba_impact_z", ascending=False, kind="stable")
    predict_df["pred_impact_rank"] = np.arange(1, len(predict_df) + 1)

    keep = [
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
        "height_in",
        "weight_lb",
        "measurement_height_in",
        "measurement_weight_lb",
        "measurement_wingspan_in",
        "measurement_standing_reach_in",
        "measurement_wingspan_minus_height",
        "measurement_reach_minus_height",
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "crafted_height_in",
        "crafted_wingspan_in",
        "crafted_length_in",
        "crafted_wingspan_minus_height",
        "pred_nba_impact_z",
        "pred_impact_rank",
    ]
    keep = [c for c in keep if c in predict_df.columns]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets_out = args.out_dir / f"nba_impact_targets_window{args.target_window_years}.csv"
    metrics_out = args.out_dir / "nba_new_joiner_impact_model_metrics.csv"
    pred_out = args.out_dir / f"nba_new_joiner_impact_predictions_draft_{args.predict_draft_year}.csv"
    holdout_out = args.out_dir / f"nba_new_joiner_impact_holdout_actual_vs_predicted_{args.test_draft_year}.csv"
    report_out = args.out_dir / "nba_new_joiner_impact_model_report.txt"

    entrants.to_csv(targets_out, index=False)
    metrics_df.to_csv(metrics_out, index=False)
    predict_df[keep + [c for c in predict_df.columns if c not in keep]].to_csv(pred_out, index=False)
    holdout_keep = [
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
        "height_in",
        "weight_lb",
        "measurement_height_in",
        "measurement_weight_lb",
        "measurement_wingspan_in",
        "measurement_standing_reach_in",
        "measurement_wingspan_minus_height",
        "measurement_reach_minus_height",
        "combine_height_wo_shoes_in",
        "combine_weight_lb",
        "combine_wingspan_in",
        "combine_standing_reach_in",
        "crafted_height_in",
        "crafted_wingspan_in",
        "crafted_length_in",
        "crafted_wingspan_minus_height",
        "target_metrics_n",
        "actual_impact_z",
        "pred_impact_z",
        "pred_impact_z_all_players",
        "actual_rank",
        "pred_rank",
        "pred_rank_drafted",
        "pred_rank_all_players",
        "rank_error",
        "abs_rank_error",
    ]
    holdout_keep = [c for c in holdout_keep if c in holdout.columns] + [c for c in holdout.columns if c not in holdout_keep]
    holdout[holdout_keep].to_csv(holdout_out, index=False)

    lines: list[str] = []
    lines.append("College -> NBA New Joiner Impact Model (RAPM-family proxy)")
    lines.append(
        f"Target season mode: {args.target_season_mode} {args.target_window_years} NBA seasons after draft"
    )
    lines.append(f"Min target metrics required: {args.min_target_metrics}")
    lines.append(f"Model selection objective: {args.selection_objective}")
    lines.append(f"Feature profile: {args.feature_profile}")
    lines.append(f"RAPM evaluation source: {args.rapm_csv}")
    lines.append(f"Combine unique rows merged: {len(comb):,}")
    lines.append(f"Crafted measurement rows merged: {len(crafted):,}")
    lines.append(
        "Measurement coverage (model rows): "
        f"height={int(pd.to_numeric(model_df.get('measurement_height_in'), errors='coerce').notna().sum()):,}/"
        f"{len(model_df):,}, "
        f"wingspan={int(pd.to_numeric(model_df.get('measurement_wingspan_in'), errors='coerce').notna().sum()):,}/"
        f"{len(model_df):,}, "
        f"reach={int(pd.to_numeric(model_df.get('measurement_standing_reach_in'), errors='coerce').notna().sum()):,}/"
        f"{len(model_df):,}"
    )
    lines.append("")
    lines.append("Target metric coverage:")
    lines.extend([f"- {x}" for x in coverage_lines])
    lines.append("")
    lines.append(f"Target-known rows used for model: {len(model_df):,}")
    lines.append(f"Holdout draft year: {args.test_draft_year} (n={len(test_df):,})")
    lines.append("")
    lines.append("Holdout metrics by model:")
    for _, r in metrics_df.iterrows():
        lines.append(
            f"- {r['model']}: RMSE={float(r['rmse']):.4f}, MAE={float(r['mae']):.4f}, "
            f"Spearman={float(r['spearman']):.4f}, Hit<=1rank={int(r['within1'])}/{int(r['n_test'])}, "
            f"Hit<=5rank={int(r['within5'])}/{int(r['n_test'])}, "
            f"RAPM_MAE={float(r['rapm_mae']):.4f}, RAPM_Hit<=1={int(r['rapm_hit100'])}/{int(r['rapm_n'])}"
        )
    lines.append("")
    lines.append(f"Selected final model: {best_name}")
    lines.append(
        f"Holdout rank quality ({args.test_draft_year}, selected model): "
        f"MAE_rank={float(np.mean(np.abs(holdout['rank_error']))):.2f}, "
        f"within5={(holdout['abs_rank_error'] <= 5).sum():,}/{len(holdout):,}, "
        f"within10={(holdout['abs_rank_error'] <= 10).sum():,}/{len(holdout):,}"
    )
    lines.append(f"Predicted draft class: {args.predict_draft_year} (n={len(predict_df):,})")
    lines.append("Top 20 predicted NBA impact (z):")
    top = predict_df[["name", "team", "pick_number", "pred_nba_impact_z", "pred_impact_rank"]].head(20)
    for _, r in top.iterrows():
        pick = "n/a" if pd.isna(r.get("pick_number")) else int(float(r["pick_number"]))
        lines.append(
            f"- #{int(r['pred_impact_rank'])}: {r['name']} ({r['team']}), "
            f"pred_impact_z={float(r['pred_nba_impact_z']):.3f}, draft_pick={pick}"
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
