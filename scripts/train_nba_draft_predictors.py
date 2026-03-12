from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


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

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hurdle-model NBA draft predictors.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
    )
    parser.add_argument(
        "--draft-csv",
        type=Path,
        default=Path("data/raw/nba/draft/nba_draft_history_2015_2025.csv"),
        help="Historical draft labels used to build top-60 holdout coverage output.",
    )
    parser.add_argument(
        "--min-train-year",
        type=int,
        default=2015,
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2025,
        help="Known-label year used for holdout evaluation.",
    )
    parser.add_argument(
        "--predict-season",
        type=int,
        default=2026,
        help="Season to score with final model.",
    )
    parser.add_argument("--logit-l2", type=float, default=0.8)
    parser.add_argument("--logit-iters", type=int, default=2000)
    parser.add_argument("--logit-lr", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=20.0,
        help="Classifier-only negative:positive cap for training (0 disables downsampling).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
    )
    return parser.parse_args()


class FeaturePrep:
    def __init__(self, numeric_cols: list[str], cat_cols: list[str]) -> None:
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.num_medians: dict[str, float] = {}
        self.num_means: dict[str, float] = {}
        self.num_stds: dict[str, float] = {}
        self.cat_values: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeaturePrep":
        for c in self.numeric_cols:
            v = pd.to_numeric(df.get(c), errors="coerce")
            med = float(v.median()) if v.notna().any() else 0.0
            filled = v.fillna(med)
            mu = float(filled.mean())
            sd = float(filled.std(ddof=0))
            if not np.isfinite(sd) or sd <= 1e-12:
                sd = 1.0
            self.num_medians[c] = med
            self.num_means[c] = mu
            self.num_stds[c] = sd

        for c in self.cat_cols:
            vals = df.get(c, pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str).str.strip()
            uniq = sorted(v for v in vals.unique().tolist() if v)
            if "unknown" not in uniq:
                uniq.append("unknown")
            self.cat_values[c] = uniq

        self.feature_names = list(self.numeric_cols)
        for c in self.cat_cols:
            self.feature_names.extend([f"{c}={v}" for v in self.cat_values[c]])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        mats: list[np.ndarray] = []
        for c in self.numeric_cols:
            v = pd.to_numeric(df.get(c), errors="coerce").fillna(self.num_medians[c]).to_numpy(dtype=float)
            z = (v - self.num_means[c]) / self.num_stds[c]
            mats.append(z.reshape(-1, 1))
        for c in self.cat_cols:
            vals = df.get(c, pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str).str.strip()
            vals = vals.where(vals != "", "unknown")
            known = set(self.cat_values[c])
            vals = vals.where(vals.isin(known), "unknown")
            for v in self.cat_values[c]:
                mats.append((vals == v).to_numpy(dtype=float).reshape(-1, 1))
        if not mats:
            return np.zeros((len(df), 0), dtype=float)
        return np.hstack(mats)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def fit_weighted_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
    lr: float,
    n_iters: int,
    pos_weight: float | None = None,
    sample_weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    n, p = X.shape
    b0 = 0.0
    b = np.zeros(p, dtype=float)

    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if pos_weight is None:
        pos_weight = n_neg / max(n_pos, 1.0)
    class_w = np.where(y == 1, pos_weight, 1.0).astype(float)
    if sample_weights is None:
        sample_w = np.ones(n, dtype=float)
    else:
        sample_w = np.asarray(sample_weights, dtype=float)
        if sample_w.shape[0] != n:
            raise ValueError("sample_weights length mismatch")
        sample_w = np.clip(sample_w, 0.0, None)
    weights = class_w * sample_w

    step = lr
    for i in range(n_iters):
        z = b0 + X @ b
        p_hat = sigmoid(z)
        err = (p_hat - y) * weights
        g0 = float(err.mean())
        g = (X.T @ err) / max(n, 1) + l2 * b
        b0 -= step * g0
        b -= step * g
        if (i + 1) % 400 == 0:
            step *= 0.8
    return b0, b


def predict_logistic(X: np.ndarray, b0: float, b: np.ndarray) -> np.ndarray:
    return sigmoid(b0 + X @ b)


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n, p = X.shape
    X1 = np.hstack([np.ones((n, 1), dtype=float), X])
    reg = np.eye(p + 1, dtype=float) * alpha
    reg[0, 0] = 0.0
    lhs = X1.T @ X1 + reg
    rhs = X1.T @ y
    beta = np.linalg.solve(lhs, rhs)
    return beta


def fit_weighted_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    n, p = X.shape
    X1 = np.hstack([np.ones((n, 1), dtype=float), X])
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=float)
        if w.shape[0] != n:
            raise ValueError("sample_weights length mismatch")
        w = np.clip(w, 0.0, None)
        sw = np.sqrt(w)
        X1 = X1 * sw[:, None]
        y = y * sw
    reg = np.eye(p + 1, dtype=float) * alpha
    reg[0, 0] = 0.0
    lhs = X1.T @ X1 + reg
    rhs = X1.T @ y
    beta = np.linalg.solve(lhs, rhs)
    return beta


def predict_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    X1 = np.hstack([np.ones((n, 1), dtype=float), X])
    return X1 @ beta


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=float)
    rank_sum_pos = ranks[y == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(int)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(precision[y_sorted == 1].sum() / n_pos)
    return ap


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = y_true.astype(float)
    return float(np.mean((y_prob - y) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def as_bool_series(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


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


def build_team_context_features(
    seasons: list[int],
    games_root: Path = Path("data/raw/cbbd/games"),
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in sorted(set(seasons)):
        path = games_root / f"season={season}" / "games.csv"
        if not path.exists():
            continue
        g = pd.read_csv(path, low_memory=False)
        if g.empty:
            continue
        req = {
            "season",
            "status",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "home_winner",
            "away_winner",
            "home_team_elo_start",
            "home_team_elo_end",
            "away_team_elo_start",
            "away_team_elo_end",
        }
        missing = [c for c in req if c not in g.columns]
        if missing:
            continue

        g = g.copy()
        g = g[g["status"].astype(str).str.lower() == "final"]
        if g.empty:
            continue
        conf_game = g.get("conference_game", pd.Series([False] * len(g), index=g.index))
        conf_game = conf_game.fillna(False).astype(bool)

        home = pd.DataFrame(
            {
                "season": pd.to_numeric(g["season"], errors="coerce").astype("Int64"),
                "team": g["home_team"],
                "team_points": pd.to_numeric(g["home_points"], errors="coerce"),
                "opp_points": pd.to_numeric(g["away_points"], errors="coerce"),
                "won": (g["home_winner"] == True).astype(int),  # noqa: E712
                "team_elo_start": pd.to_numeric(g["home_team_elo_start"], errors="coerce"),
                "team_elo_end": pd.to_numeric(g["home_team_elo_end"], errors="coerce"),
                "opp_elo_start": pd.to_numeric(g["away_team_elo_start"], errors="coerce"),
                "conference_game": conf_game.astype(int),
            }
        )
        away = pd.DataFrame(
            {
                "season": pd.to_numeric(g["season"], errors="coerce").astype("Int64"),
                "team": g["away_team"],
                "team_points": pd.to_numeric(g["away_points"], errors="coerce"),
                "opp_points": pd.to_numeric(g["home_points"], errors="coerce"),
                "won": (g["away_winner"] == True).astype(int),  # noqa: E712
                "team_elo_start": pd.to_numeric(g["away_team_elo_start"], errors="coerce"),
                "team_elo_end": pd.to_numeric(g["away_team_elo_end"], errors="coerce"),
                "opp_elo_start": pd.to_numeric(g["home_team_elo_start"], errors="coerce"),
                "conference_game": conf_game.astype(int),
            }
        )
        both = pd.concat([home, away], ignore_index=True)
        both["team_key"] = both["team"].map(normalize_text)
        both["margin"] = both["team_points"] - both["opp_points"]
        both["elo_delta"] = both["team_elo_end"] - both["team_elo_start"]

        agg = (
            both.groupby(["season", "team_key"], dropna=False)
            .agg(
                team_games=("won", "count"),
                team_win_pct=("won", "mean"),
                team_avg_margin=("margin", "mean"),
                team_avg_elo_start=("team_elo_start", "mean"),
                team_avg_opp_elo_start=("opp_elo_start", "mean"),
                team_avg_elo_delta=("elo_delta", "mean"),
                team_conference_game_rate=("conference_game", "mean"),
            )
            .reset_index()
        )
        rows.append(agg)

    if not rows:
        return pd.DataFrame(columns=["season", "team_key"])
    return pd.concat(rows, ignore_index=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team_key"] = out.get("team", pd.Series([""] * len(out), index=out.index)).map(normalize_text)

    seasons = pd.to_numeric(out.get("season"), errors="coerce").dropna().astype(int).unique().tolist()
    team_ctx = build_team_context_features(seasons=seasons)
    if not team_ctx.empty:
        out = out.merge(team_ctx, how="left", on=["season", "team_key"])

    for c in [
        "minutes",
        "points",
        "assists",
        "rebounds_total",
        "turnovers",
        "age",
        "height_in",
        "weight_lb",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "steals_per40",
        "blocks_per40",
        "usage",
        "true_shooting_pct",
        "net_rating",
        "field_goals.attempted",
        "three_point_field_goals.attempted",
        "team_games",
        "team_win_pct",
        "team_avg_margin",
        "team_avg_elo_start",
        "team_avg_opp_elo_start",
        "team_avg_elo_delta",
        "team_conference_game_rate",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["log_minutes"] = np.log1p(pd.to_numeric(out.get("minutes"), errors="coerce").fillna(0.0))
    out["log_points"] = np.log1p(pd.to_numeric(out.get("points"), errors="coerce").fillna(0.0))
    out["log_assists"] = np.log1p(pd.to_numeric(out.get("assists"), errors="coerce").fillna(0.0))
    out["log_rebounds"] = np.log1p(pd.to_numeric(out.get("rebounds_total"), errors="coerce").fillna(0.0))
    out["assist_to_turnover"] = (
        pd.to_numeric(out.get("assists"), errors="coerce")
        / (pd.to_numeric(out.get("turnovers"), errors="coerce").replace(0, np.nan))
    )
    out["size_bmi_proxy"] = (
        pd.to_numeric(out.get("weight_lb"), errors="coerce")
        / ((pd.to_numeric(out.get("height_in"), errors="coerce").replace(0, np.nan) / 12.0) ** 2)
    )
    age = pd.to_numeric(out.get("age"), errors="coerce")
    out["age_known"] = age.notna().astype(int)
    out["age_u20"] = ((age < 20.0) & age.notna()).astype(int)
    out["age_u22"] = ((age < 22.0) & age.notna()).astype(int)
    out["years_since_first_seen_sq"] = (
        pd.to_numeric(out.get("years_since_first_seen"), errors="coerce").fillna(0.0) ** 2
    )

    # Interaction features to model role archetypes (high-usage creator vs efficient finisher, etc.)
    usage = pd.to_numeric(out.get("usage"), errors="coerce")
    pts40 = pd.to_numeric(out.get("points_per40"), errors="coerce")
    ast40 = pd.to_numeric(out.get("assists_per40"), errors="coerce")
    reb40 = pd.to_numeric(out.get("rebounds_total_per40"), errors="coerce")
    ts = pd.to_numeric(out.get("true_shooting_pct"), errors="coerce")
    net = pd.to_numeric(out.get("net_rating"), errors="coerce")
    stl40 = pd.to_numeric(out.get("steals_per40"), errors="coerce")
    blk40 = pd.to_numeric(out.get("blocks_per40"), errors="coerce")

    out["scoring_load_x_usage"] = (pts40 * usage) / 100.0
    out["playmaking_load_x_usage"] = (ast40 * usage) / 100.0
    out["rebounding_load_x_usage"] = (reb40 * usage) / 100.0
    out["efficiency_x_usage"] = ts * usage
    out["net_x_minutes"] = net * np.log1p(pd.to_numeric(out.get("minutes"), errors="coerce").fillna(0.0))
    out["stocks_per40"] = stl40 + blk40

    fga = pd.to_numeric(out.get("field_goals.attempted"), errors="coerce")
    tpa = pd.to_numeric(out.get("three_point_field_goals.attempted"), errors="coerce")
    out["three_point_attempt_rate"] = tpa / fga.replace(0, np.nan)

    # Missingness indicators help separate unknown values from true-average values after imputation.
    for c in [
        "age",
        "height_in",
        "weight_lb",
        "usage",
        "true_shooting_pct",
        "net_rating",
        "ext_draft_day_age",
        "ext_rsci",
        "ext_draft_score",
        "ext_percentile_score",
    ]:
        if c in out.columns:
            out[f"missing_{c}"] = pd.to_numeric(out[c], errors="coerce").isna().astype(int)

    # Season-relative percentiles make signals more stable across class-year distributions.
    season = pd.to_numeric(out.get("season"), errors="coerce")
    for c in [
        "minutes",
        "usage",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "true_shooting_pct",
        "net_rating",
        "win_shares_total_per40",
        "scoring_load_x_usage",
        "playmaking_load_x_usage",
        "stocks_per40",
    ]:
        if c in out.columns:
            v = pd.to_numeric(out[c], errors="coerce")
            out[f"{c}_season_pct"] = (
                pd.DataFrame({"season": season, "v": v})
                .groupby("season", dropna=False)["v"]
                .rank(method="average", pct=True)
            )

    tier = out.get("conference_tier", pd.Series(["other"] * len(out), index=out.index)).fillna("other").astype(str)
    out["is_high_tier"] = (tier == "high").astype(int)
    out["is_mid_tier"] = (tier == "mid").astype(int)
    out["is_other_tier"] = (tier == "other").astype(int)
    return out


def recency_weights(season_series: pd.Series) -> np.ndarray:
    s = pd.to_numeric(season_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lo = float(np.nanmin(s))
    hi = float(np.nanmax(s))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones(len(s), dtype=float)
    # More recent seasons carry more weight to match current draft dynamics.
    return 0.65 + 0.60 * ((s - lo) / (hi - lo))


def classifier_sampling_weights(df: pd.DataFrame) -> np.ndarray:
    mins = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base = 1.0 + np.log1p(np.maximum(mins, 0.0)) / np.log(10.0)
    tier = df.get("conference_tier", pd.Series(["other"] * len(df), index=df.index)).fillna("other").astype(str)
    tier_boost = np.where(tier == "high", 2.2, np.where(tier == "mid", 1.6, 1.0))
    return base * tier_boost


def downsample_negatives(train_df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    if ratio <= 0:
        return train_df
    pos = train_df[train_df["drafted"] == 1]
    neg = train_df[train_df["drafted"] == 0]
    if pos.empty or neg.empty:
        return train_df

    target_neg = int(max(1, np.floor(ratio * len(pos))))
    if len(neg) <= target_neg:
        return train_df

    probs = classifier_sampling_weights(neg)
    probs = probs / np.maximum(probs.sum(), 1e-12)
    sample_idx = np.random.default_rng(seed).choice(neg.index.to_numpy(), size=target_neg, replace=False, p=probs)
    sampled_neg = neg.loc[sample_idx]
    out = pd.concat([pos, sampled_neg], axis=0).sort_index()
    return out


def choose_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
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
        "porpag",
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
        "is_first_year",
        "is_upperclass",
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
        "team_games",
        "team_win_pct",
        "team_avg_margin",
        "team_avg_elo_start",
        "team_avg_opp_elo_start",
        "team_avg_elo_delta",
        "team_conference_game_rate",
        "age_known",
        "age_u20",
        "age_u22",
        "is_high_tier",
        "is_mid_tier",
        "is_other_tier",
    ]
    # Include dynamic engineered columns without hard-coding every name.
    dynamic_cols = [
        c
        for c in df.columns
        if c.startswith("missing_") or c.endswith("_season_pct")
    ]
    numeric_candidates.extend(dynamic_cols)
    cat_candidates = ["conference_tier", "position_group", "ext_match_type"]
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    num_cols = []
    for c in numeric_candidates:
        if c in df.columns and c not in seen:
            num_cols.append(c)
            seen.add(c)
    cat_cols = [c for c in cat_candidates if c in df.columns]
    return num_cols, cat_cols


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input table: {args.input_csv}")
    if not args.draft_csv.exists():
        raise SystemExit(f"Missing draft CSV: {args.draft_csv}")

    df = pd.read_csv(args.input_csv, low_memory=False)
    required = {"season", "athlete_id", "name", "team", "label_known", "drafted", "pick_number"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise SystemExit(f"Input table missing required columns: {missing}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["label_known"] = as_bool_series(df["label_known"])
    df["drafted"] = pd.to_numeric(df["drafted"], errors="coerce")
    df["pick_number"] = pd.to_numeric(df["pick_number"], errors="coerce")
    df = add_engineered_features(df)

    known = df[df["label_known"] & df["drafted"].notna()].copy()
    known["drafted"] = known["drafted"].astype(int)
    if known.empty:
        raise SystemExit("No labeled rows available for training.")

    num_cols, cat_cols = choose_feature_columns(df)
    if not num_cols and not cat_cols:
        raise SystemExit("No feature columns available for model training.")

    train_mask = (known["season"] >= args.min_train_year) & (known["season"] < args.test_year)
    test_mask = known["season"] == args.test_year
    train_df = known[train_mask].copy()
    test_df = known[test_mask].copy()
    if train_df.empty:
        raise SystemExit("No training rows after applying year filters.")

    prep = FeaturePrep(num_cols, cat_cols).fit(train_df)
    train_cls = downsample_negatives(train_df, ratio=args.neg_pos_ratio, seed=args.seed)
    X_train = prep.transform(train_cls)
    y_train = train_cls["drafted"].to_numpy(dtype=float)
    cls_weights = recency_weights(train_cls["season"])
    b0, b = fit_weighted_logistic(
        X_train,
        y_train,
        l2=args.logit_l2,
        lr=args.logit_lr,
        n_iters=args.logit_iters,
        sample_weights=cls_weights,
    )

    # Fit pick-number model on drafted train subset only.
    drafted_train = train_df[train_df["drafted"] == 1].copy()
    X_pick_train = prep.transform(drafted_train)
    y_pick_train = drafted_train["pick_number"].to_numpy(dtype=float)
    ridge_w = recency_weights(drafted_train["season"])
    ridge_beta = fit_weighted_ridge_regression(X_pick_train, y_pick_train, alpha=args.ridge_alpha, sample_weights=ridge_w)

    lines: list[str] = []
    lines.append("NBA Draft Hurdle Model Report")
    lines.append(f"Train seasons: {int(train_df['season'].min())}-{int(train_df['season'].max())}")
    lines.append(f"Test season: {args.test_year}")
    lines.append(f"Train rows: {len(train_df):,}")
    lines.append(f"Train drafted: {int((train_df['drafted'] == 1).sum()):,}")
    lines.append(f"Classifier train rows after negative downsampling: {len(train_cls):,}")
    lines.append(f"Features: {len(prep.feature_names)}")

    if not test_df.empty:
        X_test = prep.transform(test_df)
        y_test = test_df["drafted"].to_numpy(dtype=float)
        p_test = predict_logistic(X_test, b0, b)
        auc = roc_auc(y_test, p_test)
        ap = average_precision(y_test, p_test)
        brier = brier_score(y_test, p_test)

        lines.append("")
        lines.append("Classification (drafted) holdout metrics:")
        lines.append(f"ROC-AUC: {auc:.4f}")
        lines.append(f"PR-AUC: {ap:.4f}")
        lines.append(f"Brier: {brier:.4f}")

        drafted_test = test_df[test_df["drafted"] == 1].copy()
        if not drafted_test.empty:
            X_pick_test = prep.transform(drafted_test)
            pick_pred = np.clip(predict_ridge(X_pick_test, ridge_beta), 1.0, 60.0)
            pick_true = drafted_test["pick_number"].to_numpy(dtype=float)
            lines.append("")
            lines.append("Regression (pick number | drafted) holdout metrics:")
            lines.append(f"MAE: {mae(pick_true, pick_pred):.3f}")
            lines.append(f"RMSE: {rmse(pick_true, pick_pred):.3f}")

        scored_test = test_df.copy()
        scored_test["p_drafted"] = p_test
        pick_if_all = np.clip(predict_ridge(X_test, ridge_beta), 1.0, 60.0)
        scored_test["pred_pick_if_drafted"] = pick_if_all
        scored_test["expected_pick"] = scored_test["p_drafted"] * scored_test["pred_pick_if_drafted"] + (
            1.0 - scored_test["p_drafted"]
        ) * 61.0
        scored_test["pred_rank_prob"] = scored_test["p_drafted"].rank(method="min", ascending=False).astype(int)
        scored_test["pred_rank_expected_pick"] = scored_test["expected_pick"].rank(
            method="min", ascending=True
        ).astype(int)
        top60_prob = scored_test.sort_values("p_drafted", ascending=False).head(60)
        top60_expected = scored_test.sort_values("expected_pick", ascending=True).head(60)
        hits_top60 = int((top60_prob["drafted"] == 1).sum())
        hits_top60_expected = int((top60_expected["drafted"] == 1).sum())
        actual_drafted = int((test_df["drafted"] == 1).sum())
        lines.append("")
        lines.append("Draft-board proxy:")
        lines.append(f"Actual drafted in test year: {actual_drafted}")
        lines.append(f"Actual drafted captured in top-60 predicted probs: {hits_top60}")
        lines.append(f"Actual drafted captured in top-60 expected-pick board: {hits_top60_expected}")

        # Build dashboard-compatible holdout coverage file:
        # all real top-60 picks for test year + model coverage/prediction fields when available.
        draft_hist = pd.read_csv(args.draft_csv, low_memory=False)
        draft_hist["draft_year"] = pd.to_numeric(draft_hist.get("draft_year"), errors="coerce")
        draft_hist["pick_overall"] = pd.to_numeric(draft_hist.get("pick_overall"), errors="coerce")
        actual_top60 = draft_hist[
            (draft_hist["draft_year"] == args.test_year) & draft_hist["pick_overall"].notna() & (draft_hist["pick_overall"] <= 60)
        ].copy()
        actual_top60 = actual_top60.sort_values("pick_overall", kind="stable")
        actual_top60["name_key"] = actual_top60.get("player_name", pd.Series([""] * len(actual_top60))).map(normalize_name)
        actual_top60["college_key"] = actual_top60.get("college_name", pd.Series([""] * len(actual_top60))).map(normalize_text)

        scored_keep = [
            "name",
            "team",
            "conference",
            "position_group",
            "minutes",
            "p_drafted",
            "pred_pick_if_drafted",
            "expected_pick",
            "pred_rank_prob",
            "pred_rank_expected_pick",
        ]
        scored_keep = [c for c in scored_keep if c in scored_test.columns]
        scored_for_join = scored_test[scored_keep].copy()
        scored_for_join["name_key"] = scored_for_join.get("name", pd.Series([""] * len(scored_for_join))).map(normalize_name)
        scored_for_join["college_key"] = scored_for_join.get("team", pd.Series([""] * len(scored_for_join))).map(normalize_text)

        merged = actual_top60.merge(
            scored_for_join,
            how="left",
            on=["name_key", "college_key"],
            suffixes=("", "_m"),
        )

        # Fallback name-only match when unique and exact school match was unavailable.
        missing_mask = merged["name"].isna()
        if missing_mask.any():
            uniq_name = scored_for_join.groupby("name_key", dropna=False).size().rename("n").reset_index()
            uniq_name = uniq_name[uniq_name["n"] == 1][["name_key"]]
            fallback_pool = scored_for_join.merge(uniq_name, on="name_key", how="inner")
            if not fallback_pool.empty:
                miss_idx = merged.index[missing_mask]
                fb = merged.loc[miss_idx, ["name_key"]].merge(
                    fallback_pool,
                    how="left",
                    on=["name_key"],
                    suffixes=("", "_m"),
                )
                fb.index = miss_idx
                for c in ["name", "team", "conference", "position_group", "minutes", "p_drafted", "pred_pick_if_drafted", "expected_pick", "pred_rank_prob", "pred_rank_expected_pick"]:
                    if c in fb.columns and c in merged.columns:
                        merged.loc[missing_mask, c] = fb[c]

        merged["model_found"] = merged["name"].notna().astype(int)
        merged = merged.rename(
            columns={
                "p_drafted": "p_drafted_pred",
                "expected_pick": "expected_pick_pred",
            }
        )
        holdout_cols = [
            "pick_overall",
            "player_name",
            "college_name",
            "model_found",
            "name",
            "team",
            "conference",
            "position_group",
            "minutes",
            "p_drafted_pred",
            "pred_pick_if_drafted",
            "expected_pick_pred",
            "pred_rank_prob",
            "pred_rank_expected_pick",
        ]
        holdout_cols = [c for c in holdout_cols if c in merged.columns]
        out_holdout = args.out_dir / f"nba_draft_holdout_{args.test_year}_actual_top60_with_model_coverage.csv"
        merged[holdout_cols].to_csv(out_holdout, index=False)
        lines.append(f"Wrote holdout coverage: {out_holdout}")

    # Refit on all known rows up to test year for inference.
    final_train = known[(known["season"] >= args.min_train_year) & (known["season"] <= args.test_year)].copy()
    final_prep = FeaturePrep(num_cols, cat_cols).fit(final_train)
    final_cls = downsample_negatives(final_train, ratio=args.neg_pos_ratio, seed=args.seed)
    X_final = final_prep.transform(final_cls)
    y_final = final_cls["drafted"].to_numpy(dtype=float)
    final_cls_w = recency_weights(final_cls["season"])
    f0, fbeta = fit_weighted_logistic(
        X_final,
        y_final,
        l2=args.logit_l2,
        lr=args.logit_lr,
        n_iters=args.logit_iters,
        sample_weights=final_cls_w,
    )

    final_drafted = final_train[final_train["drafted"] == 1].copy()
    X_final_pick = final_prep.transform(final_drafted)
    y_final_pick = final_drafted["pick_number"].to_numpy(dtype=float)
    final_ridge = fit_weighted_ridge_regression(
        X_final_pick,
        y_final_pick,
        alpha=args.ridge_alpha,
        sample_weights=recency_weights(final_drafted["season"]),
    )

    pred_df = df[df["season"] == args.predict_season].copy()
    if pred_df.empty:
        lines.append("")
        lines.append(f"No rows found for predict season {args.predict_season}.")
    else:
        X_pred = final_prep.transform(pred_df)
        p = predict_logistic(X_pred, f0, fbeta)
        pick_if = np.clip(predict_ridge(X_pred, final_ridge), 1.0, 60.0)
        expected_pick = p * pick_if + (1.0 - p) * 61.0

        pred_df = pred_df.copy()
        pred_df["p_drafted"] = p
        pred_df["pred_pick_if_drafted"] = pick_if
        pred_df["expected_pick"] = expected_pick
        pred_df["pred_rank"] = pred_df["expected_pick"].rank(method="min", ascending=True).astype(int)
        pred_df = pred_df.sort_values(["expected_pick", "p_drafted"], ascending=[True, False])

        out_pred = args.out_dir / f"nba_draft_predictions_season_{args.predict_season}.csv"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        keep_cols = [
            "season",
            "athlete_id",
            "name",
            "team",
            "conference",
            "position_group",
            "minutes",
            "points_per40",
            "assists_per40",
            "rebounds_total_per40",
            "true_shooting_pct",
            "usage",
            "net_rating",
            "height_in",
            "weight_lb",
            "age",
            "p_drafted",
            "pred_pick_if_drafted",
            "expected_pick",
            "pred_rank",
        ]
        keep_cols = [c for c in keep_cols if c in pred_df.columns] + [c for c in pred_df.columns if c not in keep_cols]
        pred_df[keep_cols].to_csv(out_pred, index=False)
        lines.append("")
        lines.append(f"Wrote predictions: {out_pred}")
        lines.append("Top 10 expected picks:")
        top10 = pred_df[["name", "team", "p_drafted", "pred_pick_if_drafted", "expected_pick"]].head(10)
        for _, r in top10.iterrows():
            lines.append(
                f"- {r['name']} ({r['team']}): p_drafted={float(r['p_drafted']):.3f}, "
                f"pick_if={float(r['pred_pick_if_drafted']):.1f}, expected={float(r['expected_pick']):.1f}"
            )

    report_path = args.out_dir / "nba_draft_model_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
