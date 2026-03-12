from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_nba_draft_predictors as model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute player-specific root causes for 2025 draft misses.")
    p.add_argument(
        "--training-csv",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
    )
    p.add_argument(
        "--holdout-csv",
        type=Path,
        default=Path("data/processed/nba_draft_holdout_2025_actual_top60_with_model_coverage.csv"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/nba_draft_holdout_2025_miss_root_causes.csv"),
    )
    p.add_argument("--min-train-year", type=int, default=2015)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--logit-l2", type=float, default=0.8)
    p.add_argument("--logit-iters", type=int, default=2000)
    p.add_argument("--logit-lr", type=float, default=0.05)
    p.add_argument("--neg-pos-ratio", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm(s: object) -> str:
    return model.normalize_text(s)


def _fmt(v: float | int | None, d: int = 1, suffix: str = "") -> str:
    if v is None:
        return "n/a"
    try:
        vv = float(v)
    except Exception:
        return "n/a"
    if not np.isfinite(vv):
        return "n/a"
    if d == 0:
        return f"{int(round(vv))}{suffix}"
    return f"{vv:.{d}f}{suffix}"


def root_cause_tag(
    holdout_row: pd.Series,
    feature_row: pd.Series,
    contrib: pd.Series,
) -> tuple[str, str, float]:
    mins = (
        float(pd.to_numeric(holdout_row.get("minutes"), errors="coerce"))
        if pd.notna(holdout_row.get("minutes"))
        else np.nan
    )
    games = (
        float(pd.to_numeric(feature_row.get("games"), errors="coerce"))
        if pd.notna(feature_row.get("games"))
        else np.nan
    )
    usage = (
        float(pd.to_numeric(feature_row.get("usage"), errors="coerce"))
        if pd.notna(feature_row.get("usage"))
        else np.nan
    )
    pts40 = (
        float(pd.to_numeric(feature_row.get("points_per40"), errors="coerce"))
        if pd.notna(feature_row.get("points_per40"))
        else np.nan
    )
    ast40 = (
        float(pd.to_numeric(feature_row.get("assists_per40"), errors="coerce"))
        if pd.notna(feature_row.get("assists_per40"))
        else np.nan
    )
    reb40 = (
        float(pd.to_numeric(feature_row.get("rebounds_total_per40"), errors="coerce"))
        if pd.notna(feature_row.get("rebounds_total_per40"))
        else np.nan
    )
    tov40 = (
        float(pd.to_numeric(feature_row.get("turnovers_per40"), errors="coerce"))
        if pd.notna(feature_row.get("turnovers_per40"))
        else np.nan
    )
    ts = (
        float(pd.to_numeric(feature_row.get("true_shooting_pct"), errors="coerce"))
        if pd.notna(feature_row.get("true_shooting_pct"))
        else np.nan
    )
    three_pa_rate = (
        float(pd.to_numeric(feature_row.get("three_point_attempt_rate"), errors="coerce"))
        if pd.notna(feature_row.get("three_point_attempt_rate"))
        else np.nan
    )
    three_pct = (
        float(pd.to_numeric(feature_row.get("three_point_pct"), errors="coerce"))
        if pd.notna(feature_row.get("three_point_pct"))
        else np.nan
    )
    three_pct_display = np.nan
    if np.isfinite(three_pct):
        # Some tables store 3P% in [0, 1], others in [0, 100].
        three_pct_display = 100.0 * three_pct if three_pct <= 1.5 else three_pct
    net = (
        float(pd.to_numeric(feature_row.get("net_rating"), errors="coerce"))
        if pd.notna(feature_row.get("net_rating"))
        else np.nan
    )
    p = (
        float(pd.to_numeric(holdout_row.get("p_drafted_pred"), errors="coerce"))
        if pd.notna(holdout_row.get("p_drafted_pred"))
        else np.nan
    )
    years = (
        float(pd.to_numeric(feature_row.get("years_since_first_seen"), errors="coerce"))
        if pd.notna(feature_row.get("years_since_first_seen"))
        else np.nan
    )

    def c(name: str) -> float:
        try:
            v = float(contrib.get(name, 0.0))
        except Exception:
            return 0.0
        return v if np.isfinite(v) else 0.0

    # For miss explanations, prefer a concise combined stat reason when
    # efficiency and on-court impact are both weak, even if a single coefficient
    # (for example rebounds) is numerically slightly more negative.
    if np.isfinite(ts) and np.isfinite(net) and ts < 0.55 and net < 0:
        msg = f"Low efficiency/impact: TS {_fmt(100.0 * ts, 1, '%')}, net {_fmt(net, 1)}"
        if np.isfinite(tov40) and tov40 >= 3.5:
            msg += f", TO {_fmt(tov40, 1)}/40"
        if np.isfinite(three_pa_rate) and three_pa_rate >= 0.40:
            msg += f", 3PA {_fmt(100.0 * three_pa_rate, 1, '%')}"
        combo_score = min(
            c("true_shooting_pct"),
            c("true_shooting_pct_season_pct"),
            c("net_rating"),
            c("net_rating_season_pct"),
        )
        if combo_score >= 0:
            combo_score = -1e-6
        return ("efficiency_impact_combo", msg, combo_score)

    candidates: list[tuple[str, str, float]] = []

    # Player-availability features.
    if np.isfinite(mins):
        candidates.append(("minutes", f"Low play time: {_fmt(mins, 0, 'm')}", c("minutes")))
    if np.isfinite(games):
        candidates.append(("games", f"Few games played: {_fmt(games, 0)}", c("games")))

    # Role / production features.
    if np.isfinite(usage):
        candidates.append(("usage", f"Low offensive role: usage {_fmt(usage, 1, '%')}", c("usage")))
    if np.isfinite(pts40):
        candidates.append(("points_per40", f"Lower scoring rate: {_fmt(pts40, 1)}/40", c("points_per40")))
    if np.isfinite(ast40):
        candidates.append(("assists_per40", f"Lower playmaking rate: {_fmt(ast40, 1)}/40", c("assists_per40")))
    if np.isfinite(reb40):
        candidates.append(("rebounds_total_per40", f"Lower rebound rate: {_fmt(reb40, 1)}/40", c("rebounds_total_per40")))
    if np.isfinite(tov40):
        candidates.append(("turnovers_per40", f"Turnover concern: {_fmt(tov40, 1)}/40", c("turnovers_per40")))
    if np.isfinite(ts):
        candidates.append(("true_shooting_pct", f"Scoring efficiency concern: TS {_fmt(100.0 * ts, 1, '%')}", c("true_shooting_pct")))
    if np.isfinite(three_pa_rate):
        candidates.append(
            (
                "three_point_attempt_rate",
                f"Low 3PA rate: {_fmt(100.0 * three_pa_rate, 1, '%')}",
                c("three_point_attempt_rate"),
            )
        )
    if np.isfinite(three_pct):
        candidates.append(("three_point_pct", f"Lower 3P%: {_fmt(three_pct_display, 1, '%')}", c("three_point_pct")))
    if np.isfinite(net):
        candidates.append(("net_rating", f"Impact signal concern: net {_fmt(net, 1)}", c("net_rating")))
    if np.isfinite(years):
        yrs = int(round(years))
        candidates.append(
            (
                "years_since_first_seen",
                f"Older prospect profile: {yrs}y since first season",
                min(c("years_since_first_seen"), c("years_since_first_seen_sq")),
            )
        )

    # Context / priors.
    candidates.append(("conference_tier=mid", "Mid-major historical draft hit rate drag", c("conference_tier=mid")))
    candidates.append(("conference_tier=other", "Non-major conference historical drag", c("conference_tier=other")))

    neg = [x for x in candidates if x[2] < 0]
    if neg:
        best = min(neg, key=lambda x: x[2])
        return best

    # Fallback: pick the strongest negative non-external signal directly from feature contributions.
    def generic_tag_from_feature(fname: str) -> str | None:
        if fname in {"minutes", "log_minutes"} and np.isfinite(mins):
            return f"Low play time: {_fmt(mins, 0, 'm')}"
        if fname == "games" and np.isfinite(games):
            return f"Few games played: {_fmt(games, 0)}"
        if fname == "usage" and np.isfinite(usage):
            return f"Low offensive role: usage {_fmt(usage, 1, '%')}"
        if fname == "points_per40" and np.isfinite(pts40):
            return f"Lower scoring rate: {_fmt(pts40, 1)}/40"
        if fname == "assists_per40" and np.isfinite(ast40):
            return f"Lower playmaking rate: {_fmt(ast40, 1)}/40"
        if fname == "rebounds_total_per40" and np.isfinite(reb40):
            return f"Lower rebound rate: {_fmt(reb40, 1)}/40"
        if fname == "turnovers_per40" and np.isfinite(tov40):
            return f"Turnover concern: {_fmt(tov40, 1)}/40"
        if fname == "true_shooting_pct" and np.isfinite(ts):
            return f"Scoring efficiency concern: TS {_fmt(100.0 * ts, 1, '%')}"
        if fname in {"three_point_attempt_rate", "three_point_attempt_rate_season_pct"} and np.isfinite(three_pa_rate):
            return f"Low 3PA rate: {_fmt(100.0 * three_pa_rate, 1, '%')}"
        if fname in {"three_point_pct", "three_point_pct_season_pct"} and np.isfinite(three_pct):
            return f"Lower 3P%: {_fmt(three_pct_display, 1, '%')}"
        if fname == "net_rating" and np.isfinite(net):
            return f"Impact signal concern: net {_fmt(net, 1)}"
        if fname in {"years_since_first_seen", "years_since_first_seen_sq"} and np.isfinite(years):
            return f"Older prospect profile: {int(round(years))}y since first season"
        if fname == "conference_tier=mid":
            return "Mid-major historical draft hit rate drag"
        if fname == "conference_tier=other":
            return "Non-major conference historical drag"
        if fname == "is_upperclass":
            return "Upperclass profile signal"
        if fname == "is_first_year":
            return "Not a freshman breakout profile"
        return None

    for fname, val in contrib.sort_values().items():
        if float(val) >= 0:
            break
        if fname.startswith("ext_") or fname.startswith("missing_ext") or fname == "ext_school_match":
            continue
        tag = generic_tag_from_feature(str(fname))
        if tag:
            return (str(fname), tag, float(val))

    # If no clear negative driver is available, use a concise probability summary.
    if np.isfinite(p):
        return ("p_drafted_pred", f"Draft chance stayed modest: {_fmt(100.0 * p, 1, '%')}", 0.0)
    return ("unknown", "Insufficient feature signal for a specific cause", 0.0)


def main() -> None:
    args = parse_args()
    if not args.training_csv.exists():
        raise SystemExit(f"Missing training table: {args.training_csv}")
    if not args.holdout_csv.exists():
        raise SystemExit(f"Missing holdout table: {args.holdout_csv}")

    train_raw = pd.read_csv(args.training_csv, low_memory=False)
    train_raw["season"] = _safe_num(train_raw["season"]).astype("Int64")
    train_raw["label_known"] = model.as_bool_series(train_raw["label_known"])
    train_raw["drafted"] = _safe_num(train_raw["drafted"])
    train_raw["pick_number"] = _safe_num(train_raw["pick_number"])
    train_df = model.add_engineered_features(train_raw)

    known = train_df[train_df["label_known"] & train_df["drafted"].notna()].copy()
    known["drafted"] = known["drafted"].astype(int)

    num_cols, cat_cols = model.choose_feature_columns(train_df)
    train_mask = (known["season"] >= args.min_train_year) & (known["season"] < args.test_year)
    test_mask = known["season"] == args.test_year
    train_split = known[train_mask].copy()
    test_split = known[test_mask].copy()

    prep = model.FeaturePrep(num_cols, cat_cols).fit(train_split)
    train_cls = model.downsample_negatives(train_split, ratio=args.neg_pos_ratio, seed=args.seed)
    X_train = prep.transform(train_cls)
    y_train = train_cls["drafted"].to_numpy(dtype=float)
    cls_weights = model.recency_weights(train_cls["season"])
    b0, b = model.fit_weighted_logistic(
        X_train,
        y_train,
        l2=args.logit_l2,
        lr=args.logit_lr,
        n_iters=args.logit_iters,
        sample_weights=cls_weights,
    )

    hold = pd.read_csv(args.holdout_csv, low_memory=False)
    for c in ["pick_overall", "model_found", "p_drafted_pred", "pred_pick_if_drafted", "expected_pick_pred", "pred_rank_expected_pick"]:
        if c in hold.columns:
            hold[c] = _safe_num(hold[c])
    hold = hold.copy()
    hold["name_key"] = hold.get("name", hold.get("player_name", pd.Series([""] * len(hold), index=hold.index))).map(_norm)
    hold["team_key"] = hold.get("team", pd.Series([""] * len(hold), index=hold.index)).map(_norm)

    test_split = test_split.copy()
    test_split["name_key"] = test_split.get("name", pd.Series([""] * len(test_split), index=test_split.index)).map(_norm)
    test_split["team_key"] = test_split.get("team", pd.Series([""] * len(test_split), index=test_split.index)).map(_norm)
    test_ix = {(r.name_key, r.team_key): i for i, r in test_split.reset_index(drop=True).iterrows()}
    X_test = prep.transform(test_split.reset_index(drop=True))

    out_rows: list[dict[str, object]] = []
    miss = hold[(hold["model_found"] == 1) & (hold["pred_rank_expected_pick"] > 60)].copy()
    for _, r in miss.iterrows():
        key = (_norm(r.get("name")), _norm(r.get("team")))
        i = test_ix.get(key)
        if i is None:
            out_rows.append(
                {
                    "pick_overall": r.get("pick_overall"),
                    "player_name": r.get("player_name"),
                    "name": r.get("name"),
                    "team": r.get("team"),
                    "miss_tag_debug": "No matching feature row in 2025 table",
                    "miss_driver_feature": "missing_match",
                    "miss_driver_contrib": np.nan,
                }
            )
            continue

        row = test_split.reset_index(drop=True).iloc[i]
        contrib = pd.Series(X_test[i] * b, index=prep.feature_names)
        feature, tag, score = root_cause_tag(r, row, contrib)
        out_rows.append(
            {
                "pick_overall": r.get("pick_overall"),
                "player_name": r.get("player_name"),
                "name": r.get("name"),
                "team": r.get("team"),
                "miss_tag_debug": tag,
                "miss_driver_feature": feature,
                "miss_driver_contrib": float(score),
                "p_drafted_pred": r.get("p_drafted_pred"),
                "expected_pick_pred": r.get("expected_pick_pred"),
                "pred_rank_expected_pick": r.get("pred_rank_expected_pick"),
                "minutes": r.get("minutes"),
                "games": row.get("games"),
                "usage": row.get("usage"),
                "points_per40": row.get("points_per40"),
                "true_shooting_pct": row.get("true_shooting_pct"),
                "net_rating": row.get("net_rating"),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["pick_overall", "player_name"], kind="stable")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
