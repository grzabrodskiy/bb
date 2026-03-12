from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import train_nba_draft_predictors as t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze real top-60 draft misses (predicted rank > 60) and test a pure-stats rescue rerank."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
    )
    parser.add_argument("--min-train-year", type=int, default=2015)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--logit-l2", type=float, default=0.8)
    parser.add_argument("--logit-iters", type=int, default=2000)
    parser.add_argument("--logit-lr", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--neg-pos-ratio", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)

    # Rescue-rerank coefficients from holdout sweep.
    parser.add_argument("--coef-upperclass", type=float, default=3.0)
    parser.add_argument("--coef-minutes-z", type=float, default=2.0)
    parser.add_argument("--coef-high-tier", type=float, default=1.5)
    parser.add_argument("--coef-net-z", type=float, default=0.6)
    parser.add_argument("--coef-usage-z-penalty", type=float, default=0.6)

    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/draft_holdout_2025_boosted_rerank_analysis.csv"),
    )
    parser.add_argument(
        "--out-summary-txt",
        type=Path,
        default=Path("data/processed/draft_holdout_2025_boosted_rerank_summary.txt"),
    )
    return parser.parse_args()


def _zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0.0)
    sd = float(v.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
    return (v - float(v.mean())) / sd


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input table: {args.input_csv}")

    df = pd.read_csv(args.input_csv, low_memory=False)
    required = {"season", "label_known", "drafted", "pick_number"}
    miss = [c for c in sorted(required) if c not in df.columns]
    if miss:
        raise SystemExit(f"Input table missing required columns: {miss}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["label_known"] = t.as_bool_series(df["label_known"])
    df["drafted"] = pd.to_numeric(df["drafted"], errors="coerce")
    df["pick_number"] = pd.to_numeric(df["pick_number"], errors="coerce")
    df = t.add_engineered_features(df)

    known = df[df["label_known"] & df["drafted"].notna()].copy()
    known["drafted"] = known["drafted"].astype(int)

    train_df = known[(known["season"] >= args.min_train_year) & (known["season"] < args.test_year)].copy()
    test_df = known[known["season"] == args.test_year].copy()
    if train_df.empty or test_df.empty:
        raise SystemExit("Train/test split is empty with the provided year filters.")

    num_cols, cat_cols = t.choose_feature_columns(df)
    prep = t.FeaturePrep(num_cols, cat_cols).fit(train_df)

    train_cls = t.downsample_negatives(train_df, ratio=args.neg_pos_ratio, seed=args.seed)
    X_train = prep.transform(train_cls)
    y_train = train_cls["drafted"].to_numpy(dtype=float)
    cls_w = t.recency_weights(train_cls["season"])
    b0, b = t.fit_weighted_logistic(
        X_train,
        y_train,
        l2=args.logit_l2,
        lr=args.logit_lr,
        n_iters=args.logit_iters,
        sample_weights=cls_w,
    )

    drafted_train = train_df[train_df["drafted"] == 1].copy()
    X_pick_train = prep.transform(drafted_train)
    y_pick_train = drafted_train["pick_number"].to_numpy(dtype=float)
    ridge_w = t.recency_weights(drafted_train["season"])
    ridge_beta = t.fit_weighted_ridge_regression(
        X_pick_train,
        y_pick_train,
        alpha=args.ridge_alpha,
        sample_weights=ridge_w,
    )

    X_test = prep.transform(test_df)
    p = t.predict_logistic(X_test, b0, b)
    pick_if = np.clip(t.predict_ridge(X_test, ridge_beta), 1.0, 60.0)

    out = test_df.copy()
    out["p_drafted"] = p
    out["pred_pick_if_drafted"] = pick_if
    out["expected_pick"] = out["p_drafted"] * out["pred_pick_if_drafted"] + (1.0 - out["p_drafted"]) * 61.0
    out["rank_base"] = out["expected_pick"].rank(method="min", ascending=True).astype(int)

    minutes_z = _zscore(out.get("minutes", pd.Series([0.0] * len(out), index=out.index)))
    net_z = _zscore(out.get("net_rating", pd.Series([0.0] * len(out), index=out.index)))
    usage_z = _zscore(out.get("usage", pd.Series([0.0] * len(out), index=out.index)))
    upperclass = pd.to_numeric(out.get("is_upperclass", 0), errors="coerce").fillna(0.0)
    high_tier = pd.to_numeric(out.get("is_high_tier", 0), errors="coerce").fillna(0.0)

    out["expected_pick_boost"] = (
        out["expected_pick"]
        - args.coef_upperclass * upperclass
        - args.coef_minutes_z * minutes_z
        - args.coef_high_tier * high_tier
        - args.coef_net_z * net_z
        + args.coef_usage_z_penalty * usage_z
    )
    out["rank_boost"] = out["expected_pick_boost"].rank(method="min", ascending=True).astype(int)

    drafted = out[out["drafted"] == 1].copy()
    base_hits = int((drafted["rank_base"] <= 60).sum())
    boost_hits = int((drafted["rank_boost"] <= 60).sum())
    rescued = drafted[(drafted["rank_base"] > 60) & (drafted["rank_boost"] <= 60)].copy()
    dropped = drafted[(drafted["rank_base"] <= 60) & (drafted["rank_boost"] > 60)].copy()

    keep_cols = [
        "season",
        "athlete_id",
        "name",
        "team",
        "conference",
        "drafted",
        "pick_number",
        "p_drafted",
        "pred_pick_if_drafted",
        "expected_pick",
        "rank_base",
        "expected_pick_boost",
        "rank_boost",
        "minutes",
        "usage",
        "net_rating",
        "is_upperclass",
        "is_first_year",
        "is_high_tier",
        "is_mid_tier",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].sort_values(["rank_boost", "rank_base"]).copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    lines: list[str] = []
    lines.append("Top-60 Miss Recovery Analysis")
    lines.append(f"Test year: {args.test_year}")
    lines.append(f"Drafted in test year (covered by model universe): {len(drafted)}")
    lines.append(f"Baseline top-60 captured: {base_hits}")
    lines.append(f"Boosted top-60 captured: {boost_hits}")
    lines.append(f"Net change: {boost_hits - base_hits:+d}")
    lines.append("")
    lines.append("Boost coefficients:")
    lines.append(f"- upperclass: {args.coef_upperclass}")
    lines.append(f"- minutes_z: {args.coef_minutes_z}")
    lines.append(f"- high_tier: {args.coef_high_tier}")
    lines.append(f"- net_z: {args.coef_net_z}")
    lines.append(f"- usage_z_penalty: {args.coef_usage_z_penalty}")
    lines.append("")
    lines.append(f"Rescued players ({len(rescued)}):")
    if rescued.empty:
        lines.append("- (none)")
    else:
        for _, r in rescued.sort_values("rank_base").iterrows():
            lines.append(
                f"- {r.get('name')} ({r.get('team')}): pick={r.get('pick_number')}, "
                f"base_rank={int(r.get('rank_base'))}, boost_rank={int(r.get('rank_boost'))}"
            )
    lines.append("")
    lines.append(f"Dropped players ({len(dropped)}):")
    if dropped.empty:
        lines.append("- (none)")
    else:
        for _, r in dropped.sort_values("rank_base").iterrows():
            lines.append(
                f"- {r.get('name')} ({r.get('team')}): pick={r.get('pick_number')}, "
                f"base_rank={int(r.get('rank_base'))}, boost_rank={int(r.get('rank_boost'))}"
            )

    args.out_summary_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote detailed table: {args.out_csv}")
    print(f"Wrote summary: {args.out_summary_txt}")


if __name__ == "__main__":
    main()
