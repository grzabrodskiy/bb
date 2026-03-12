from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import train_nba_draft_predictors as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sklearn draft models on 2025 holdout.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
    )
    parser.add_argument("--min-train-year", type=int, default=2015)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("data/processed/nba_draft_model_compare_sklearn_2025.txt"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/nba_draft_model_compare_sklearn_2025.csv"),
    )
    return parser.parse_args()


def make_preprocessor(num_cols: list[str], cat_cols: list[str], dense: bool) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input table: {args.input_csv}")

    df = pd.read_csv(args.input_csv, low_memory=False)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["label_known"] = base.as_bool_series(df["label_known"])
    df["drafted"] = pd.to_numeric(df["drafted"], errors="coerce")
    df["pick_number"] = pd.to_numeric(df["pick_number"], errors="coerce")
    df = base.add_engineered_features(df)

    known = df[df["label_known"] & df["drafted"].notna()].copy()
    known["drafted"] = known["drafted"].astype(int)

    train_df = known[(known["season"] >= args.min_train_year) & (known["season"] < args.test_year)].copy()
    test_df = known[known["season"] == args.test_year].copy()
    if train_df.empty or test_df.empty:
        raise SystemExit("Empty train/test split.")

    num_cols, cat_cols = base.choose_feature_columns(df)
    feats = num_cols + cat_cols
    X_train = train_df[feats].copy()
    X_test = test_df[feats].copy()
    y_train = train_df["drafted"].to_numpy(dtype=int)
    y_test = test_df["drafted"].to_numpy(dtype=int)
    w_train = base.recency_weights(train_df["season"])

    drafted_train = train_df[train_df["drafted"] == 1].copy()
    X_train_pick = drafted_train[feats].copy()
    y_train_pick = drafted_train["pick_number"].to_numpy(dtype=float)
    w_train_pick = base.recency_weights(drafted_train["season"])

    models = {
        "rf": {
            "clf": RandomForestClassifier(
                n_estimators=700,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=args.seed,
            ),
            "reg": RandomForestRegressor(
                n_estimators=700,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=args.seed,
            ),
            "dense": False,
        },
        "hgbt": {
            "clf": HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=420,
                random_state=args.seed,
            ),
            "reg": HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=520,
                random_state=args.seed,
            ),
            "dense": True,
        },
    }

    rows: list[dict[str, object]] = []
    lines: list[str] = []
    lines.append("Sklearn Model Comparison (2025 holdout)")
    lines.append(f"Train rows: {len(train_df):,}, test rows: {len(test_df):,}, drafted test: {int((y_test==1).sum())}")
    lines.append("")

    for name, cfg in models.items():
        pre = make_preprocessor(num_cols, cat_cols, dense=bool(cfg["dense"]))
        clf = Pipeline([("pre", pre), ("model", cfg["clf"])])
        reg = Pipeline([("pre", pre), ("model", cfg["reg"])])

        clf.fit(X_train, y_train, model__sample_weight=w_train)
        p = clf.predict_proba(X_test)[:, 1]

        reg.fit(X_train_pick, y_train_pick, model__sample_weight=w_train_pick)
        pick_if = np.clip(reg.predict(X_test), 1.0, 60.0)

        auc = base.roc_auc(y_test.astype(float), p)
        ap = base.average_precision(y_test.astype(float), p)
        brier = base.brier_score(y_test.astype(float), p)

        drafted_mask = test_df["drafted"] == 1
        mae = base.mae(test_df.loc[drafted_mask, "pick_number"].to_numpy(dtype=float), pick_if[drafted_mask.to_numpy()])
        rmse = base.rmse(test_df.loc[drafted_mask, "pick_number"].to_numpy(dtype=float), pick_if[drafted_mask.to_numpy()])

        for K in [50.0, 61.0]:
            expected = p * pick_if + (1.0 - p) * K
            rank_expected = pd.Series(expected, index=test_df.index).rank(method="min", ascending=True)
            rank_prob = pd.Series(p, index=test_df.index).rank(method="min", ascending=False)
            hits_expected = int((rank_expected.loc[test_df.index[drafted_mask]] <= 60).sum())
            hits_prob = int((rank_prob.loc[test_df.index[drafted_mask]] <= 60).sum())
            rows.append(
                {
                    "model": name,
                    "K": K,
                    "roc_auc": auc,
                    "pr_auc": ap,
                    "brier": brier,
                    "mae_pickif": mae,
                    "rmse_pickif": rmse,
                    "hits_top60_prob": hits_prob,
                    "hits_top60_expected": hits_expected,
                }
            )
            lines.append(
                f"{name} (K={int(K)}): ROC={auc:.4f}, PR={ap:.4f}, Brier={brier:.4f}, "
                f"MAE={mae:.3f}, RMSE={rmse:.3f}, top60_prob={hits_prob}/45, top60_expected={hits_expected}/45"
            )

    out = pd.DataFrame(rows).sort_values(
        ["hits_top60_expected", "hits_top60_prob", "pr_auc", "mae_pickif"],
        ascending=[False, False, False, True],
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    best = out.iloc[0].to_dict()
    lines.append("")
    lines.append(
        "Best by expected top-60 capture: "
        f"model={best['model']}, K={int(best['K'])}, "
        f"top60_expected={int(best['hits_top60_expected'])}/45, top60_prob={int(best['hits_top60_prob'])}/45"
    )
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {args.out_csv}")
    print(f"Wrote: {args.out_report}")


if __name__ == "__main__":
    main()
