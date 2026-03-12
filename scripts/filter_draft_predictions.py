from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter draft prediction rows to a cleaner D1 candidate board using pure local stats fields."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nba_draft_predictions_season_2026.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/nba_draft_predictions_season_2026_clean_board.csv"),
    )
    parser.add_argument(
        "--out-top-csv",
        type=Path,
        default=Path("data/processed/nba_draft_predictions_season_2026_clean_board_top30.csv"),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=300.0,
        help="Minimum season minutes required.",
    )
    parser.add_argument(
        "--max-years-since-first-seen",
        type=float,
        default=4.0,
        help="Maximum years since first tracked college season.",
    )
    parser.add_argument(
        "--allow-missing-conference",
        action="store_true",
        help="If set, do not require a non-empty conference value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input: {args.input_csv}")

    df = pd.read_csv(args.input_csv, low_memory=False)
    required = {"name", "team", "minutes", "expected_pick", "p_drafted", "years_since_first_seen"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise SystemExit(f"Input missing columns: {missing}")

    work = df.copy()
    work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce")
    work["years_since_first_seen"] = pd.to_numeric(work["years_since_first_seen"], errors="coerce")
    work["conference"] = work.get("conference", pd.Series([""] * len(work), index=work.index))
    conf_ok = work["conference"].fillna("").astype(str).str.strip() != ""
    mins_ok = work["minutes"].fillna(0.0) >= float(args.min_minutes)
    yrs_ok = work["years_since_first_seen"].fillna(999.0) <= float(args.max_years_since_first_seen)

    if args.allow_missing_conference:
        conf_mask = pd.Series(True, index=work.index)
    else:
        conf_mask = conf_ok

    mask = conf_mask & mins_ok & yrs_ok
    filtered = work[mask].copy()
    filtered = filtered.sort_values(["expected_pick", "p_drafted"], ascending=[True, False]).reset_index(drop=True)
    filtered["clean_rank"] = filtered.index + 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out_csv, index=False)

    top_n = int(max(args.top_n, 1))
    top = filtered.head(top_n).copy()
    args.out_top_csv.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.out_top_csv, index=False)

    print("Clean board filter summary")
    print(f"Input rows: {len(work):,}")
    print(f"Output rows: {len(filtered):,}")
    print(
        "Rule: "
        f"minutes >= {args.min_minutes:g}, "
        f"years_since_first_seen <= {args.max_years_since_first_seen:g}, "
        f"require conference = {not args.allow_missing_conference}"
    )
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote top-{top_n}: {args.out_top_csv}")


if __name__ == "__main__":
    main()
