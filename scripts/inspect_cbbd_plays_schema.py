from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CBBD plays CSV schema.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--team", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safe_team = args.team.lower().replace(" ", "_")
    path = Path("data/raw/cbbd/plays") / f"season={args.season}" / f"team={safe_team}" / "plays.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run plays download first.")

    df = pd.read_csv(path)
    print(f"Rows: {len(df)}")
    print("Columns:")
    print(df.columns.tolist())
    print("\nSample rows:")
    print(df.head(args.limit).T)


if __name__ == "__main__":
    main()
