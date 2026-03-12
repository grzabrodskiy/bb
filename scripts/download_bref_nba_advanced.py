from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download NBA player advanced tables from Basketball-Reference.")
    p.add_argument("--year-start", type=int, default=2010)
    p.add_argument("--year-end", type=int, default=2026)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/raw/nba/bref/player_advanced_2010_2026.csv"),
    )
    p.add_argument("--sleep-sec", type=float, default=1.0)
    return p.parse_args()


def normalize_advanced_table(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # Some pages include repeated header rows inside the table.
    if "Rk" in out.columns:
        out = out[out["Rk"].astype(str) != "Rk"]

    keep = [
        "Player",
        "Pos",
        "Tm",
        "Age",
        "G",
        "MP",
        "PER",
        "TS%",
        "3PAr",
        "FTr",
        "ORB%",
        "DRB%",
        "TRB%",
        "AST%",
        "STL%",
        "BLK%",
        "TOV%",
        "USG%",
        "OWS",
        "DWS",
        "WS",
        "WS/48",
        "OBPM",
        "DBPM",
        "BPM",
        "VORP",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()
    out["year"] = year
    return out


def main() -> None:
    args = parse_args()
    rows: list[pd.DataFrame] = []
    for year in range(args.year_start, args.year_end + 1):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        print(f"Downloading year={year} ...")
        tables = pd.read_html(url)
        if not tables:
            print(f"  no table found at {url}")
            continue
        df = normalize_advanced_table(tables[0], year=year)
        rows.append(df)
        print(f"  rows={len(df):,}")
        time.sleep(max(args.sleep_sec, 0.0))

    if not rows:
        raise SystemExit("No rows downloaded.")

    out = pd.concat(rows, ignore_index=True)
    out["Age"] = pd.to_numeric(out.get("Age"), errors="coerce")
    for c in [
        "G",
        "MP",
        "PER",
        "TS%",
        "3PAr",
        "FTr",
        "ORB%",
        "DRB%",
        "TRB%",
        "AST%",
        "STL%",
        "BLK%",
        "TOV%",
        "USG%",
        "OWS",
        "DWS",
        "WS",
        "WS/48",
        "OBPM",
        "DBPM",
        "BPM",
        "VORP",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
