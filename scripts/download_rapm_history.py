from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "https://basketball-analytics.gitlab.io/rapm-data/data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download free historical NBA RAPM season data.")
    p.add_argument("--start-year", type=int, default=1996, help="Start season year, e.g. 1996 for 1996-97.")
    p.add_argument("--end-year", type=int, default=2026, help="End season year probe (exclusive upper season start).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/external/rapm_history"),
    )
    p.add_argument("--sleep-secs", type=float, default=0.05)
    return p.parse_args()


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def fetch_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def rows_from_payload(payload: dict, season: str, season_type: str) -> list[dict[str, object]]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    rows: list[dict[str, object]] = []
    start_year = int(season.split("-")[0])
    for r in data:
        if not isinstance(r, list) or len(r) < 7:
            continue
        rows.append(
            {
                "season": season,
                "season_start_year": start_year,
                "season_end_year": start_year + 1,
                "season_type": season_type,
                "rank": r[0],
                "player_name": r[1],
                "team": r[2],
                "possessions": r[3],
                "orapm": r[4],
                "drapm": r[5],
                "rapm": r[6],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    hit_count = 0

    for y in range(args.start_year, args.end_year):
        s = season_label(y)
        for suffix, stype in [("rapm", "regular"), ("playoffs-rapm", "playoffs")]:
            rel = f"{s}-{suffix}.json"
            url = f"{BASE_URL}/{rel}"
            payload = fetch_json(url)
            if payload is None:
                continue
            out_json = json_dir / rel
            out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            rows = rows_from_payload(payload, season=s, season_type=stype)
            all_rows.extend(rows)
            hit_count += 1
            print(f"[ok] {rel}: {len(rows):,} rows")
            if args.sleep_secs > 0:
                time.sleep(args.sleep_secs)

    if not all_rows:
        raise SystemExit("No RAPM files were downloaded. Check network/source availability.")

    out = pd.DataFrame(all_rows)
    for c in ["rank", "possessions", "orapm", "drapm", "rapm", "season_start_year", "season_end_year"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values(["season_start_year", "season_type", "rank"], kind="stable").reset_index(drop=True)
    out_all = args.out_dir / "rapm_history_all.csv"
    out_reg = args.out_dir / "rapm_history_regular.csv"
    out_po = args.out_dir / "rapm_history_playoffs.csv"
    out.to_csv(out_all, index=False)
    out[out["season_type"] == "regular"].to_csv(out_reg, index=False)
    out[out["season_type"] == "playoffs"].to_csv(out_po, index=False)

    print("")
    print(f"Downloaded files: {hit_count}")
    print(f"Rows total: {len(out):,}")
    print(f"Regular seasons: {out['season'].loc[out['season_type'] == 'regular'].nunique()}")
    print(f"Playoffs seasons: {out['season'].loc[out['season_type'] == 'playoffs'].nunique()}")
    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_reg}")
    print(f"Wrote: {out_po}")


if __name__ == "__main__":
    main()
