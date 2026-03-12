from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "https://stats.nba.com/stats/draftcombineplayeranthro"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download NBA Draft Combine anthropometric measurements.")
    p.add_argument("--start-draft-year", type=int, default=2015, help="First draft year (inclusive).")
    p.add_argument("--end-draft-year", type=int, default=2026, help="Last draft year (inclusive).")
    p.add_argument("--league-id", type=str, default="00")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/external/nba_stats_draft/antro/antro"),
    )
    p.add_argument("--timeout", type=float, default=35.0)
    p.add_argument("--sleep-sec", type=float, default=0.05)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip years that already have non-empty Draft_antro_YYYY.csv output.",
    )
    return p.parse_args()


def season_year_for_draft_year(draft_year: int) -> str:
    # NBA stats Draft Combine endpoints are keyed by "draft-year season label",
    # e.g. draft 2022 -> SeasonYear "2022-23".
    return f"{draft_year}-{(draft_year + 1) % 100:02d}"


def _request_json(league_id: str, season_year: str, timeout: float) -> dict:
    qs = urllib.parse.urlencode({"LeagueID": league_id, "SeasonYear": season_year})
    url = f"{BASE_URL}?{qs}"
    req = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _payload_to_df(payload: dict) -> pd.DataFrame:
    result_sets = payload.get("resultSets")
    if isinstance(result_sets, list) and result_sets:
        rs0 = result_sets[0]
        headers = rs0.get("headers", [])
        rows = rs0.get("rowSet", [])
        if headers and isinstance(rows, list):
            return pd.DataFrame(rows, columns=headers)
    result_set = payload.get("resultSet")
    if isinstance(result_set, dict):
        headers = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])
        if headers and isinstance(rows, list):
            return pd.DataFrame(rows, columns=headers)
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    if args.end_draft_year < args.start_draft_year:
        raise SystemExit("--end-draft-year must be >= --start-draft-year")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    ok = 0
    fail = 0

    for draft_year in range(args.start_draft_year, args.end_draft_year + 1):
        out_csv = args.out_dir / f"Draft_antro_{draft_year}.csv"
        if args.skip_existing and out_csv.exists() and out_csv.stat().st_size > 0:
            print(f"[skip] draft={draft_year}: {out_csv}")
            continue

        season_year = season_year_for_draft_year(draft_year)
        try:
            payload = _request_json(
                league_id=args.league_id,
                season_year=season_year,
                timeout=args.timeout,
            )
            df = _payload_to_df(payload)
            if df.empty:
                raise RuntimeError("empty result set")
            df["DRAFT_YEAR"] = draft_year
            df["SEASON_YEAR"] = season_year
            df.to_csv(out_csv, index=False)
            print(f"[ok] draft={draft_year} season={season_year}: rows={len(df):,} -> {out_csv}")
            total_rows += int(len(df))
            ok += 1
        except Exception as exc:
            print(f"[fail] draft={draft_year} season={season_year}: {exc}")
            fail += 1
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print("")
    print(f"Completed. Success years: {ok}, failed years: {fail}, rows downloaded: {total_rows:,}")


if __name__ == "__main__":
    main()
