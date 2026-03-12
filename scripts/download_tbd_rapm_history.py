from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT_PAGE = "https://thebasketballdatabase.com/203999RegularSeasonAdvanced.html"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download per-season RAPM from thebasketballdatabase.com player pages.")
    p.add_argument(
        "--training-csv",
        type=Path,
        default=Path("data/processed/nba_draft_training_table.csv"),
        help="Training table used to select drafted player names.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/raw/external/rapm_history/rapm_history_regular_tbd.csv"),
    )
    p.add_argument("--sleep-secs", type=float, default=0.03)
    return p.parse_args()


def name_key(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.lower().strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def read_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "ignore")


def build_player_id_map() -> pd.DataFrame:
    html = read_html(ROOT_PAGE)
    pairs = re.findall(r'<a\s+href="\\(\d+)RegularSeasonAdvanced\.html"[^>]*>\s*([^<]+?)\s*</a>', html)
    if not pairs:
        raise RuntimeError("Could not parse player id map from root page.")
    m = pd.DataFrame(pairs, columns=["nba_id", "player_name"])
    m["nba_id"] = pd.to_numeric(m["nba_id"], errors="coerce")
    m = m.dropna(subset=["nba_id"]).copy()
    m["nba_id"] = m["nba_id"].astype(int)
    m["name_key"] = m["player_name"].map(name_key)
    m = m.sort_values("nba_id", kind="stable").drop_duplicates(subset=["name_key"], keep="last")
    return m


def extract_one_year_rapm(player_id: int) -> pd.DataFrame:
    url = f"https://thebasketballdatabase.com/{player_id}RegularSeasonAdvanced.html"
    html = read_html(url)
    tables = pd.read_html(html)
    target = None
    for t in tables:
        cols = {str(c).strip().upper() for c in t.columns}
        if "END_SEASON" in cols and "ONE_YR_RAPM" in cols:
            target = t.copy()
            break
    if target is None:
        return pd.DataFrame(columns=["season_end_year", "rapm"])
    out = pd.DataFrame()
    out["season_end_year"] = pd.to_numeric(target.get("END_SEASON"), errors="coerce")
    out["rapm"] = pd.to_numeric(target.get("ONE_YR_RAPM"), errors="coerce")
    out = out.dropna(subset=["season_end_year", "rapm"]).copy()
    out["season_end_year"] = out["season_end_year"].astype(int)
    return out


def main() -> None:
    args = parse_args()
    if not args.training_csv.exists():
        raise SystemExit(f"Missing training CSV: {args.training_csv}")

    train = pd.read_csv(args.training_csv, low_memory=False)
    train["drafted"] = pd.to_numeric(train.get("drafted"), errors="coerce")
    drafted_names = (
        train[train["drafted"] == 1]["name"].dropna().astype(str).drop_duplicates().sort_values(kind="stable").tolist()
    )
    drafted = pd.DataFrame({"name": drafted_names})
    drafted["name_key"] = drafted["name"].map(name_key)

    id_map = build_player_id_map()
    todo = drafted.merge(id_map[["name_key", "nba_id"]], how="left", on="name_key")
    todo = todo.dropna(subset=["nba_id"]).copy()
    todo["nba_id"] = todo["nba_id"].astype(int)

    rows: list[pd.DataFrame] = []
    ok = 0
    for _, r in todo.iterrows():
        nm = str(r["name"])
        pid = int(r["nba_id"])
        try:
            d = extract_one_year_rapm(pid)
            if d.empty:
                continue
            d["player_name"] = nm
            d["nba_id"] = pid
            d["possessions"] = 1.0
            rows.append(d[["player_name", "nba_id", "season_end_year", "rapm", "possessions"]])
            ok += 1
        except Exception:
            continue
        if args.sleep_secs > 0:
            time.sleep(args.sleep_secs)

    if not rows:
        raise SystemExit("No RAPM rows downloaded from TBD pages.")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["season_end_year", "player_name"], kind="stable").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Matched players with IDs: {len(todo):,}")
    print(f"Players with RAPM rows: {ok:,}")
    print(f"Rows written: {len(out):,}")
    print(f"Season range: {int(out['season_end_year'].min())}..{int(out['season_end_year'].max())}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()

