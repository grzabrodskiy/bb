from __future__ import annotations

import argparse
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


class DraftTableParser(HTMLParser):
    """Parse Basketball-Reference draft rows from table id='stats'."""

    def __init__(self) -> None:
        super().__init__()
        self.in_stats_table = False
        self.in_row = False
        self.current_field: Optional[str] = None
        self.current_text: list[str] = []
        self.current_href: Optional[str] = None
        self.row: dict[str, object] = {}
        self.rows: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attr = {k: v for k, v in attrs}
        if tag == "table" and attr.get("id") == "stats":
            self.in_stats_table = True
            return
        if not self.in_stats_table:
            return

        if tag == "tr":
            self.in_row = True
            self.row = {}
            return

        if not self.in_row:
            return

        if tag in {"th", "td"}:
            data_stat = attr.get("data-stat")
            if data_stat:
                self.current_field = data_stat
                self.current_text = []
                self.current_href = None
            return

        if tag == "a" and self.current_field is not None and self.current_href is None:
            self.current_href = attr.get("href")

    def handle_data(self, data: str) -> None:
        if self.current_field is not None:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_stats_table:
            return

        if tag in {"th", "td"} and self.current_field is not None:
            text = "".join(self.current_text).strip()
            self.row[self.current_field] = text
            if self.current_href:
                self.row[f"{self.current_field}_href"] = self.current_href
            self.current_field = None
            self.current_text = []
            self.current_href = None
            return

        if tag == "tr" and self.in_row:
            if self.row:
                self.rows.append(self.row)
            self.in_row = False
            self.row = {}
            return

        if tag == "table":
            self.in_stats_table = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NBA draft history from Basketball-Reference.")
    parser.add_argument("--start-year", type=int, default=2015, help="First draft year (inclusive).")
    parser.add_argument("--end-year", type=int, default=2025, help="Last draft year (inclusive).")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between requests (seconds).")
    parser.add_argument("--retries", type=int, default=3, help="Request retries per year.")
    parser.add_argument("--timeout", type=float, default=25.0, help="Request timeout (seconds).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/nba/draft/nba_draft_history_2015_2025.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def _to_int(val: object) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _extract_bref_id(href: object) -> Optional[str]:
    if href is None:
        return None
    s = str(href)
    m = re.search(r"/players/[a-z]/([a-z0-9]+)\.html", s)
    if not m:
        return None
    return m.group(1)


def fetch_year_html(year: int, timeout: float) -> str:
    url = f"https://www.basketball-reference.com/draft/NBA_{year}.html"
    req = Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; BB-Draft-Downloader/1.0; +https://example.local)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_year(year: int, html_text: str) -> pd.DataFrame:
    parser = DraftTableParser()
    parser.feed(html_text)
    rows = parser.rows
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "pick_overall" not in df.columns or "player" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "draft_year": year,
            "pick_overall": df["pick_overall"].map(_to_int),
            "pick_round": df.get("round", pd.Series([None] * len(df))).map(_to_int),
            "team_abbr": df.get("team_id"),
            "player_name": df.get("player"),
            "college_name": df.get("college_name"),
            "player_bref_id": df.get("player_href", pd.Series([None] * len(df))).map(_extract_bref_id),
        }
    )

    out["player_name"] = out["player_name"].fillna("").astype(str).str.strip()
    out = out[out["pick_overall"].notna() & (out["pick_overall"] > 0) & (out["player_name"] != "")]
    out["pick_overall"] = out["pick_overall"].astype(int)
    out["pick_round"] = pd.to_numeric(out["pick_round"], errors="coerce").astype("Int64")
    # Some pages omit explicit round; infer from overall pick.
    inferred_round = (out["pick_overall"] > 30).astype(int) + 1
    out["pick_round"] = out["pick_round"].fillna(inferred_round).astype("Int64")
    out = out.sort_values(["pick_overall", "player_name"], kind="stable")
    out = out.drop_duplicates(subset=["draft_year", "pick_overall"], keep="first")
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be >= --start-year")

    all_rows: list[pd.DataFrame] = []
    years = list(range(args.start_year, args.end_year + 1))
    for year in years:
        last_err: Optional[Exception] = None
        year_df = pd.DataFrame()
        for attempt in range(1, args.retries + 1):
            try:
                html_text = fetch_year_html(year=year, timeout=args.timeout)
                year_df = parse_year(year=year, html_text=html_text)
                if year_df.empty:
                    raise RuntimeError(f"No draft rows parsed for {year}")
                break
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                last_err = exc
                if attempt < args.retries:
                    time.sleep(min(2.0, args.sleep + 0.35 * attempt))
        if year_df.empty:
            msg = f"Failed year {year}"
            if last_err is not None:
                msg += f": {last_err}"
            raise SystemExit(msg)

        all_rows.append(year_df)
        print(f"{year}: {len(year_df):,} picks")
        time.sleep(max(0.0, args.sleep))

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["draft_year", "pick_overall"], kind="stable").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out):,} total picks)")


if __name__ == "__main__":
    main()
