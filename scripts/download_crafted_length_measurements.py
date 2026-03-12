from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path

import pandas as pd


CRAFTED_LENGTH_URL = "https://craftednba.com/player-traits/length"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CraftedNBA Height/Wingspan table for measurement fallback coverage."
    )
    p.add_argument("--url", type=str, default=CRAFTED_LENGTH_URL)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/raw/external/craftednba/player_traits_length.csv"),
    )
    p.add_argument("--timeout-sec", type=float, default=45.0)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if output CSV exists and is non-empty.",
    )
    return p.parse_args()


def _name_key(v: object) -> str:
    s = "" if v is None else str(v).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _strip_tags(v: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", v)).strip()


def _parse_inches(v: str) -> float:
    s = (v or "").strip()
    m = re.search(r"(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*\"", s)
    if not m:
        return float("nan")
    ft = float(m.group(1))
    inch = float(m.group(2))
    return ft * 12.0 + inch


def fetch_html(url: str, timeout_sec: float) -> str:
    cmd = ["curl", "-L", "-s", "-A", "Mozilla/5.0", url]
    p = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    return p.stdout


def parse_rows(page: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for m in re.finditer(r"<tr[^>]*>(.*?)</tr>", page, flags=re.IGNORECASE | re.DOTALL):
        tr = m.group(1)
        if "/players/" not in tr:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.IGNORECASE | re.DOTALL)
        if len(tds) < 5:
            continue

        anchor = re.search(r'<a[^>]*href="(/players/[^"]+)"[^>]*>(.*?)</a>', tds[1], flags=re.IGNORECASE | re.DOTALL)
        if not anchor:
            continue
        player_href = anchor.group(1).strip()
        player_slug = player_href.split("/players/")[-1].strip("/")
        player_name = _strip_tags(anchor.group(2))
        if not player_name:
            continue

        height_txt = _strip_tags(tds[2])
        wingspan_txt = _strip_tags(tds[3])
        length_txt = _strip_tags(tds[4])

        rows.append(
            {
                "player_name": player_name,
                "name_key": _name_key(player_name),
                "crafted_slug": player_slug,
                "crafted_height_text": height_txt,
                "crafted_wingspan_text": wingspan_txt,
                "crafted_length_text": length_txt,
                "crafted_height_in": _parse_inches(height_txt),
                "crafted_wingspan_in": _parse_inches(wingspan_txt),
                "crafted_length_in": pd.to_numeric(pd.Series([length_txt]), errors="coerce").iloc[0],
                "source_url": CRAFTED_LENGTH_URL,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "player_name",
                "name_key",
                "crafted_slug",
                "crafted_height_text",
                "crafted_wingspan_text",
                "crafted_length_text",
                "crafted_height_in",
                "crafted_wingspan_in",
                "crafted_length_in",
                "source_url",
            ]
        )

    out = pd.DataFrame(rows)
    out = out[out["name_key"] != ""].copy()
    out = out.sort_values(["name_key", "player_name"], kind="stable")
    out = out.drop_duplicates(subset=["name_key"], keep="first")
    return out


def main() -> None:
    args = parse_args()
    if args.skip_existing and args.out_csv.exists() and args.out_csv.stat().st_size > 0:
        print(f"[skip] existing: {args.out_csv}")
        return

    page = fetch_html(args.url, args.timeout_sec)
    df = parse_rows(page)
    if df.empty:
        raise SystemExit("No player rows parsed from Crafted length page.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[ok] rows={len(df):,} -> {args.out_csv}")


if __name__ == "__main__":
    main()

