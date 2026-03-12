from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_ENDPOINTS = [
    "TimedecayRAPM",
    "current_comp",
    "player_stats_export",
    "mamba",
    "DARKO",
    "lebron",
    "raptor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download free NBA datasets from nbarapm.com")
    parser.add_argument(
        "--endpoints",
        type=str,
        default=",".join(DEFAULT_ENDPOINTS),
        help="Comma-separated endpoint names for /load/{endpoint}.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/external/nbarapm"),
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def fetch_json(endpoint: str, timeout: float, retries: int) -> object:
    url = f"https://www.nbarapm.com/load/{endpoint}"
    err: Exception | None = None
    for i in range(retries + 1):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            with urlopen(req, timeout=timeout) as r:
                raw = r.read()
            return json.loads(raw)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            err = e
            if i >= retries:
                break
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"Failed {url}: {err}")


def to_csv_if_tabular(obj: object, out_csv: Path) -> bool:
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        df = pd.DataFrame(obj)
        df.to_csv(out_csv, index=False)
        return True
    return False


def main() -> None:
    args = parse_args()
    endpoints = [x.strip() for x in args.endpoints.split(",") if x.strip()]
    if not endpoints:
        raise SystemExit("No endpoints provided.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest: list[dict[str, object]] = []

    for ep in endpoints:
        print(f"Downloading endpoint={ep} ...")
        obj = fetch_json(ep, timeout=args.timeout, retries=args.retries)

        out_json = args.out_dir / f"{ep}.json"
        out_json.write_text(json.dumps(obj, ensure_ascii=True), encoding="utf-8")

        row_count: int | None = None
        n_cols: int | None = None
        csv_written = False
        if isinstance(obj, list):
            row_count = len(obj)
            if obj and isinstance(obj[0], dict):
                n_cols = len(obj[0].keys())
            out_csv = args.out_dir / f"{ep}.csv"
            csv_written = to_csv_if_tabular(obj, out_csv)

        manifest.append(
            {
                "endpoint": ep,
                "fetched_at_utc": stamp,
                "json_path": str(out_json),
                "csv_path": str(args.out_dir / f"{ep}.csv") if csv_written else "",
                "row_count": row_count if row_count is not None else "",
                "n_columns": n_cols if n_cols is not None else "",
            }
        )
        print(
            f"  wrote {out_json}"
            + (f" and {(args.out_dir / f'{ep}.csv')}" if csv_written else "")
            + (f" | rows={row_count}" if row_count is not None else "")
        )

    man_path = args.out_dir / "manifest.csv"
    pd.DataFrame(manifest).to_csv(man_path, index=False)
    print(f"Wrote manifest: {man_path}")


if __name__ == "__main__":
    main()
