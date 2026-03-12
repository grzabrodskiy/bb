from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


ESPN_ATHLETE_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
    "mens-college-basketball/athletes/{athlete_source_id}?lang=en&region=us"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download free ESPN athlete bio fields (height/weight/etc.) for CBBD players."
    )
    parser.add_argument("--season", type=int, required=True, help="Target season (e.g., 2025).")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional input players CSV. Defaults to data/raw/cbbd/players/season=YYYY/player_season_stats.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV. Defaults to data/raw/espn/player_bios/season=YYYY/player_bios.csv",
    )
    parser.add_argument(
        "--workers", type=int, default=12, help="Parallel workers for ESPN API calls (default: 12)."
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=0,
        help="Optional cap on players to fetch this run (0 = all pending).",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries per athlete on transient failures."
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=0.6,
        help="Base retry backoff in seconds (multiplied by attempt number).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip athlete_source_ids already present in output CSV.",
    )
    return parser.parse_args()


def _default_input_path(season: int) -> Path:
    return Path("data/raw/cbbd/players") / f"season={season}" / "player_season_stats.csv"


def _default_output_path(season: int) -> Path:
    return Path("data/raw/espn/player_bios") / f"season={season}" / "player_bios.csv"


def _safe_num(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x):
        return None
    return x


def _flatten_birth_place(obj: Any) -> tuple[str | None, str | None, str | None]:
    if not isinstance(obj, dict):
        return (None, None, None)
    return (
        obj.get("city"),
        obj.get("state"),
        obj.get("country"),
    )


def _extract_fields(payload: dict[str, Any]) -> dict[str, Any]:
    pos = payload.get("position")
    if isinstance(pos, dict):
        position_name = pos.get("displayName") or pos.get("name")
        position_abbr = pos.get("abbreviation")
    else:
        position_name = None
        position_abbr = None

    city, state, country = _flatten_birth_place(payload.get("birthPlace"))
    height_in = _safe_num(payload.get("height"))
    weight_lb = _safe_num(payload.get("weight"))

    return {
        "espn_id": str(payload.get("id") or ""),
        "espn_uid": payload.get("uid"),
        "espn_guid": payload.get("guid"),
        "display_name": payload.get("displayName"),
        "full_name": payload.get("fullName"),
        "short_name": payload.get("shortName"),
        "first_name": payload.get("firstName"),
        "last_name": payload.get("lastName"),
        "slug": payload.get("slug"),
        "jersey": payload.get("jersey"),
        "display_height": payload.get("displayHeight"),
        "display_weight": payload.get("displayWeight"),
        "height_in": height_in,
        "weight_lb": weight_lb,
        "height_cm": (height_in * 2.54) if height_in is not None else None,
        "weight_kg": (weight_lb * 0.45359237) if weight_lb is not None else None,
        "age": payload.get("age"),
        "date_of_birth": payload.get("dateOfBirth"),
        "position_name": position_name,
        "position_abbr": position_abbr,
        "birth_city": city,
        "birth_state": state,
        "birth_country": country,
    }


def _fetch_athlete(athlete_source_id: str, timeout: float, retries: int, retry_sleep: float) -> dict[str, Any]:
    url = ESPN_ATHLETE_URL.format(athlete_source_id=athlete_source_id)
    last_err = ""
    for attempt in range(1, retries + 2):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            out = _extract_fields(payload)
            out["status"] = "ok"
            out["http_status"] = 200
            out["error"] = None
            return out
        except urllib.error.HTTPError as e:
            body = e.read(200).decode("utf-8", errors="replace")
            last_err = f"HTTP {e.code}: {body[:160]}"
            # Retry on transient/rate-limit responses only.
            if e.code not in (429, 500, 502, 503, 504) or attempt > retries:
                return {
                    "status": "error",
                    "http_status": e.code,
                    "error": last_err,
                }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            if attempt > retries:
                return {
                    "status": "error",
                    "http_status": None,
                    "error": last_err,
                }

        time.sleep(retry_sleep * attempt)

    return {
        "status": "error",
        "http_status": None,
        "error": last_err or "unknown_error",
    }


def _load_candidates(input_csv: Path) -> pd.DataFrame:
    usecols = [
        "athlete_id",
        "athlete_source_id",
        "name",
        "team",
        "conference",
        "position",
        "minutes",
    ]
    df = pd.read_csv(input_csv, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["athlete_source_id"]).copy()
    df["athlete_source_id"] = df["athlete_source_id"].astype(str).str.strip()
    df["athlete_id"] = df["athlete_id"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["team"] = df["team"].fillna("").astype(str)
    df["conference"] = df["conference"].fillna("").astype(str)
    df["position"] = df["position"].fillna("").astype(str)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    # Keep the most-played row per source id for stable metadata.
    df = df.sort_values(["athlete_source_id", "minutes"], ascending=[True, False]).drop_duplicates(
        subset=["athlete_source_id"], keep="first"
    )
    return df


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or _default_input_path(args.season)
    out_csv = args.out_csv or _default_output_path(args.season)

    if not input_csv.exists():
        raise SystemExit(f"Input not found: {input_csv}")

    candidates = _load_candidates(input_csv)
    print(f"Loaded candidate players: {len(candidates)} from {input_csv}")

    existing = pd.DataFrame()
    done_ids: set[str] = set()
    if args.skip_existing and out_csv.exists():
        existing = pd.read_csv(out_csv, low_memory=False)
        if "athlete_source_id" in existing.columns:
            done_ids = set(existing["athlete_source_id"].dropna().astype(str))
        print(f"Existing output rows: {len(existing)} (skip set size: {len(done_ids)})")

    pending = candidates[~candidates["athlete_source_id"].isin(done_ids)].copy()
    if args.max_players and args.max_players > 0:
        pending = pending.head(args.max_players).copy()
    print(f"Pending downloads: {len(pending)}")

    if pending.empty:
        print("Nothing to download.")
        return

    meta_cols = ["athlete_id", "athlete_source_id", "name", "team", "conference", "position"]
    rows: list[dict[str, Any]] = []

    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_to_meta = {
            pool.submit(
                _fetch_athlete,
                str(row.athlete_source_id),
                args.timeout,
                int(args.retries),
                float(args.retry_sleep),
            ): row
            for row in pending[meta_cols].itertuples(index=False)
        }
        total = len(fut_to_meta)
        done = 0
        ok = 0
        for fut in as_completed(fut_to_meta):
            meta = fut_to_meta[fut]
            result = fut.result()
            rec = {
                "season": args.season,
                "athlete_id": str(meta.athlete_id),
                "athlete_source_id": str(meta.athlete_source_id),
                "name": meta.name,
                "team": meta.team,
                "conference": meta.conference,
                "position_cbbd": meta.position,
            }
            rec.update(result)
            rows.append(rec)
            done += 1
            if result.get("status") == "ok":
                ok += 1
            if done % 250 == 0 or done == total:
                print(f"Progress: {done}/{total} complete | ok={ok} | errors={done - ok}")

    new_df = pd.DataFrame(rows)
    out_df = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    out_df["athlete_source_id"] = out_df["athlete_source_id"].astype(str)
    out_df = out_df.drop_duplicates(subset=["athlete_source_id"], keep="last")
    out_df = out_df.sort_values("athlete_source_id").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    ok_final = int((out_df.get("status") == "ok").sum()) if "status" in out_df.columns else 0
    print(f"Wrote {out_csv}")
    print(f"Output rows: {len(out_df)} | ok rows: {ok_final} | non-ok rows: {len(out_df) - ok_final}")


if __name__ == "__main__":
    main()
