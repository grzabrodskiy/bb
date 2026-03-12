from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.cbbd import (
    CbbdClient,
    export_games,
    export_lineups_by_team_season,
    export_player_season_stats,
    export_plays_by_team,
)


def team_slug(team_name: str) -> str:
    return team_name.lower().replace(" ", "_")


def plays_out_path(out_dir: Path, season: int, team_name: str) -> Path:
    return out_dir / "plays" / f"season={season}" / f"team={team_slug(team_name)}" / "plays.csv"


def lineups_out_path(out_dir: Path, season: int, team_name: str) -> Path:
    return out_dir / "lineups" / f"season={season}" / f"team={team_slug(team_name)}" / "lineups.csv"


def find_existing_nonempty(path: Path) -> Path | None:
    candidates = [path]
    if path.suffix == ".csv":
        candidates.append(path.parent / f"{path.name}.gz")
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download data from CollegeBasketballData.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--team", type=str, default=None)
    parser.add_argument("--shooting-only", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/cbbd"),
        help="Output directory for raw CBBD data",
    )
    parser.add_argument(
        "--games",
        action="store_true",
        help="Also download the season games list",
    )
    parser.add_argument(
        "--players",
        action="store_true",
        help="Download player season stats for the season",
    )
    parser.add_argument(
        "--lineups",
        action="store_true",
        help="Download lineup stats (requires --team or --lineups-all)",
    )
    parser.add_argument(
        "--lineups-all",
        action="store_true",
        help="Download lineup stats for all teams in the season",
    )
    parser.add_argument(
        "--plays-all",
        action="store_true",
        help="Download play-by-play for all teams in the season",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between team requests for all-team downloads",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip all-team downloads when output file already exists and is non-empty",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    if args.games:
        path = export_games(season=args.season, out_dir=out_dir)
        print(f"Wrote {path}")

    if args.players:
        path = export_player_season_stats(season=args.season, out_dir=out_dir)
        print(f"Wrote {path}")

    if args.team:
        target_path = plays_out_path(out_dir=out_dir, season=args.season, team_name=args.team)
        existing = find_existing_nonempty(target_path)
        if args.skip_existing and existing is not None:
            print(f"Skipping existing team={args.team}: {existing}")
        else:
            path = export_plays_by_team(
                season=args.season,
                team=args.team,
                shooting_only=args.shooting_only,
                out_dir=out_dir,
            )
            print(f"Wrote {path}")
    elif not args.games and not args.players and not args.lineups and not args.lineups_all and not args.plays_all:
        raise SystemExit(
            "No action taken. Provide --team and/or --games and/or --players and/or --lineups."
        )

    if args.lineups:
        if not args.team:
            raise SystemExit("--lineups requires --team or use --lineups-all.")
        target_path = lineups_out_path(out_dir=out_dir, season=args.season, team_name=args.team)
        existing = find_existing_nonempty(target_path)
        if args.skip_existing and existing is not None:
            print(f"Skipping existing team={args.team}: {existing}")
        else:
            path = export_lineups_by_team_season(
                season=args.season,
                team=args.team,
                out_dir=out_dir,
            )
            print(f"Wrote {path}")

    if args.lineups_all:
        client = CbbdClient()
        teams = client.get_teams(season=args.season)
        for t in teams:
            team_name = getattr(t, "name", None) or getattr(t, "school", None) or str(t)
            out_path = lineups_out_path(out_dir=out_dir, season=args.season, team_name=team_name)
            existing = find_existing_nonempty(out_path)
            if args.skip_existing and existing is not None:
                print(f"Skipping existing team={team_name}: {existing}")
                if args.sleep:
                    time.sleep(args.sleep)
                continue
            try:
                path = export_lineups_by_team_season(
                    season=args.season,
                    team=team_name,
                    out_dir=out_dir,
                )
                print(f"Wrote {path}")
            except Exception as exc:
                print(f"Failed team={team_name}: {exc}")
            if args.sleep:
                time.sleep(args.sleep)

    if args.plays_all:
        client = CbbdClient()
        teams = client.get_teams(season=args.season)
        for t in teams:
            team_name = getattr(t, "name", None) or getattr(t, "school", None) or str(t)
            out_path = plays_out_path(out_dir=out_dir, season=args.season, team_name=team_name)
            existing = find_existing_nonempty(out_path)
            if args.skip_existing and existing is not None:
                print(f"Skipping existing team={team_name}: {existing}")
                if args.sleep:
                    time.sleep(args.sleep)
                continue
            try:
                path = export_plays_by_team(
                    season=args.season,
                    team=team_name,
                    shooting_only=args.shooting_only,
                    out_dir=out_dir,
                )
                print(f"Wrote {path}")
            except Exception as exc:
                print(f"Failed team={team_name}: {exc}")
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
