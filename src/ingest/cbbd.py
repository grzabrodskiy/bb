from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

import cbbd

from src.utils.io import to_records, write_csv


class CbbdClient:
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("CBBD_API_KEY")
        if not api_key:
            raise RuntimeError("Missing CBBD_API_KEY in environment.")
        self.config = cbbd.Configuration(access_token=api_key)

    def _client(self) -> cbbd.ApiClient:
        return cbbd.ApiClient(self.config)

    def get_games(self, season: int) -> Iterable[object]:
        with self._client() as api_client:
            api = cbbd.GamesApi(api_client)
            return api.get_games(season=season)

    def get_plays_by_team(
        self,
        season: int,
        team: str,
        shooting_only: bool = False,
    ) -> Iterable[object]:
        with self._client() as api_client:
            api = cbbd.PlaysApi(api_client)
            import inspect

            kwargs = {"season": season, "team": team}
            # Some CBBD versions don't accept shooting_only.
            try:
                params = inspect.signature(api.get_plays_by_team).parameters
                if "shooting_only" in params:
                    kwargs["shooting_only"] = shooting_only
            except (ValueError, TypeError):
                pass
            return api.get_plays_by_team(**kwargs)

    def get_teams(self, season: int) -> Iterable[object]:
        with self._client() as api_client:
            api = cbbd.TeamsApi(api_client)
            return api.get_teams(season=season)

    def get_lineups_by_team_season(self, season: int, team: str) -> Iterable[object]:
        with self._client() as api_client:
            api = cbbd.LineupsApi(api_client)
            try:
                return api.get_lineups_by_team_season(season=season, team=team)
            except Exception:
                # Fall back to raw JSON to avoid model validation errors on nulls.
                query_params = [("season", season), ("team", team)]
                return api_client.call_api(
                    "/lineups/team",
                    "GET",
                    {},
                    query_params,
                    {"Accept": "application/json"},
                    response_types_map={"200": "object"},
                    auth_settings=["apiKey"],
                    _return_http_data_only=True,
                )

    def get_player_season_stats(self, season: int) -> Iterable[object]:
        with self._client() as api_client:
            api = cbbd.StatsApi(api_client)
            return api.get_player_season_stats(season=season)


def export_games(season: int, out_dir: Path) -> Path:
    client = CbbdClient()
    games = client.get_games(season=season)
    records = to_records(games)
    out_path = out_dir / "games" / f"season={season}" / "games.csv"
    write_csv(records, out_path)
    return out_path


def export_plays_by_team(
    season: int,
    team: str,
    shooting_only: bool,
    out_dir: Path,
) -> Path:
    client = CbbdClient()
    plays = client.get_plays_by_team(
        season=season,
        team=team,
        shooting_only=shooting_only,
    )
    records = to_records(plays)
    safe_team = team.lower().replace(" ", "_")
    out_path = (
        out_dir
        / "plays"
        / f"season={season}"
        / f"team={safe_team}"
        / "plays.csv"
    )
    write_csv(records, out_path)
    return out_path


def export_player_season_stats(season: int, out_dir: Path) -> Path:
    client = CbbdClient()
    players = client.get_player_season_stats(season=season)
    records = to_records(players)
    out_path = out_dir / "players" / f"season={season}" / "player_season_stats.csv"
    write_csv(records, out_path)
    return out_path


def export_lineups_by_team_season(season: int, team: str, out_dir: Path) -> Path:
    client = CbbdClient()
    lineups = client.get_lineups_by_team_season(season=season, team=team)
    records = to_records(lineups)
    safe_team = team.lower().replace(" ", "_")
    out_path = (
        out_dir
        / "lineups"
        / f"season={season}"
        / f"team={safe_team}"
        / "lineups.csv"
    )
    write_csv(records, out_path)
    return out_path
