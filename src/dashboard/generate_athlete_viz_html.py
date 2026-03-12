from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


METRICS = [
    "rapm_hca",
    "rapm_cv",
    "rapm_bayes_box",
    "rapm_close",
    "rapm_eb",
    "orapm",
    "drapm",
    "box_prior",
    "possessions",
]


METRIC_LABELS = {
    "rapm_hca": "RAPM (Home-Court Adj.)",
    "rapm_cv": "RAPM (CV)",
    "rapm_bayes_box": "RAPM (Bayes + Box Prior)",
    "rapm_close": "RAPM (Close Games)",
    "rapm_eb": "RAPM (EB Shrink)",
    "orapm": "Offensive RAPM",
    "drapm": "Defensive RAPM",
    "box_prior": "Box Prior",
    "possessions": "Possessions",
}

EXTRA_STAT_LABELS = {
    "games": "Games",
    "minutes": "Minutes",
    "minutes_per_game": "Min/Game",
    "points": "Points",
    "rebounds_total": "Rebounds",
    "assists": "Assists",
    "steals": "Steals",
    "blocks": "Blocks",
    "turnovers": "Turnovers",
    "points_per40": "Pts/40",
    "rebounds_total_per40": "Reb/40",
    "assists_per40": "Ast/40",
    "steals_per40": "Stl/40",
    "blocks_per40": "Blk/40",
    "turnovers_per40": "TO/40",
    "true_shooting_pct": "TS%",
    "field_goals_pct": "FG%",
    "three_point_field_goals_pct": "3P%",
    "free_throws_pct": "FT%",
    "usage": "Usage",
    "offensive_rating": "Off Rating",
    "defensive_rating": "Def Rating",
    "net_rating": "Net Rating",
    "porpag": "PORPAG",
    "win_shares_total": "Win Shares",
    "win_shares_offensive": "WS Off",
    "win_shares_defensive": "WS Def",
}

BIO_STAT_LABELS = {
    "bio_position": "Position",
    "bio_height": "Height",
    "bio_weight": "Weight",
    "bio_age": "Age",
    "bio_dob": "DOB",
    "bio_birth_place": "Birthplace",
    "bio_jersey": "Jersey",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate standalone HTML visualizations for athlete model outputs.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input model CSV (e.g., rapm_variants_season_2025_freshmen.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/viz"),
        help="Directory for generated HTML files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Default top N shown in leaderboard-focused views.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season for player stat enrichment. If omitted, inferred from input filename.",
    )
    return parser.parse_args()


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"player_id", "player_name", "player_team", "player_conference"} | set(METRICS)
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in input CSV: {missing}")


def _prepare_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for m in METRICS:
        out[m] = pd.to_numeric(out[m], errors="coerce").fillna(0.0)

    out["player_name"] = out["player_name"].fillna("").astype(str)
    out["player_team"] = out["player_team"].fillna("").astype(str)
    out["player_conference"] = out["player_conference"].fillna("").astype(str)
    out["player_id"] = out["player_id"].fillna("").astype(str)

    for m in METRICS:
        out[f"rank_{m}"] = out[m].rank(ascending=False, method="min").astype(int)
        out[f"pct_{m}"] = (100.0 * out[m].rank(pct=True, method="average")).round(2)

    extra_cols = [c for c in EXTRA_STAT_LABELS if c in out.columns]
    bio_cols = [c for c in BIO_STAT_LABELS if c in out.columns]
    cols = [
        "player_id",
        "player_name",
        "player_team",
        "player_conference",
        *METRICS,
        *extra_cols,
        *bio_cols,
        *[f"rank_{m}" for m in METRICS],
        *[f"pct_{m}" for m in METRICS],
    ]
    return out[cols].to_dict(orient="records")


def _infer_season(input_csv: Path) -> int | None:
    m = re.search(r"season_(\d{4})", input_csv.stem)
    if m:
        return int(m.group(1))
    return None


def _safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def enrich_with_player_stats(df: pd.DataFrame, season: int | None) -> pd.DataFrame:
    if season is None:
        return df

    path = Path("data/raw/cbbd/players") / f"season={season}" / "player_season_stats.csv"
    if not path.exists():
        return df

    wanted_cols = [
        "athlete_id",
        "name",
        "team",
        "conference",
        "games",
        "minutes",
        "points",
        "assists",
        "turnovers",
        "steals",
        "blocks",
        "rebounds.total",
        "true_shooting_pct",
        "field_goals.pct",
        "three_point_field_goals.pct",
        "free_throws.pct",
        "usage",
        "offensive_rating",
        "defensive_rating",
        "net_rating",
        "porpag",
        "win_shares.total",
        "win_shares.offensive",
        "win_shares.defensive",
    ]
    available_cols = set(pd.read_csv(path, nrows=0).columns.tolist())
    usecols = [c for c in wanted_cols if c in available_cols]
    if "athlete_id" not in usecols:
        return df
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    if raw.empty:
        return df

    raw["athlete_id"] = raw["athlete_id"].astype(str)
    raw["_minutes_w"] = pd.to_numeric(raw["minutes"], errors="coerce").fillna(0.0)

    sum_cols = [
        "games",
        "minutes",
        "points",
        "assists",
        "turnovers",
        "steals",
        "blocks",
        "rebounds.total",
        "win_shares.total",
        "win_shares.offensive",
        "win_shares.defensive",
    ]
    weighted_cols = [
        "true_shooting_pct",
        "field_goals.pct",
        "three_point_field_goals.pct",
        "free_throws.pct",
        "usage",
        "offensive_rating",
        "defensive_rating",
        "net_rating",
        "porpag",
    ]
    sum_cols = [c for c in sum_cols if c in raw.columns]
    weighted_cols = [c for c in weighted_cols if c in raw.columns]

    rows: list[dict] = []
    for pid, g in raw.groupby("athlete_id", sort=False):
        top = g.sort_values("_minutes_w", ascending=False).iloc[0]
        out: dict[str, object] = {
            "player_id": str(pid),
            "player_name": top.get("name"),
            "player_team": top.get("team"),
            "player_conference": top.get("conference"),
        }
        for c in sum_cols:
            out[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0).sum()
        for c in weighted_cols:
            out[c] = _safe_weighted_mean(g[c], g["_minutes_w"])
        rows.append(out)

    box = pd.DataFrame(rows)
    if box.empty:
        return df

    box = box.rename(
        columns={
            "rebounds.total": "rebounds_total",
            "field_goals.pct": "field_goals_pct",
            "three_point_field_goals.pct": "three_point_field_goals_pct",
            "free_throws.pct": "free_throws_pct",
            "win_shares.total": "win_shares_total",
            "win_shares.offensive": "win_shares_offensive",
            "win_shares.defensive": "win_shares_defensive",
        }
    )
    if "minutes" in box.columns:
        for c in ["points", "rebounds_total", "assists", "steals", "blocks", "turnovers"]:
            if c in box.columns:
                box[f"{c}_per40"] = 40.0 * pd.to_numeric(box[c], errors="coerce") / pd.to_numeric(
                    box["minutes"], errors="coerce"
                ).replace(0, pd.NA)
    if "minutes" in box.columns and "games" in box.columns:
        box["minutes_per_game"] = pd.to_numeric(box["minutes"], errors="coerce") / pd.to_numeric(
            box["games"], errors="coerce"
        ).replace(0, pd.NA)

    merged = df.copy()
    merged["player_id"] = merged["player_id"].astype(str)
    merged = merged.merge(box, how="left", on="player_id", suffixes=("", "_stat"))

    for c in ["player_name", "player_team", "player_conference"]:
        alt = f"{c}_stat"
        if alt in merged.columns:
            merged[c] = merged[c].fillna(merged[alt])
            merged = merged.drop(columns=[alt])

    return merged


def enrich_with_espn_bios(df: pd.DataFrame, season: int | None) -> pd.DataFrame:
    if season is None:
        return df

    path = Path("data/raw/espn/player_bios") / f"season={season}" / "player_bios.csv"
    if not path.exists():
        return df

    wanted_cols = [
        "athlete_id",
        "status",
        "position_name",
        "display_height",
        "display_weight",
        "age",
        "date_of_birth",
        "birth_city",
        "birth_state",
        "birth_country",
        "jersey",
    ]
    available = set(pd.read_csv(path, nrows=0).columns.tolist())
    usecols = [c for c in wanted_cols if c in available]
    if "athlete_id" not in usecols:
        return df

    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    if raw.empty:
        return df

    raw["athlete_id"] = raw["athlete_id"].astype(str)

    if "status" in raw.columns:
        raw["_ok"] = (raw["status"].astype(str).str.lower() == "ok").astype(int)
    else:
        raw["_ok"] = 0

    completeness_cols = [
        c
        for c in [
            "position_name",
            "display_height",
            "display_weight",
            "age",
            "date_of_birth",
            "birth_city",
            "birth_state",
            "birth_country",
            "jersey",
        ]
        if c in raw.columns
    ]
    if completeness_cols:
        raw["_filled"] = raw[completeness_cols].notna().sum(axis=1)
    else:
        raw["_filled"] = 0

    raw = raw.sort_values(["athlete_id", "_ok", "_filled"], ascending=[True, False, False]).drop_duplicates(
        subset=["athlete_id"], keep="first"
    )

    parts = []
    for col in ["birth_city", "birth_state", "birth_country"]:
        if col in raw.columns:
            parts.append(raw[col].fillna("").astype(str).str.strip())
        else:
            parts.append(pd.Series([""] * len(raw), index=raw.index))
    birth = (parts[0] + ", " + parts[1] + ", " + parts[2]).str.replace(r"(,\s*){2,}", ", ", regex=True)
    birth = birth.str.strip(", ").replace("", pd.NA)

    bio = pd.DataFrame(
        {
            "player_id": raw["athlete_id"].astype(str),
            "bio_position": raw["position_name"] if "position_name" in raw.columns else pd.NA,
            "bio_height": raw["display_height"] if "display_height" in raw.columns else pd.NA,
            "bio_weight": raw["display_weight"] if "display_weight" in raw.columns else pd.NA,
            "bio_age": pd.to_numeric(raw["age"], errors="coerce") if "age" in raw.columns else pd.NA,
            "bio_dob": (
                pd.to_datetime(raw["date_of_birth"], errors="coerce").dt.date.astype(str).replace("NaT", pd.NA)
                if "date_of_birth" in raw.columns
                else pd.NA
            ),
            "bio_birth_place": birth,
            "bio_jersey": raw["jersey"] if "jersey" in raw.columns else pd.NA,
        }
    )
    bio["bio_jersey"] = bio["bio_jersey"].fillna("").astype(str).str.strip().replace("", pd.NA)

    merged = df.copy()
    merged["player_id"] = merged["player_id"].astype(str)
    return merged.merge(bio, how="left", on="player_id")


def _html_shell(title: str, body: str, script: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
    :root {{
      --bg-0: #f4efe6;
      --bg-1: #e7dcc9;
      --ink: #132a2f;
      --muted: #5e6f71;
      --card: rgba(255, 252, 245, 0.78);
      --stroke: rgba(22, 55, 58, 0.18);
      --teal: #0b7a75;
      --amber: #c77f2b;
      --rose: #b14e41;
      --lime: #4a8f41;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 500px at 10% -10%, rgba(11, 122, 117, 0.15), transparent 60%),
        radial-gradient(900px 450px at 90% -20%, rgba(199, 127, 43, 0.18), transparent 58%),
        linear-gradient(165deg, var(--bg-0) 0%, var(--bg-1) 100%);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .top-nav {{
      margin-bottom: 12px;
    }}
    .back-link {{
      display: inline-block;
      text-decoration: none;
      font-size: 0.86rem;
      color: var(--ink);
      border: 1px solid var(--stroke);
      background: rgba(255, 255, 255, 0.72);
      padding: 7px 11px;
      border-radius: 10px;
      transition: background 0.15s ease, transform 0.15s ease;
    }}
    .back-link:hover {{
      background: rgba(11, 122, 117, 0.14);
      transform: translateY(-1px);
    }}
    .hero {{
      margin-bottom: 18px;
      border: 1px solid var(--stroke);
      background: var(--card);
      border-radius: 18px;
      padding: 18px 20px;
      backdrop-filter: blur(4px);
    }}
    .hero h1 {{
      margin: 0;
      font-family: "Fraunces", serif;
      font-size: clamp(1.5rem, 2.5vw, 2.2rem);
      letter-spacing: 0.01em;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    .card {{
      border: 1px solid var(--stroke);
      background: var(--card);
      border-radius: 16px;
      padding: 14px;
      backdrop-filter: blur(4px);
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .controls label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .controls input, .controls select {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--stroke);
      background: #fffdf8;
      color: var(--ink);
      font-family: inherit;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 12px;
    }}
    .grid-1 {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    thead th {{
      text-align: left;
      color: var(--muted);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 8px 6px;
      border-bottom: 1px solid var(--stroke);
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    tbody td {{
      padding: 8px 6px;
      border-bottom: 1px dashed rgba(19, 42, 47, 0.12);
      vertical-align: middle;
    }}
    tbody tr {{
      cursor: pointer;
      transition: background 0.2s ease;
    }}
    tbody tr:hover {{
      background: rgba(11, 122, 117, 0.08);
    }}
    .mono {{ font-variant-numeric: tabular-nums; }}
    .meter {{
      height: 8px;
      background: rgba(19, 42, 47, 0.08);
      border-radius: 999px;
      overflow: hidden;
      min-width: 120px;
    }}
    .meter > span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--teal), var(--amber));
      border-radius: inherit;
    }}
    .range-track {{
      position: relative;
      height: 10px;
      border-radius: 999px;
      background: rgba(19, 42, 47, 0.1);
      overflow: hidden;
      min-width: 140px;
    }}
    .range-span {{
      position: absolute;
      top: 2px;
      height: 6px;
      border-radius: 999px;
      background: rgba(11, 122, 117, 0.3);
    }}
    .range-dot {{
      position: absolute;
      top: 1px;
      width: 8px;
      height: 8px;
      border-radius: 999px;
      border: 1px solid rgba(19, 42, 47, 0.35);
      transform: translateX(-50%);
    }}
    .pill {{
      display: inline-block;
      font-size: 0.72rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--stroke);
      color: var(--muted);
      background: rgba(255, 255, 255, 0.7);
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 10px;
    }}
    .tab-btn {{
      appearance: none;
      border: 1px solid var(--stroke);
      background: rgba(255, 255, 255, 0.74);
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.78rem;
      font-family: inherit;
      cursor: pointer;
      transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    }}
    .tab-btn.active {{
      color: var(--ink);
      border-color: rgba(11, 122, 117, 0.5);
      background: rgba(11, 122, 117, 0.12);
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.active {{
      display: block;
    }}
    .kv {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 8px;
      margin-bottom: 7px;
    }}
    .kv .bar {{
      position: relative;
      height: 8px;
      background: rgba(19, 42, 47, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }}
    .kv .bar span {{
      position: absolute;
      inset: 0 auto 0 0;
      width: 0%;
      background: linear-gradient(90deg, var(--lime), var(--teal));
    }}
    .muted {{ color: var(--muted); }}
    .canvas-wrap {{
      border: 1px solid var(--stroke);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.72);
      overflow: hidden;
    }}
    .footer-note {{
      margin-top: 12px;
      font-size: 0.78rem;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top-nav">
      <a class="back-link" href="index.html">Back to Index</a>
    </div>
    {body}
  </div>
  <script>
  {script}
  </script>
</body>
</html>
"""


def build_v1(records_json: str, top_n: int) -> str:
    title = "Evidence Cards Leaderboard"
    body = """
<section class="hero">
  <h1>Evidence Cards</h1>
  <p>Leaderboard + player evidence panel. Click a player row to see why the model pushes them up.</p>
</section>
<section class="card">
  <div class="controls">
    <div>
      <label for="metric">Ranking metric</label>
      <select id="metric"></select>
    </div>
    <div>
      <label for="query">Search player/team</label>
      <input id="query" placeholder="e.g. Flagg, Duke" />
    </div>
    <div>
      <label for="conf">Conference filter</label>
      <select id="conf"></select>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <table>
        <thead>
          <tr><th>Rank</th><th>Player</th><th>Team</th><th>Value</th><th>Percentile</th></tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="card">
      <div id="detail-empty" class="muted">Select a row to inspect player evidence.</div>
      <div id="detail" style="display:none;">
        <h3 id="d-name" style="margin:0 0 4px;"></h3>
        <div id="d-meta" class="muted" style="margin-bottom:10px;"></div>
        <div id="d-tabs" class="tabs">
          <button type="button" class="tab-btn active" data-tab="stats">Stat Measures</button>
          <button type="button" class="tab-btn" data-tab="bio">Bio</button>
          <button type="button" class="tab-btn" data-tab="box">Box Score Context</button>
        </div>
        <div id="d-panel-stats" class="tab-panel active">
          <div id="d-tags" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;"></div>
          <div id="d-bars"></div>
          <div class="footer-note" id="d-why"></div>
        </div>
        <div id="d-panel-bio" class="tab-panel"></div>
        <div id="d-panel-box" class="tab-panel"></div>
      </div>
    </div>
  </div>
</section>
"""
    script = f"""
const DATA = {records_json};
const METRICS = {{
  rapm_hca: "RAPM (Home-Court Adj.)",
  rapm_cv: "RAPM (CV)",
  rapm_bayes_box: "RAPM (Bayes + Box Prior)",
  rapm_close: "RAPM (Close Games)",
  rapm_eb: "RAPM (EB Shrink)",
  orapm: "Offensive RAPM",
  drapm: "Defensive RAPM",
  box_prior: "Box Prior",
  possessions: "Possessions",
}};
const EXTRA = {{
  games: "Games",
  minutes: "Minutes",
  minutes_per_game: "Min/Game",
  points: "Points",
  rebounds_total: "Rebounds",
  assists: "Assists",
  steals: "Steals",
  blocks: "Blocks",
  turnovers: "Turnovers",
  points_per40: "Pts/40",
  rebounds_total_per40: "Reb/40",
  assists_per40: "Ast/40",
  steals_per40: "Stl/40",
  blocks_per40: "Blk/40",
  turnovers_per40: "TO/40",
  true_shooting_pct: "TS%",
  field_goals_pct: "FG%",
  three_point_field_goals_pct: "3P%",
  free_throws_pct: "FT%",
  usage: "Usage",
  offensive_rating: "Off Rating",
  defensive_rating: "Def Rating",
  net_rating: "Net Rating",
  porpag: "PORPAG",
  win_shares_total: "Win Shares",
  win_shares_offensive: "WS Off",
  win_shares_defensive: "WS Def",
  bio_position: "Position",
  bio_height: "Height",
  bio_weight: "Weight",
  bio_age: "Age",
  bio_dob: "DOB",
  bio_birth_place: "Birthplace",
  bio_jersey: "Jersey",
}};
const TOP_N = {int(top_n)};
const detailMetrics = ["rapm_hca","rapm_cv","rapm_bayes_box","rapm_close","orapm","drapm","box_prior","possessions"];
const detailBioOrder = ["bio_position","bio_height","bio_weight","bio_age","bio_dob","bio_birth_place","bio_jersey"];
const detailBoxOrder = ["games","minutes","minutes_per_game","points","rebounds_total","assists","steals","blocks","turnovers","points_per40","rebounds_total_per40","assists_per40","steals_per40","blocks_per40","turnovers_per40","true_shooting_pct","field_goals_pct","three_point_field_goals_pct","free_throws_pct","usage","offensive_rating","defensive_rating","net_rating","porpag","win_shares_total","win_shares_offensive","win_shares_defensive"];

const metricSel = document.getElementById("metric");
const query = document.getElementById("query");
const confSel = document.getElementById("conf");
const rowsEl = document.getElementById("rows");
const tabButtons = Array.from(document.querySelectorAll("#d-tabs .tab-btn"));
const panelStats = document.getElementById("d-panel-stats");
const panelBio = document.getElementById("d-panel-bio");
const panelBox = document.getElementById("d-panel-box");

Object.entries(METRICS).forEach(([k, v]) => {{
  const op = document.createElement("option");
  op.value = k;
  op.textContent = v;
  metricSel.appendChild(op);
}});
metricSel.value = "rapm_hca";

const confs = Array.from(new Set(DATA.map(d => d.player_conference || "Unknown"))).sort();
["All", ...confs].forEach(c => {{
  const op = document.createElement("option");
  op.value = c;
  op.textContent = c;
  confSel.appendChild(op);
}});

function rank(data, metric) {{
  return [...data].sort((a,b) => (b[metric] - a[metric]) || a.player_name.localeCompare(b.player_name));
}}

function filtered() {{
  const q = query.value.trim().toLowerCase();
  const c = confSel.value;
  return DATA.filter(d => {{
    const okQ = !q || d.player_name.toLowerCase().includes(q) || d.player_team.toLowerCase().includes(q);
    const okC = c === "All" || (d.player_conference || "Unknown") === c;
    return okQ && okC;
  }});
}}

function renderRows() {{
  const metric = metricSel.value;
  const arr = rank(filtered(), metric).slice(0, TOP_N);
  rowsEl.innerHTML = "";
  arr.forEach((d, i) => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${{i + 1}}</td>
      <td>${{d.player_name}}</td>
      <td>${{d.player_team}}</td>
      <td class="mono">${{d[metric].toFixed(3)}}</td>
      <td>
        <div class="meter"><span style="width:${{Math.max(0, Math.min(100, d["pct_" + metric]))}}%"></span></div>
      </td>`;
    tr.addEventListener("click", () => renderDetail(d));
    rowsEl.appendChild(tr);
  }});
}}

function fmtVal(key, val) {{
  if (val === null || val === undefined || Number.isNaN(val)) return "n/a";
  if (key === "bio_age" || key === "bio_jersey") {{
    const n = Number(val);
    if (Number.isFinite(n)) return String(Math.round(n));
    return String(val);
  }}
  if (typeof val === "string") return val;
  if (["games","minutes","points","rebounds_total","assists","steals","blocks","turnovers"].includes(key)) return String(Math.round(val));
  if (["true_shooting_pct","field_goals_pct","three_point_field_goals_pct","free_throws_pct","usage"].includes(key)) {{
    if (val > 1 && val <= 100) return `${{val.toFixed(1)}}%`;
    return `${{(val * 100).toFixed(1)}}%`;
  }}
  return Number(val).toFixed(3);
}}

function whyText(d) {{
  const comps = [
    ["Offensive RAPM", d.pct_orapm],
    ["Defensive RAPM", d.pct_drapm],
    ["Close-game RAPM", d.pct_rapm_close],
    ["Bayesian RAPM", d.pct_rapm_bayes_box],
    ["Possession reliability", d.pct_possessions]
  ].sort((a,b) => b[1] - a[1]).slice(0, 3);
  return "Top drivers: " + comps.map(x => `${{x[0]}} (${{x[1].toFixed(1)}}th pct)`).join(", ") + ".";
}}

function hasValue(v) {{
  if (v === undefined || v === null) return false;
  if (typeof v === "number" && Number.isNaN(v)) return false;
  if (typeof v === "string" && v.trim() === "") return false;
  return true;
}}

function setDetailTab(tab) {{
  tabButtons.forEach(btn => {{
    btn.classList.toggle("active", btn.dataset.tab === tab);
  }});
  panelStats.classList.toggle("active", tab === "stats");
  panelBio.classList.toggle("active", tab === "bio");
  panelBox.classList.toggle("active", tab === "box");
}}

function renderKeyValueList(container, keys, d, emptyText) {{
  container.innerHTML = "";
  const available = keys.filter(k => hasValue(d[k]));
  if (!available.length) {{
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }}
  available.forEach(k => {{
    const row = document.createElement("div");
    row.className = "kv";
    row.style.gridTemplateColumns = "auto 1fr auto";
    row.innerHTML = `
      <div>${{EXTRA[k] || k}}</div>
      <div></div>
      <div class="mono">${{fmtVal(k, d[k])}}</div>
    `;
    container.appendChild(row);
  }});
}}

function renderDetail(d) {{
  document.getElementById("detail-empty").style.display = "none";
  document.getElementById("detail").style.display = "block";
  document.getElementById("d-name").textContent = d.player_name;
  document.getElementById("d-meta").textContent = `${{d.player_team}} · ${{d.player_conference || "Unknown"}}`;

  const tags = document.getElementById("d-tags");
  tags.innerHTML = "";
  const tagData = [
    ["RAPM HCA", d.rapm_hca],
    ["CV Rank", d.rank_rapm_cv],
    ["Poss", d.possessions],
  ];
  tagData.filter(([_, v]) => v !== null && v !== undefined && !Number.isNaN(v)).forEach(([k, v]) => {{
    const span = document.createElement("span");
    span.className = "pill mono";
    span.textContent = `${{k}}: ${{Number.isFinite(Number(v)) ? Number(v).toFixed(3) : v}}`;
    tags.appendChild(span);
  }});

  const bars = document.getElementById("d-bars");
  bars.innerHTML = "";
  detailMetrics.forEach(m => {{
    const row = document.createElement("div");
    row.className = "kv";
    const pct = Math.max(0, Math.min(100, d["pct_" + m] || 0));
    row.innerHTML = `
      <div>${{METRICS[m]}}</div>
      <div class="bar"><span style="width:${{pct}}%"></span></div>
      <div class="mono">${{(d[m] || 0).toFixed(3)}}</div>
    `;
    bars.appendChild(row);
  }});

  renderKeyValueList(
    panelBio,
    detailBioOrder,
    d,
    "No bio fields available for this player in current source data."
  );
  renderKeyValueList(
    panelBox,
    detailBoxOrder,
    d,
    "No box score context available for this player."
  );
  document.getElementById("d-why").textContent = whyText(d);
  setDetailTab("stats");
}}

metricSel.addEventListener("change", renderRows);
query.addEventListener("input", renderRows);
confSel.addEventListener("change", renderRows);
tabButtons.forEach(btn => btn.addEventListener("click", () => setDetailTab(btn.dataset.tab)));
renderRows();
"""
    return _html_shell(title=title, body=body, script=script)


def build_v2(records_json: str) -> str:
    title = "Archetype Map"
    body = """
<section class="hero">
  <h1>Archetype Map</h1>
  <p>Offense vs defense landscape. Bubble size = possessions, color = RAPM (home-court adjusted) percentile.</p>
</section>
<section class="card">
  <div class="controls">
    <div>
      <label for="q2">Search player/team</label>
      <input id="q2" placeholder="filter points by text" />
    </div>
    <div>
      <label for="conf2">Conference</label>
      <select id="conf2"></select>
    </div>
    <div>
      <label for="labeln">Show labels for top N</label>
      <input id="labeln" type="number" min="0" max="50" value="12" />
    </div>
  </div>
  <div class="canvas-wrap">
    <svg id="plot" width="100%" viewBox="0 0 980 620" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
  <div id="tip" class="footer-note">Hover a point for details.</div>
</section>
"""
    script = f"""
const DATA = {records_json};
const conf2 = document.getElementById("conf2");
const q2 = document.getElementById("q2");
const labelN = document.getElementById("labeln");
const svg = document.getElementById("plot");
const tip = document.getElementById("tip");

const W = 980, H = 620;
const M = {{l: 70, r: 28, t: 28, b: 58}};
const PW = W - M.l - M.r, PH = H - M.t - M.b;

const confs = Array.from(new Set(DATA.map(d => d.player_conference || "Unknown"))).sort();
["All", ...confs].forEach(c => {{
  const op = document.createElement("option");
  op.value = c;
  op.textContent = c;
  conf2.appendChild(op);
}});

function colScale(pct) {{
  // 0 -> rust, 50 -> sand, 100 -> teal
  const p = Math.max(0, Math.min(100, pct)) / 100;
  const r = Math.round((1-p) * 190 + p * 11);
  const g = Math.round((1-p) * 88 + p * 122);
  const b = Math.round((1-p) * 57 + p * 117);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function scale(v, lo, hi, a, b) {{
  if (hi <= lo) return (a + b) / 2;
  return a + (v - lo) * (b - a) / (hi - lo);
}}

function draw() {{
  const q = q2.value.trim().toLowerCase();
  const c = conf2.value;
  const data = DATA.filter(d => {{
    const okQ = !q || d.player_name.toLowerCase().includes(q) || d.player_team.toLowerCase().includes(q);
    const okC = c === "All" || (d.player_conference || "Unknown") === c;
    return okQ && okC;
  }});
  const xVals = data.map(d => d.orapm);
  const yVals = data.map(d => d.drapm);
  const pVals = data.map(d => d.possessions);
  const xLo = Math.min(...xVals), xHi = Math.max(...xVals);
  const yLo = Math.min(...yVals), yHi = Math.max(...yVals);
  const pLo = Math.min(...pVals), pHi = Math.max(...pVals);

  const top = [...data].sort((a,b) => b.rapm_hca - a.rapm_hca).slice(0, Math.max(0, parseInt(labelN.value || "0", 10)));
  const topIds = new Set(top.map(d => d.player_id));

  let html = "";
  html += `<rect x="0" y="0" width="${{W}}" height="${{H}}" fill="rgba(255,255,255,0.72)" />`;
  html += `<line x1="${{M.l}}" y1="${{M.t + PH/2}}" x2="${{M.l + PW}}" y2="${{M.t + PH/2}}" stroke="rgba(19,42,47,0.25)" stroke-dasharray="4 4"/>`;
  html += `<line x1="${{M.l + PW/2}}" y1="${{M.t}}" x2="${{M.l + PW/2}}" y2="${{M.t + PH}}" stroke="rgba(19,42,47,0.25)" stroke-dasharray="4 4"/>`;

  for (let i = 0; i < 6; i++) {{
    const x = M.l + (i / 5) * PW;
    const y = M.t + (i / 5) * PH;
    html += `<line x1="${{x}}" y1="${{M.t}}" x2="${{x}}" y2="${{M.t + PH}}" stroke="rgba(19,42,47,0.06)" />`;
    html += `<line x1="${{M.l}}" y1="${{y}}" x2="${{M.l + PW}}" y2="${{y}}" stroke="rgba(19,42,47,0.06)" />`;
  }}

  data.forEach(d => {{
    const x = scale(d.orapm, xLo, xHi, M.l, M.l + PW);
    const y = scale(d.drapm, yLo, yHi, M.t + PH, M.t);
    const r = scale(d.possessions, pLo, pHi, 3.5, 10.5);
    const fill = colScale(d.pct_rapm_hca);
    const stroke = topIds.has(d.player_id) ? "#132a2f" : "rgba(19,42,47,0.35)";
    const sw = topIds.has(d.player_id) ? 1.8 : 0.8;
    html += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{r.toFixed(2)}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="${{sw}}" data-id="${{d.player_id}}" />`;
  }});

  top.forEach(d => {{
    const x = scale(d.orapm, xLo, xHi, M.l, M.l + PW);
    const y = scale(d.drapm, yLo, yHi, M.t + PH, M.t);
    html += `<text x="${{(x + 8).toFixed(2)}}" y="${{(y - 8).toFixed(2)}}" font-size="11" fill="#132a2f">${{d.player_name}}</text>`;
  }});

  html += `<text x="${{W/2}}" y="${{H-14}}" text-anchor="middle" font-size="13" fill="#5e6f71">Offensive RAPM</text>`;
  html += `<text x="18" y="${{H/2}}" transform="rotate(-90,18,${{H/2}})" text-anchor="middle" font-size="13" fill="#5e6f71">Defensive RAPM</text>`;
  svg.innerHTML = html;

  const circles = svg.querySelectorAll("circle");
  circles.forEach(c => {{
    c.addEventListener("mouseenter", () => {{
      const d = data.find(x => x.player_id === c.dataset.id);
      if (!d) return;
      const bioBits = [d.bio_height, d.bio_weight, d.bio_position].filter(x => !!x);
      const bioText = bioBits.length ? ` | ${{bioBits.join(" · ")}}` : "";
      tip.textContent = `${{d.player_name}} · ${{d.player_team}} | RAPM HCA: ${{d.rapm_hca.toFixed(3)}} | O: ${{d.orapm.toFixed(3)}} | D: ${{d.drapm.toFixed(3)}} | Poss: ${{d.possessions.toFixed(1)}}${{bioText}}`;
    }});
  }});
}}

q2.addEventListener("input", draw);
conf2.addEventListener("change", draw);
labelN.addEventListener("input", draw);
draw();
"""
    return _html_shell(title=title, body=body, script=script)


def build_v3(records_json: str) -> str:
    title = "Head-to-Head Lab"
    body = """
<section class="hero">
  <h1>Head-to-Head Lab</h1>
  <p>Compare two athletes across model families and reliability markers to explain ranking separation.</p>
</section>
<section class="card">
  <div class="controls">
    <div>
      <label for="aSel">Player A</label>
      <select id="aSel"></select>
    </div>
    <div>
      <label for="bSel">Player B</label>
      <select id="bSel"></select>
    </div>
    <div>
      <label for="focusMetric">Focus rank metric</label>
      <select id="focusMetric"></select>
    </div>
  </div>
  <div class="grid-1">
    <div class="card" id="summary"></div>
    <div class="card">
      <table>
        <thead><tr><th>Metric</th><th>A</th><th>B</th><th>Delta (A-B)</th></tr></thead>
        <tbody id="cmpRows"></tbody>
      </table>
    </div>
  </div>
</section>
"""
    script = f"""
const DATA = {records_json};
const METRICS = {{
  rapm_hca: "RAPM (Home-Court Adj.)",
  rapm_cv: "RAPM (CV)",
  rapm_bayes_box: "RAPM (Bayes + Box Prior)",
  rapm_close: "RAPM (Close Games)",
  rapm_eb: "RAPM (EB Shrink)",
  orapm: "Offensive RAPM",
  drapm: "Defensive RAPM",
  box_prior: "Box Prior",
  possessions: "Possessions",
  games: "Games",
  minutes: "Minutes",
  minutes_per_game: "Min/Game",
  points: "Points",
  rebounds_total: "Rebounds",
  assists: "Assists",
  steals: "Steals",
  blocks: "Blocks",
  turnovers: "Turnovers",
  points_per40: "Pts/40",
  rebounds_total_per40: "Reb/40",
  assists_per40: "Ast/40",
  steals_per40: "Stl/40",
  blocks_per40: "Blk/40",
  turnovers_per40: "TO/40",
  true_shooting_pct: "TS%",
  field_goals_pct: "FG%",
  three_point_field_goals_pct: "3P%",
  free_throws_pct: "FT%",
  usage: "Usage",
  net_rating: "Net Rating",
  offensive_rating: "Off Rating",
  defensive_rating: "Def Rating",
  porpag: "PORPAG",
  win_shares_total: "Win Shares",
  win_shares_offensive: "WS Off",
  win_shares_defensive: "WS Def",
}};

const baseCompareMetrics = ["rapm_hca","rapm_cv","rapm_bayes_box","rapm_close","rapm_eb","orapm","drapm","box_prior","possessions"];
const extraCompareMetrics = ["games","minutes","minutes_per_game","points","rebounds_total","assists","steals","blocks","turnovers","points_per40","rebounds_total_per40","assists_per40","steals_per40","blocks_per40","turnovers_per40","true_shooting_pct","field_goals_pct","three_point_field_goals_pct","free_throws_pct","usage","net_rating","offensive_rating","defensive_rating","porpag","win_shares_total","win_shares_offensive","win_shares_defensive"];
const compareMetrics = [...baseCompareMetrics, ...extraCompareMetrics.filter(c => DATA.some(d => Number.isFinite(Number(d[c]))))];
const rankMetrics = ["rapm_hca","rapm_cv","rapm_bayes_box","rapm_close","rapm_eb"];
const countMetrics = new Set(["games","minutes","points","rebounds_total","assists","steals","blocks","turnovers"]);
const pctMetrics = new Set(["true_shooting_pct","field_goals_pct","three_point_field_goals_pct","free_throws_pct","usage"]);

const aSel = document.getElementById("aSel");
const bSel = document.getElementById("bSel");
const focusMetric = document.getElementById("focusMetric");
const cmpRows = document.getElementById("cmpRows");
const summary = document.getElementById("summary");

function optionLabel(d) {{
  return `${{d.player_name}} · ${{d.player_team}}`;
}}

const sorted = [...DATA].sort((a,b) => b.rapm_hca - a.rapm_hca);
sorted.forEach(d => {{
  [aSel, bSel].forEach(sel => {{
    const op = document.createElement("option");
    op.value = d.player_id;
    op.textContent = optionLabel(d);
    sel.appendChild(op);
  }});
}});

Object.entries(METRICS).forEach(([k,v]) => {{
  if (!rankMetrics.includes(k)) return;
  const op = document.createElement("option");
  op.value = k;
  op.textContent = v;
  focusMetric.appendChild(op);
}});

aSel.value = sorted[0]?.player_id || "";
bSel.value = sorted[1]?.player_id || "";
focusMetric.value = "rapm_hca";

function findById(id) {{
  return DATA.find(d => d.player_id === id);
}}

function bioText(d) {{
  const bits = [d.bio_position, d.bio_height, d.bio_weight];
  if (d.bio_age !== undefined && d.bio_age !== null && !Number.isNaN(d.bio_age)) {{
    bits.push(`Age ${{Math.round(Number(d.bio_age))}}`);
  }}
  if (d.bio_birth_place) bits.push(d.bio_birth_place);
  return bits.filter(x => !!x && String(x).trim() !== "").join(" | ");
}}

function fmtMetric(key, value) {{
  const num = Number(value);
  if (!Number.isFinite(num)) return "n/a";
  if (countMetrics.has(key)) return String(Math.round(num));
  if (pctMetrics.has(key)) {{
    if (num > 1 && num <= 100) return `${{num.toFixed(1)}}%`;
    return `${{(num * 100).toFixed(1)}}%`;
  }}
  return num.toFixed(3);
}}

function rankText(d, key) {{
  const r = Number(d["rank_" + key]);
  if (!Number.isFinite(r)) return "";
  return ` <span class="muted">(#${{Math.round(r)}})</span>`;
}}

function render() {{
  const a = findById(aSel.value);
  const b = findById(bSel.value);
  if (!a || !b) return;

  const f = focusMetric.value;
  const ar = a["rank_" + f], br = b["rank_" + f];
  summary.innerHTML = `
    <h3 style="margin:0 0 8px;">${{optionLabel(a)}} vs ${{optionLabel(b)}}</h3>
    <div class="muted">
      Focus metric: <strong>${{METRICS[f]}}</strong> |
      Rank A: <strong>#${{ar}}</strong> |
      Rank B: <strong>#${{br}}</strong> |
      Gap: <strong>${{Math.abs(ar - br)}}</strong>
    </div>
    <div class="muted" style="margin-top:6px;"><strong>A Bio:</strong> ${{bioText(a) || "n/a"}}</div>
    <div class="muted" style="margin-top:2px;"><strong>B Bio:</strong> ${{bioText(b) || "n/a"}}</div>
  `;

  cmpRows.innerHTML = "";
  compareMetrics.forEach(m => {{
    const tr = document.createElement("tr");
    const av = Number(a[m]), bv = Number(b[m]);
    const dv = Number.isFinite(av) && Number.isFinite(bv) ? av - bv : NaN;
    const deltaText = Number.isFinite(dv) ? `${{dv >= 0 ? '+' : ''}}${{dv.toFixed(3)}}` : "n/a";
    const deltaColor = Number.isFinite(dv) ? (dv >= 0 ? '#0b7a75' : '#b14e41') : '#5e6f71';
    tr.innerHTML = `
      <td>${{METRICS[m]}}</td>
      <td class="mono">${{fmtMetric(m, av)}}${{rankText(a, m)}}</td>
      <td class="mono">${{fmtMetric(m, bv)}}${{rankText(b, m)}}</td>
      <td class="mono" style="color:${{deltaColor}};">${{deltaText}}</td>
    `;
    cmpRows.appendChild(tr);
  }});
}}

[aSel, bSel, focusMetric].forEach(el => el.addEventListener("change", render));
render();
"""
    return _html_shell(title=title, body=body, script=script)


def build_v4(records_json: str) -> str:
    title = "Group Explorer"
    body = """
<section class="hero">
  <h1>Group Explorer</h1>
  <p>Aggregate performance by team or conference. Bars show possession-weighted model score, with player counts and top representative.</p>
</section>
<section class="card">
  <div class="controls">
    <div>
      <label for="metric4">Metric</label>
      <select id="metric4"></select>
    </div>
    <div>
      <label for="group4">Group by</label>
      <select id="group4">
        <option value="player_conference">Conference</option>
        <option value="player_team">Team</option>
      </select>
    </div>
    <div>
      <label for="topk4">Top K groups</label>
      <input id="topk4" type="number" min="5" max="80" value="20" />
    </div>
  </div>
  <div class="grid-2">
    <div class="card canvas-wrap">
      <svg id="plot4" width="100%" viewBox="0 0 980 640" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Rank</th><th>Group</th><th>Score</th><th>Players</th><th>Top Player</th></tr></thead>
        <tbody id="rows4"></tbody>
      </table>
    </div>
  </div>
</section>
"""
    script = f"""
const DATA = {records_json};
const METRICS = {{
  rapm_hca: "RAPM (Home-Court Adj.)",
  rapm_cv: "RAPM (CV)",
  rapm_bayes_box: "RAPM (Bayes + Box Prior)",
  rapm_close: "RAPM (Close Games)",
  rapm_eb: "RAPM (EB Shrink)",
  orapm: "Offensive RAPM",
  drapm: "Defensive RAPM",
  box_prior: "Box Prior",
  possessions: "Possessions",
}};

const metric4 = document.getElementById("metric4");
const group4 = document.getElementById("group4");
const topk4 = document.getElementById("topk4");
const rows4 = document.getElementById("rows4");
const svg4 = document.getElementById("plot4");

Object.entries(METRICS).forEach(([k, v]) => {{
  const op = document.createElement("option");
  op.value = k;
  op.textContent = v;
  metric4.appendChild(op);
}});
metric4.value = "rapm_hca";

function aggregate(metric, groupField) {{
  const map = new Map();
  DATA.forEach(d => {{
    const key = (d[groupField] || "Unknown").trim() || "Unknown";
    const w = Math.max(1, d.possessions || 0);
    const cur = map.get(key) || {{
      group: key,
      weighted: 0,
      poss: 0,
      players: 0,
      top_name: "",
      top_team: "",
      top_val: -1e18,
    }};
    cur.weighted += d[metric] * w;
    cur.poss += w;
    cur.players += 1;
    if (d[metric] > cur.top_val) {{
      cur.top_val = d[metric];
      cur.top_name = d.player_name;
      cur.top_team = d.player_team;
    }}
    map.set(key, cur);
  }});
  const out = Array.from(map.values()).map(x => {{
    x.score = x.poss > 0 ? x.weighted / x.poss : 0;
    return x;
  }});
  out.sort((a,b) => (b.score - a.score) || a.group.localeCompare(b.group));
  return out;
}}

function drawBars(data) {{
  const W = 980, H = 640;
  const M = {{ l: 180, r: 18, t: 20, b: 22 }};
  const PW = W - M.l - M.r;
  const PH = H - M.t - M.b;
  if (!data.length) {{
    svg4.innerHTML = `<text x="${{W/2}}" y="${{H/2}}" text-anchor="middle" fill="#5e6f71">No groups to display</text>`;
    return;
  }}
  const vals = data.map(x => x.score);
  const lo = Math.min(0, ...vals);
  const hi = Math.max(0, ...vals);
  const span = Math.max(1e-9, hi - lo);
  const x = v => M.l + ((v - lo) / span) * PW;
  const x0 = x(0);
  const rowH = PH / data.length;

  let html = "";
  html += `<rect x="0" y="0" width="${{W}}" height="${{H}}" fill="rgba(255,255,255,0.72)" />`;
  html += `<line x1="${{x0}}" y1="${{M.t}}" x2="${{x0}}" y2="${{H - M.b}}" stroke="rgba(19,42,47,0.35)" />`;

  data.forEach((d, i) => {{
    const cy = M.t + i * rowH + rowH * 0.5;
    const y = cy - rowH * 0.34;
    const h = rowH * 0.68;
    const xv = x(d.score);
    const left = Math.min(x0, xv);
    const w = Math.max(1.5, Math.abs(xv - x0));
    const fill = d.score >= 0 ? "rgba(11,122,117,0.75)" : "rgba(177,78,65,0.72)";
    html += `<text x="${{M.l - 8}}" y="${{cy + 4}}" text-anchor="end" font-size="12" fill="#1a2e32">${{d.group}}</text>`;
    html += `<rect x="${{left}}" y="${{y}}" width="${{w}}" height="${{h}}" fill="${{fill}}" rx="4" />`;
    html += `<text x="${{xv + (d.score >= 0 ? 6 : -6)}}" y="${{cy + 4}}" text-anchor="${{d.score >= 0 ? "start" : "end"}}" font-size="11" fill="#1a2e32">${{d.score.toFixed(3)}}</text>`;
  }});

  svg4.innerHTML = html;
}}

function render4() {{
  const k = Math.max(5, Math.min(80, parseInt(topk4.value || "20", 10)));
  const grouped = aggregate(metric4.value, group4.value).slice(0, k);
  drawBars(grouped);

  rows4.innerHTML = "";
  grouped.forEach((d, i) => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${{i + 1}}</td>
      <td>${{d.group}}</td>
      <td class="mono">${{d.score.toFixed(3)}}</td>
      <td class="mono">${{d.players}}</td>
      <td>${{d.top_name}} <span class="muted">(${{d.top_team}})</span></td>
    `;
    rows4.appendChild(tr);
  }});
}}

[metric4, group4, topk4].forEach(el => el.addEventListener("input", render4));
render4();
"""
    return _html_shell(title=title, body=body, script=script)


def build_v5(records_json: str, top_n: int) -> str:
    title = "Model Agreement Matrix"
    body = """
<section class="hero">
  <h1>Model Agreement Matrix</h1>
  <p>Compare rank agreement across RAPM variants. Lower agreement spread means the player is consistently strong across methods.</p>
</section>
<section class="card">
  <div class="controls">
    <div>
      <label for="focus5">Focus metric</label>
      <select id="focus5"></select>
    </div>
    <div>
      <label for="topn5">Rows shown</label>
      <input id="topn5" type="number" min="10" max="120" value="40" />
    </div>
    <div>
      <label for="q5">Search player/team</label>
      <input id="q5" placeholder="e.g. Flagg, Duke" />
    </div>
  </div>
  <div class="grid-1">
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Player</th><th>Team</th><th>Focus</th>
            <th>HCA</th><th>CV</th><th>Bayes</th><th>Close</th><th>EB</th>
            <th>Agreement SD</th><th>Poss</th>
          </tr>
        </thead>
        <tbody id="rows5"></tbody>
      </table>
    </div>
  </div>
</section>
"""
    script = f"""
const DATA = {records_json};
const METRICS = {{
  rapm_hca: "RAPM (Home-Court Adj.)",
  rapm_cv: "RAPM (CV)",
  rapm_bayes_box: "RAPM (Bayes + Box Prior)",
  rapm_close: "RAPM (Close Games)",
  rapm_eb: "RAPM (EB Shrink)",
}};
const modelCols = ["rapm_hca","rapm_cv","rapm_bayes_box","rapm_close","rapm_eb"];

const focus5 = document.getElementById("focus5");
const topn5 = document.getElementById("topn5");
const q5 = document.getElementById("q5");
const rows5 = document.getElementById("rows5");

Object.entries(METRICS).forEach(([k,v]) => {{
  const op = document.createElement("option");
  op.value = k;
  op.textContent = v;
  focus5.appendChild(op);
}});
focus5.value = "rapm_hca";
topn5.value = String({int(top_n)});

function std(vals) {{
  const n = vals.length || 1;
  const mu = vals.reduce((a,b)=>a+b,0) / n;
  const v = vals.reduce((a,b)=>a+(b-mu)*(b-mu),0) / n;
  return Math.sqrt(v);
}}

function render5() {{
  const q = q5.value.trim().toLowerCase();
  const f = focus5.value;
  const n = Math.max(10, Math.min(120, parseInt(topn5.value || "40", 10)));
  const arr = DATA.filter(d => {{
    if (!q) return true;
    return d.player_name.toLowerCase().includes(q) || d.player_team.toLowerCase().includes(q);
  }});
  arr.sort((a,b) => (b[f] - a[f]) || a.player_name.localeCompare(b.player_name));
  const show = arr.slice(0, n);
  rows5.innerHTML = "";
  show.forEach((d, i) => {{
    const ranks = modelCols.map(c => d["rank_" + c]);
    const sd = std(ranks);
    const tr = document.createElement("tr");
    const chip = v => `<span class="mono">${{v}}</span>`;
    tr.innerHTML = `
      <td class="mono">${{i + 1}}</td>
      <td>${{d.player_name}}</td>
      <td>${{d.player_team}}</td>
      <td class="mono">${{d[f].toFixed(3)}}</td>
      <td>${{chip(d.rank_rapm_hca)}}</td>
      <td>${{chip(d.rank_rapm_cv)}}</td>
      <td>${{chip(d.rank_rapm_bayes_box)}}</td>
      <td>${{chip(d.rank_rapm_close)}}</td>
      <td>${{chip(d.rank_rapm_eb)}}</td>
      <td class="mono" style="color:${{sd <= 15 ? '#0b7a75' : (sd <= 40 ? '#5e6f71' : '#b14e41')}};">${{sd.toFixed(1)}}</td>
      <td class="mono">${{d.possessions.toFixed(1)}}</td>
    `;
    rows5.appendChild(tr);
  }});
}}

[focus5, topn5, q5].forEach(el => el.addEventListener("input", render5));
render5();
"""
    return _html_shell(title=title, body=body, script=script)


def build_v6(records_json: str, top_n: int) -> str:
    title = "Reliability Frontier"
    body = """
<section class="hero">
  <h1>Reliability Frontier</h1>
  <p>Balance impact and confidence. Points combine model score, possession volume, and cross-model consistency.</p>
</section>
<section class="card">
  <div class="controls">
    <div>
      <label for="focus6">Focus metric</label>
      <select id="focus6"></select>
    </div>
    <div>
      <label for="conf6">Conference</label>
      <select id="conf6"></select>
    </div>
    <div>
      <label for="topn6">Rows shown</label>
      <input id="topn6" type="number" min="10" max="120" value="40" />
    </div>
  </div>
  <div class="controls">
    <div>
      <label for="q6">Search player/team</label>
      <input id="q6" placeholder="e.g. Flagg, Duke" />
    </div>
    <div>
      <label for="wFocus6">Weight: model score (%)</label>
      <input id="wFocus6" type="number" min="10" max="80" value="55" />
    </div>
    <div>
      <label for="wPoss6">Weight: possessions (%)</label>
      <input id="wPoss6" type="number" min="10" max="70" value="30" />
    </div>
  </div>
  <div class="grid-2">
    <div class="card canvas-wrap">
      <svg id="plot6" width="100%" viewBox="0 0 980 620" preserveAspectRatio="xMidYMid meet"></svg>
      <div id="tip6" class="footer-note">Hover a point for details.</div>
    </div>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Player</th><th>Team</th><th>Reliability</th>
            <th>Focus</th><th>Poss</th><th>Agreement SD</th>
          </tr>
        </thead>
        <tbody id="rows6"></tbody>
      </table>
    </div>
  </div>
</section>
"""
    script = f"""
const DATA = {records_json};
const METRICS = {{
  rapm_hca: "RAPM (Home-Court Adj.)",
  rapm_cv: "RAPM (CV)",
  rapm_bayes_box: "RAPM (Bayes + Box Prior)",
  rapm_close: "RAPM (Close Games)",
  rapm_eb: "RAPM (EB Shrink)",
}};
const modelCols6 = ["rapm_hca","rapm_cv","rapm_bayes_box","rapm_close","rapm_eb"];
const focus6 = document.getElementById("focus6");
const conf6 = document.getElementById("conf6");
const topn6 = document.getElementById("topn6");
const q6 = document.getElementById("q6");
const wFocus6 = document.getElementById("wFocus6");
const wPoss6 = document.getElementById("wPoss6");
const rows6 = document.getElementById("rows6");
const plot6 = document.getElementById("plot6");
const tip6 = document.getElementById("tip6");

Object.entries(METRICS).forEach(([k, v]) => {{
  const op = document.createElement("option");
  op.value = k;
  op.textContent = v;
  focus6.appendChild(op);
}});
focus6.value = "rapm_hca";
topn6.value = String({int(top_n)});

const confs6 = Array.from(new Set(DATA.map(d => d.player_conference || "Unknown"))).sort();
["All", ...confs6].forEach(c => {{
  const op = document.createElement("option");
  op.value = c;
  op.textContent = c;
  conf6.appendChild(op);
}});

function std(vals) {{
  const n = vals.length || 1;
  const mu = vals.reduce((a,b)=>a+b,0) / n;
  const v = vals.reduce((a,b)=>a+(b-mu)*(b-mu),0) / n;
  return Math.sqrt(v);
}}

function scale(v, lo, hi, a, b) {{
  if (hi <= lo) return (a + b) / 2;
  return a + (v - lo) * (b - a) / (hi - lo);
}}

function scoreConsistency(sd, lo, hi) {{
  if (!Number.isFinite(sd)) return 0;
  if (hi <= lo) return 100;
  return 100 * (hi - sd) / (hi - lo);
}}

function colorByConsistency(pct) {{
  const p = Math.max(0, Math.min(100, pct)) / 100;
  const r = Math.round((1-p) * 177 + p * 11);
  const g = Math.round((1-p) * 78 + p * 122);
  const b = Math.round((1-p) * 65 + p * 117);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function filtered6() {{
  const q = q6.value.trim().toLowerCase();
  const c = conf6.value;
  return DATA.filter(d => {{
    const okQ = !q || d.player_name.toLowerCase().includes(q) || d.player_team.toLowerCase().includes(q);
    const okC = c === "All" || (d.player_conference || "Unknown") === c;
    return okQ && okC;
  }});
}}

function enrich6(data, focus) {{
  const sds = data.map(d => std(modelCols6.map(c => Number(d["rank_" + c]) || 0)));
  const sdLo = Math.min(...sds);
  const sdHi = Math.max(...sds);
  const wf = Math.max(10, Math.min(80, Number(wFocus6.value) || 55)) / 100;
  const wp = Math.max(10, Math.min(70, Number(wPoss6.value) || 30)) / 100;
  const wc = Math.max(0, 1 - wf - wp);

  return data.map((d, i) => {{
    const sd = sds[i];
    const focusPct = Number(d["pct_" + focus]) || 0;
    const possPct = Number(d.pct_possessions) || 0;
    const consistencyPct = scoreConsistency(sd, sdLo, sdHi);
    const reliability = wf * focusPct + wp * possPct + wc * consistencyPct;
    return {{
      ...d,
      _sd: sd,
      _focusPct: focusPct,
      _possPct: possPct,
      _consistencyPct: consistencyPct,
      _reliability: reliability,
    }};
  }});
}}

function render6() {{
  const focus = focus6.value;
  const n = Math.max(10, Math.min(120, parseInt(topn6.value || "40", 10)));
  const data = enrich6(filtered6(), focus);
  if (!data.length) {{
    rows6.innerHTML = "";
    plot6.innerHTML = `<text x="490" y="310" text-anchor="middle" fill="#5e6f71">No players match current filters</text>`;
    tip6.textContent = "No rows for current selection.";
    return;
  }}

  data.sort((a,b) => (b._reliability - a._reliability) || a.player_name.localeCompare(b.player_name));
  const show = data.slice(0, n);

  rows6.innerHTML = "";
  show.forEach((d, i) => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${{i + 1}}</td>
      <td>${{d.player_name}}</td>
      <td>${{d.player_team}}</td>
      <td class="mono">${{d._reliability.toFixed(1)}}</td>
      <td class="mono">${{d[focus].toFixed(3)}}</td>
      <td class="mono">${{d.possessions.toFixed(1)}}</td>
      <td class="mono" style="color:${{d._sd <= 15 ? '#0b7a75' : (d._sd <= 40 ? '#5e6f71' : '#b14e41')}};">${{d._sd.toFixed(1)}}</td>
    `;
    rows6.appendChild(tr);
  }});

  const W = 980, H = 620;
  const M = {{ l: 74, r: 26, t: 24, b: 58 }};
  const PW = W - M.l - M.r, PH = H - M.t - M.b;
  const xVals = data.map(d => Math.log10(Math.max(1, d.possessions)));
  const yVals = data.map(d => Number(d[focus]) || 0);
  const xLo = Math.min(...xVals), xHi = Math.max(...xVals);
  const yLo = Math.min(...yVals), yHi = Math.max(...yVals);
  const topIds = new Set(show.slice(0, Math.min(12, show.length)).map(d => d.player_id));

  let html = "";
  html += `<rect x="0" y="0" width="${{W}}" height="${{H}}" fill="rgba(255,255,255,0.72)" />`;
  for (let i = 0; i <= 5; i++) {{
    const x = M.l + (i / 5) * PW;
    const y = M.t + (i / 5) * PH;
    html += `<line x1="${{x}}" y1="${{M.t}}" x2="${{x}}" y2="${{M.t + PH}}" stroke="rgba(19,42,47,0.06)" />`;
    html += `<line x1="${{M.l}}" y1="${{y}}" x2="${{M.l + PW}}" y2="${{y}}" stroke="rgba(19,42,47,0.06)" />`;
  }}

  const x0 = scale(Math.log10(Math.max(1, medianPoss(data))), xLo, xHi, M.l, M.l + PW);
  html += `<line x1="${{x0}}" y1="${{M.t}}" x2="${{x0}}" y2="${{M.t + PH}}" stroke="rgba(19,42,47,0.22)" stroke-dasharray="4 4"/>`;

  data.forEach(d => {{
    const x = scale(Math.log10(Math.max(1, d.possessions)), xLo, xHi, M.l, M.l + PW);
    const y = scale(Number(d[focus]) || 0, yLo, yHi, M.t + PH, M.t);
    const r = 3.6 + (Math.max(0, Math.min(100, d._reliability)) / 100) * 4.8;
    const fill = colorByConsistency(d._consistencyPct);
    const sw = topIds.has(d.player_id) ? 1.7 : 0.8;
    const stroke = topIds.has(d.player_id) ? "#132a2f" : "rgba(19,42,47,0.30)";
    html += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{r.toFixed(2)}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="${{sw}}" data-id="${{d.player_id}}" />`;
  }});

  show.slice(0, Math.min(12, show.length)).forEach(d => {{
    const x = scale(Math.log10(Math.max(1, d.possessions)), xLo, xHi, M.l, M.l + PW);
    const y = scale(Number(d[focus]) || 0, yLo, yHi, M.t + PH, M.t);
    html += `<text x="${{(x + 8).toFixed(1)}}" y="${{(y - 8).toFixed(1)}}" font-size="11" fill="#132a2f">${{d.player_name}}</text>`;
  }});

  html += `<text x="${{W/2}}" y="${{H-14}}" text-anchor="middle" font-size="13" fill="#5e6f71">Possessions (log scale)</text>`;
  html += `<text x="18" y="${{H/2}}" transform="rotate(-90,18,${{H/2}})" text-anchor="middle" font-size="13" fill="#5e6f71">${{METRICS[focus]}}</text>`;
  plot6.innerHTML = html;

  const lookup = new Map(data.map(d => [d.player_id, d]));
  plot6.querySelectorAll("circle").forEach(c => {{
    c.addEventListener("mouseenter", () => {{
      const d = lookup.get(c.dataset.id);
      if (!d) return;
      tip6.textContent = `${{d.player_name}} · ${{d.player_team}} | Reliability: ${{d._reliability.toFixed(1)}} | ${{METRICS[focus]}}: ${{d[focus].toFixed(3)}} | Poss: ${{d.possessions.toFixed(1)}} | Agreement SD: ${{d._sd.toFixed(1)}}`;
    }});
  }});
}}

function medianPoss(data) {{
  const vals = data.map(d => Number(d.possessions) || 0).sort((a,b) => a-b);
  if (!vals.length) return 1;
  const mid = Math.floor(vals.length / 2);
  return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
}}

[focus6, conf6, topn6, q6, wFocus6, wPoss6].forEach(el => el.addEventListener("input", render6));
render6();
"""
    return _html_shell(title=title, body=body, script=script)


def build_v7(records_json: str, top_n: int) -> str:
    title = "Consensus Range"
    body = """
<section class="hero">
  <h1>Consensus Range</h1>
  <p>See where each player lands across RAPM variants. Narrower ranges indicate stronger cross-model agreement.</p>
</section>
<section class="card">
  <div class="controls">
    <div>
      <label for="focus7">Focus metric</label>
      <select id="focus7"></select>
    </div>
    <div>
      <label for="sort7">Sort by</label>
      <select id="sort7">
        <option value="focus">Best focus rank</option>
        <option value="consensus">Best mean rank</option>
        <option value="stable">Smallest range</option>
      </select>
    </div>
    <div>
      <label for="topn7">Rows shown</label>
      <input id="topn7" type="number" min="10" max="120" value="40" />
    </div>
  </div>
  <div class="controls">
    <div>
      <label for="q7">Search player/team</label>
      <input id="q7" placeholder="e.g. Flagg, Duke" />
    </div>
    <div>
      <label for="conf7">Conference</label>
      <select id="conf7"></select>
    </div>
    <div></div>
  </div>
  <div class="grid-1">
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Player</th><th>Team</th><th>Mean Rank</th><th>Range</th><th>Best / Worst</th><th>Consensus Strip</th>
          </tr>
        </thead>
        <tbody id="rows7"></tbody>
      </table>
    </div>
  </div>
</section>
"""
    script = f"""
const DATA = {records_json};
const METRICS = {{
  rapm_hca: "HCA",
  rapm_cv: "CV",
  rapm_bayes_box: "Bayes",
  rapm_close: "Close",
  rapm_eb: "EB",
}};
const modelCols7 = ["rapm_hca","rapm_cv","rapm_bayes_box","rapm_close","rapm_eb"];
const focus7 = document.getElementById("focus7");
const sort7 = document.getElementById("sort7");
const topn7 = document.getElementById("topn7");
const q7 = document.getElementById("q7");
const conf7 = document.getElementById("conf7");
const rows7 = document.getElementById("rows7");
const maxRank7 = Math.max(1, ...DATA.map(d => modelCols7.map(c => Number(d["rank_" + c]) || 0)).flat());

Object.entries(METRICS).forEach(([k, v]) => {{
  const op = document.createElement("option");
  op.value = k;
  op.textContent = v;
  focus7.appendChild(op);
}});
focus7.value = "rapm_hca";
topn7.value = String({int(top_n)});

const confs7 = Array.from(new Set(DATA.map(d => d.player_conference || "Unknown"))).sort();
["All", ...confs7].forEach(c => {{
  const op = document.createElement("option");
  op.value = c;
  op.textContent = c;
  conf7.appendChild(op);
}});

function filtered7() {{
  const q = q7.value.trim().toLowerCase();
  const c = conf7.value;
  return DATA.filter(d => {{
    const okQ = !q || d.player_name.toLowerCase().includes(q) || d.player_team.toLowerCase().includes(q);
    const okC = c === "All" || (d.player_conference || "Unknown") === c;
    return okQ && okC;
  }});
}}

function summarize(d, focus) {{
  const pairs = modelCols7.map(c => {{
    return {{ key: c, rank: Number(d["rank_" + c]) || maxRank7 }};
  }});
  const ranks = pairs.map(x => x.rank);
  const minRank = Math.min(...ranks);
  const maxRank = Math.max(...ranks);
  const meanRank = ranks.reduce((a,b)=>a+b,0) / ranks.length;
  const focusRank = Number(d["rank_" + focus]) || maxRank7;
  const best = pairs.reduce((a,b) => b.rank < a.rank ? b : a);
  const worst = pairs.reduce((a,b) => b.rank > a.rank ? b : a);
  return {{
    ...d,
    _pairs: pairs,
    _minRank: minRank,
    _maxRank: maxRank,
    _meanRank: meanRank,
    _range: maxRank - minRank,
    _focusRank: focusRank,
    _best: best,
    _worst: worst,
  }};
}}

function pctForRank(rank) {{
  return 100 * (Math.max(1, rank) - 1) / Math.max(1, maxRank7 - 1);
}}

function sortRows(rows, mode) {{
  if (mode === "consensus") {{
    rows.sort((a,b) => (a._meanRank - b._meanRank) || a.player_name.localeCompare(b.player_name));
    return;
  }}
  if (mode === "stable") {{
    rows.sort((a,b) => (a._range - b._range) || (a._meanRank - b._meanRank) || a.player_name.localeCompare(b.player_name));
    return;
  }}
  rows.sort((a,b) => (a._focusRank - b._focusRank) || (a._meanRank - b._meanRank) || a.player_name.localeCompare(b.player_name));
}}

function render7() {{
  const focus = focus7.value;
  const mode = sort7.value;
  const n = Math.max(10, Math.min(120, parseInt(topn7.value || "40", 10)));
  const rows = filtered7().map(d => summarize(d, focus));
  sortRows(rows, mode);
  const show = rows.slice(0, n);

  rows7.innerHTML = "";
  show.forEach((d, i) => {{
    const left = pctForRank(d._minRank);
    const right = pctForRank(d._maxRank);
    const width = Math.max(1.2, right - left);
    const meanPos = pctForRank(d._meanRank);
    const focusPos = pctForRank(d._focusRank);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${{i + 1}}</td>
      <td>${{d.player_name}}</td>
      <td>${{d.player_team}}</td>
      <td class="mono">${{d._meanRank.toFixed(1)}}</td>
      <td class="mono">${{d._range}}</td>
      <td class="mono">${{METRICS[d._best.key]}} #${{d._best.rank}} / ${{METRICS[d._worst.key]}} #${{d._worst.rank}}</td>
      <td>
        <div class="range-track">
          <span class="range-span" style="left:${{left}}%;width:${{width}}%;"></span>
          <span class="range-dot" style="left:${{meanPos}}%;background:#0b7a75;"></span>
          <span class="range-dot" style="left:${{focusPos}}%;background:#c77f2b;"></span>
        </div>
      </td>
    `;
    rows7.appendChild(tr);
  }});
}}

[focus7, sort7, topn7, q7, conf7].forEach(el => el.addEventListener("input", render7));
render7();
"""
    return _html_shell(title=title, body=body, script=script)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv, low_memory=False)
    _ensure_required_columns(df)
    season = args.season if args.season is not None else _infer_season(args.input_csv)
    df = enrich_with_player_stats(df, season=season)
    df = enrich_with_espn_bios(df, season=season)
    records = _prepare_records(df)
    records_json = json.dumps(records, ensure_ascii=True, separators=(",", ":"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_csv.stem

    out1 = args.out_dir / f"{stem}_v1_evidence_cards.html"
    out2 = args.out_dir / f"{stem}_v2_archetype_map.html"
    out3 = args.out_dir / f"{stem}_v3_head_to_head.html"
    out4 = args.out_dir / f"{stem}_v4_group_explorer.html"
    out5 = args.out_dir / f"{stem}_v5_agreement_matrix.html"
    out6 = args.out_dir / f"{stem}_v6_reliability_frontier.html"
    out7 = args.out_dir / f"{stem}_v7_consensus_range.html"

    out1.write_text(build_v1(records_json=records_json, top_n=args.top_n), encoding="utf-8")
    out2.write_text(build_v2(records_json=records_json), encoding="utf-8")
    out3.write_text(build_v3(records_json=records_json), encoding="utf-8")
    out4.write_text(build_v4(records_json=records_json), encoding="utf-8")
    out5.write_text(build_v5(records_json=records_json, top_n=args.top_n), encoding="utf-8")
    out6.write_text(build_v6(records_json=records_json, top_n=args.top_n), encoding="utf-8")
    out7.write_text(build_v7(records_json=records_json, top_n=args.top_n), encoding="utf-8")

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")
    print(f"Wrote {out3}")
    print(f"Wrote {out4}")
    print(f"Wrote {out5}")
    print(f"Wrote {out6}")
    print(f"Wrote {out7}")


if __name__ == "__main__":
    main()
