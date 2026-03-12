from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Dashboard 1 style page for NBA success holdout (top-60 real picks)."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nba_new_joiner_impact_holdout_actual_vs_predicted_2022.csv"),
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("data/processed/viz/nba_impact_dashboard1_real_top60_2022.html"),
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=10.0,
        help="Absolute impact-rank error threshold to count as success.",
    )
    parser.add_argument(
        "--rapm-csv",
        type=Path,
        default=Path("data/raw/external/nbarapm/TimedecayRAPM.csv"),
        help="Optional TimedecayRAPM CSV to enrich rows with current NBA RAPM.",
    )
    return parser.parse_args()


def _to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _name_key(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _build_driver_and_miss_text(top60: pd.DataFrame, success_threshold: float) -> pd.DataFrame:
    df = top60.copy()
    if "stocks_per40" not in df.columns:
        df["stocks_per40"] = pd.to_numeric(df.get("blocks_per40"), errors="coerce").fillna(0.0) + pd.to_numeric(
            df.get("steals_per40"), errors="coerce"
        ).fillna(0.0)

    # pct rank is computed across holdout top-60 picks only, so each phrase is relative to this cohort.
    metric_defs = [
        ("minutes", "minutes", True, lambda v: f"minutes {int(round(v))}"),
        ("true_shooting_pct", "TS", True, lambda v: f"TS {v:.3f}"),
        ("three_point_attempt_rate", "3PA rate", True, lambda v: f"3PA/FGA {v:.3f}"),
        ("three_point_pct", "3P%", True, lambda v: f"3P% {v:.3f}"),
        ("points_per40", "scoring", True, lambda v: f"PTS/40 {v:.1f}"),
        ("assists_per40", "playmaking", True, lambda v: f"AST/40 {v:.1f}"),
        ("rebounds_total_per40", "rebounding", True, lambda v: f"REB/40 {v:.1f}"),
        ("stocks_per40", "stocks", True, lambda v: f"STL+BLK/40 {v:.1f}"),
        ("net_rating", "net", True, lambda v: f"net {v:.1f}"),
        ("turnovers_per40", "turnovers", False, lambda v: f"TOV/40 {v:.1f}"),
    ]

    for col, _, _, _ in metric_defs:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pct_col_map: dict[str, str] = {}
    for col, label, higher_is_better, _ in metric_defs:
        s = pd.to_numeric(df.get(col), errors="coerce")
        if not higher_is_better:
            s = -s
        pcol = f"pct_{label.replace(' ', '_').replace('%', 'pct')}"
        df[pcol] = s.rank(pct=True, method="average")
        pct_col_map[col] = pcol

    pos_driver: list[str] = []
    neg_driver: list[str] = []
    miss_explain: list[str] = []
    for _, r in df.iterrows():
        scored: list[tuple[float, str]] = []
        weak_scored: list[tuple[float, str]] = []
        for col, label, _, fmt_fn in metric_defs:
            if col not in df.columns:
                continue
            p = r.get(pct_col_map[col])
            v = r.get(col)
            if pd.isna(p) or pd.isna(v):
                continue
            phrase = f"{label}: {fmt_fn(float(v))}"
            scored.append((float(p), phrase))
            weak_scored.append((float(p), phrase))

        scored.sort(key=lambda x: x[0], reverse=True)
        weak_scored.sort(key=lambda x: x[0])

        pos = [t for p, t in scored if p >= 0.65][:2]
        if not pos and scored:
            pos = [scored[0][1]]

        neg = [t for p, t in weak_scored if p <= 0.35][:2]
        if not neg and weak_scored:
            neg = [weak_scored[0][1]]

        pos_driver.append("; ".join(pos))
        neg_driver.append("; ".join(neg))

        covered = pd.notna(r.get("pred_rank"))
        err = pd.to_numeric(r.get("abs_rank_error"), errors="coerce")
        rank_error = pd.to_numeric(r.get("rank_error"), errors="coerce")
        if not covered:
            miss_explain.append("No model score for this player.")
            continue
        if pd.notna(err) and float(err) <= float(success_threshold):
            miss_explain.append("")
            continue

        if pd.isna(rank_error):
            miss_explain.append("Rank gap is large, but score direction is missing.")
            continue

        if float(rank_error) > 0:
            # Pred rank is worse (larger number) than actual rank.
            extra = ""
            all_rank = pd.to_numeric(r.get("pred_rank_all_players"), errors="coerce")
            if pd.notna(all_rank) and float(all_rank) > 60:
                extra = f" outside top-60 board (#{int(all_rank)});"
            if neg:
                miss_explain.append("Underrated by model" + extra + " weaker college profile: " + "; ".join(neg))
            else:
                miss_explain.append("Underrated by model" + extra + " limited weak signals in college profile.")
        else:
            # Pred rank is better (smaller number) than actual rank.
            if pos:
                miss_explain.append("Overrated by model from strong college signals: " + "; ".join(pos))
            else:
                miss_explain.append("Overrated by model from limited strong signals in college profile.")

    return pd.DataFrame(
        {
            "driver_positive_short": pd.Series(pos_driver, index=top60.index, dtype="object"),
            "driver_negative_short": pd.Series(neg_driver, index=top60.index, dtype="object"),
            "miss_explain_short": pd.Series(miss_explain, index=top60.index, dtype="object"),
        },
        index=top60.index,
    )


def build_html(records_json: str, metrics_json: str, success_threshold: float, holdout_year: str) -> str:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NBA Success Dashboard 1: Real Top-60 Picks vs Predicted (__HOLDOUT_YEAR__ Holdout)</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
    :root {{
      --bg0:#f3eee4; --bg1:#e0d5c1; --ink:#11292f; --muted:#5b6c70;
      --card:rgba(255,253,247,0.79); --stroke:rgba(17,41,47,0.18);
      --good:#0b7a75; --warn:#c27d2a; --bad:#9e473e;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); font-family:"Space Grotesk",sans-serif;
      background:radial-gradient(1180px 470px at 5% -10%, rgba(11,122,117,.12), transparent 60%),
                 radial-gradient(880px 410px at 95% -10%, rgba(194,125,42,.16), transparent 58%),
                 linear-gradient(160deg,var(--bg0) 0%,var(--bg1) 100%);
      min-height:100vh;
    }}
    .wrap {{ max-width:1240px; margin:0 auto; padding:24px; }}
    .hero,.metric,.panel {{
      border:1px solid var(--stroke); border-radius:16px; background:var(--card);
    }}
    .hero {{ padding:18px 20px; margin-bottom:12px; }}
    .hero h1 {{ margin:0; font-size:clamp(1.35rem,2.35vw,2rem); font-family:"Fraunces",serif; line-height:1.18; }}
    .hero p {{ margin:8px 0 0; color:var(--muted); }}
    .kicker {{
      display:inline-block; margin-bottom:8px; padding:4px 10px; border-radius:999px;
      border:1px solid var(--stroke); background:rgba(255,255,255,.76);
      font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted);
    }}
    .back {{
      display:inline-block; margin-bottom:12px; text-decoration:none; color:var(--ink);
      border:1px solid var(--stroke); border-radius:10px; padding:7px 11px; background:rgba(255,255,255,.72);
      font-size:.88rem;
    }}
    .success-banner {{
      margin-top:10px; padding:10px 12px; border-radius:12px;
      border:1px solid rgba(11,122,117,.28); background:rgba(11,122,117,.12); font-size:.92rem;
    }}
    .success-banner b {{ font-size:1.02rem; }}
    .metrics {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin-bottom:12px; }}
    .metric {{ padding:10px 11px; }}
    .metric .k {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.03em; }}
    .metric .v {{ margin-top:3px; font-weight:700; font-size:1.08rem; font-variant-numeric:tabular-nums; }}
    .controls {{
      display:grid; grid-template-columns:1.25fr .9fr .9fr .9fr; gap:10px; margin-bottom:12px;
    }}
    .controls label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }}
    .controls input,.controls select {{
      width:100%; padding:8px 10px; border-radius:10px; border:1px solid var(--stroke);
      background:#fffdf8; color:var(--ink); font-family:inherit;
    }}
    .panel {{ padding:14px; }}
    table {{ width:100%; border-collapse:collapse; font-size:.89rem; }}
    th {{
      text-align:left; color:var(--muted); font-size:.73rem; text-transform:uppercase; letter-spacing:.03em;
      border-bottom:1px solid var(--stroke); padding:7px 6px;
    }}
    td {{ border-bottom:1px dashed rgba(17,41,47,.12); padding:7px 6px; vertical-align:middle; }}
    .mono {{ font-variant-numeric:tabular-nums; }}
    .ok {{ color:var(--good); font-weight:600; }}
    .miss {{ color:var(--bad); font-weight:600; }}
    tr.row-hit td {{ background:rgba(11,122,117,.09); }}
    tr.row-miss td {{ background:rgba(158,71,62,.10); }}
    tr.row-close td {{ background:rgba(194,125,42,.10); }}
    .tag {{ color:#445d63; font-size:.82rem; line-height:1.2; max-width:320px; }}
    .note {{ margin-top:8px; color:var(--muted); font-size:.8rem; }}
    .explain h3 {{
      margin:0 0 8px;
      font-size:.88rem;
      text-transform:uppercase;
      letter-spacing:.03em;
      color:var(--muted);
    }}
    .explain ul {{
      margin:0;
      padding-left:18px;
      color:#2f464d;
      line-height:1.4;
      font-size:.87rem;
    }}
    @media (max-width:1020px) {{
      .metrics {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
      .controls {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <a class="back" href="index.html">Back to Index</a>
    <section class="hero">
      <div class="kicker">Dashboard 1</div>
      <h1>NBA Success Dashboard 1:<br/>Real Top-60 Picks vs Predicted Impact (__HOLDOUT_YEAR__ Holdout)</h1>
      <p>This view keeps only real top-60 draft picks and compares actual vs predicted NBA impact in a compact table layout.</p>
      <div id="successSummary" class="success-banner"></div>
    </section>
    <section class="metrics" id="metrics"></section>
    <section class="panel explain">
      <h3>Column Definitions</h3>
      <ul>
        <li><b>Actual Impact Rank</b>: Rank of each player's observed NBA impact score in the holdout set. Higher impact score gets better rank (#1 is best).</li>
        <li><b>Pred Impact Rank</b>: Rank of the model's predicted NBA impact score from college data. Higher predicted score gets better rank.</li>
        <li><b>Pred Rank (All Players)</b>: Same model score ranked against all players in that college season. This can be above 60.</li>
        <li><b>How scores are built</b>: The observed NBA impact score is the mean of available z-scores from each player's best two NBA seasons after draft across free impact metrics (LEBRON, DARKO DPM, MAMBA, RAPTOR, BRef BPM, BRef WS/48).</li>
        <li><b>Abs Err</b>: Absolute difference between Pred Impact Rank and Actual Impact Rank.</li>
        <li><b>Miss</b>: A covered player with absolute rank error strictly greater than __SUCCESS_THRESHOLD_INT__. A hit is <= __SUCCESS_THRESHOLD_INT__.</li>
        <li><b>Drivers (+/-)</b>: Strongest and weakest college-stat signals for this player relative to the same holdout cohort.</li>
      </ul>
    </section>
    <section class="panel">
      <div class="controls">
        <div><label for="q">Search player or college</label><input id="q" placeholder="e.g. Banchero, Duke, Auburn" /></div>
        <div><label for="outcome">Outcome filter</label><select id="outcome"><option value="all">All</option><option value="hit">Hits</option><option value="miss">Misses</option></select></div>
        <div><label for="maxpick">Max real pick shown</label><input id="maxpick" type="number" min="1" max="60" value="60" /></div>
        <div><label for="rowsN">Rows shown</label><input id="rowsN" type="number" min="10" max="120" value="60" /></div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Pick</th><th>Player</th><th>College</th><th>Actual Impact Rank</th><th>Pred Impact Rank</th><th>Pred Rank (All Players)</th><th>Abs Err</th><th>Actual RAPM Peak</th><th>Pred RAPM Peak</th><th>NBA RAPM</th><th>RAPM Rank</th><th>Outcome</th><th>Drivers (+)</th><th>Drivers (-)</th><th>Miss Explanation</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="note">Actual Impact Rank = observed NBA impact order in this holdout set. Pred Impact Rank = college-model prediction ranked within covered drafted players (used for hit/miss). Pred Rank (All Players) is ranked over the full season player pool and can exceed 60. Hit rule: absolute impact-rank error <= __SUCCESS_THRESHOLD_INT__. Table color: green hit, amber close miss (<=15), red miss (>15).</div>
    </section>
  </div>
  <script>
  const DATA = __RECORDS_JSON__;
  const MET = __METRICS_JSON__;
  const HIT_THRESHOLD = __SUCCESS_THRESHOLD_FLOAT__;
  const fmt = (v,d=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "n/a";
  const esc = (s) => String(s ?? "");
  const num = (v) => {{
    const x = Number(v);
    return Number.isFinite(x) ? x : NaN;
  }};
  const q = document.getElementById("q");
  const outcome = document.getElementById("outcome");
  const maxpick = document.getElementById("maxpick");
  const rowsN = document.getElementById("rowsN");
  const rows = document.getElementById("rows");
  const successSummary = document.getElementById("successSummary");

  document.getElementById("metrics").innerHTML = [
    ["Top-60 Picks In File", MET.real_top60_total],
    ["Model Coverage", MET.coverage],
    [`Hits (<=${Math.round(HIT_THRESHOLD)} err)`, `${MET.success_count}/${MET.eval_covered}`],
    ["Hit Rate", MET.success_rate],
    ["MAE (covered)", MET.mae_covered],
    ["Within 5", MET.within5]
  ].map(([k,v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  function isCovered(d) {
    return Number.isFinite(Number(d.pred_rank));
  }

  function isHit(d) {
    return isCovered(d) && Number(d.abs_rank_error) <= HIT_THRESHOLD;
  }

  function missDetail(d) {
    if (!isCovered(d) || isHit(d)) return "";
    return String(d.miss_explain_short || "").trim();
  }

  function posDriver(d) {
    return String(d.driver_positive_short || "").trim();
  }

  function negDriver(d) {
    return String(d.driver_negative_short || "").trim();
  }

  function actualRapmPeak(d) {
    const rapm = num((d.actual_rapm_best2_mean ?? d.rapm_best2_mean));
    if (Number.isFinite(rapm)) return rapm;
    const z = num(d.actual_impact_z);
    return Number.isFinite(z) ? z : NaN;
  }

  function predRapmPeak(d) {
    const rapm = num(d.pred_rapm_best2_mean);
    if (Number.isFinite(rapm)) return rapm;
    const z = num(d.pred_impact_z);
    return Number.isFinite(z) ? z : NaN;
  }

  function rowClass(d) {
    if (!isCovered(d)) return "row-miss";
    const e = Number(d.abs_rank_error);
    if (e <= HIT_THRESHOLD) return "row-hit";
    if (e <= 15) return "row-close";
    return "row-miss";
  }

  function filtered() {
    const qq = q.value.trim().toLowerCase();
    const out = outcome.value;
    const mp = Math.max(1, Math.min(60, parseInt(maxpick.value || "60", 10)));
    return DATA.filter(d => {
      if (Number(d.pick_number) > mp) return false;
      const txt = `${d.name || ""} ${d.team || ""}`.toLowerCase();
      const okQ = !qq || txt.includes(qq);
      const hit = isHit(d);
      const okO = out === "all" || (out === "hit" ? hit : !hit);
      return okQ && okO;
    });
  }

  function render() {
    const n = Math.max(10, Math.min(120, parseInt(rowsN.value || "60", 10)));
    const data = filtered().sort((a,b) => {
      const ar = Number(a.actual_rank);
      const br = Number(b.actual_rank);
      const aHas = Number.isFinite(ar);
      const bHas = Number.isFinite(br);
      if (aHas && bHas && ar !== br) return ar - br;
      if (aHas !== bHas) return aHas ? -1 : 1;
      const ap = Number(a.pick_number);
      const bp = Number(b.pick_number);
      const aPick = Number.isFinite(ap);
      const bPick = Number.isFinite(bp);
      if (aPick && bPick && ap !== bp) return ap - bp;
      if (aPick !== bPick) return aPick ? -1 : 1;
      return String(a.name || "").localeCompare(String(b.name || ""));
    });
    const shown = data.slice(0, n);

    const coveredShown = shown.filter(isCovered);
    const hitsShown = coveredShown.filter(isHit);
    const rateShown = coveredShown.length ? (100 * hitsShown.length / coveredShown.length).toFixed(1) : "0.0";
    successSummary.innerHTML =
      `<b>Hit Rate (current filter): ${hitsShown.length}/${coveredShown.length}</b> ` +
      `(${rateShown}%). Hit means abs impact-rank error <= ${Math.round(HIT_THRESHOLD)}.`;

    rows.innerHTML = shown.map(d => {
      const hit = isHit(d);
      const covered = isCovered(d);
      const outcomeTxt = hit ? `<span class="ok">Hit</span>` : `<span class="miss">${covered ? "Miss" : "Missing"}</span>`;
      return `<tr class="${rowClass(d)}">
        <td class="mono">${Number.isFinite(Number(d.pick_number)) ? Number(d.pick_number) : "n/a"}</td>
        <td>${esc(d.name)}</td>
        <td>${esc(d.team)}</td>
        <td class="mono">${Number.isFinite(Number(d.actual_rank)) ? Number(d.actual_rank) : "n/a"}</td>
        <td class="mono">${Number.isFinite(Number(d.pred_rank)) ? Number(d.pred_rank) : "n/a"}</td>
        <td class="mono">${Number.isFinite(Number(d.pred_rank_all_players)) ? Number(d.pred_rank_all_players) : "n/a"}</td>
        <td class="mono">${covered ? fmt(d.abs_rank_error,0) : "n/a"}</td>
        <td class="mono">${fmt(actualRapmPeak(d),3)}</td>
        <td class="mono">${fmt(predRapmPeak(d),3)}</td>
        <td class="mono">${Number.isFinite(Number(d.nba_rapm_current)) ? fmt(d.nba_rapm_current,2) : "n/a"}</td>
        <td class="mono">${Number.isFinite(Number(d.nba_rapm_rank_current)) ? fmt(d.nba_rapm_rank_current,0) : "n/a"}</td>
        <td>${outcomeTxt}</td>
        <td class="tag">${esc(posDriver(d))}</td>
        <td class="tag">${esc(negDriver(d))}</td>
        <td class="tag">${esc(missDetail(d))}</td>
      </tr>`;
    }).join("");

  }

  [q, outcome, maxpick, rowsN].forEach(el => el.addEventListener("input", render));
  render();
  </script>
</body>
</html>
"""
    html = html.replace("{{", "{").replace("}}", "}")
    html = html.replace("__RECORDS_JSON__", records_json)
    html = html.replace("__METRICS_JSON__", metrics_json)
    html = html.replace("__SUCCESS_THRESHOLD_FLOAT__", f"{success_threshold:.6f}")
    html = html.replace("__SUCCESS_THRESHOLD_INT__", f"{success_threshold:.0f}")
    html = html.replace("__HOLDOUT_YEAR__", holdout_year)
    return html


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input CSV: {args.input_csv}")

    df = pd.read_csv(args.input_csv, low_memory=False)
    _to_num(
        df,
        [
            "pick_number",
            "actual_rank",
            "pred_rank",
            "pred_rank_drafted",
            "pred_rank_all_players",
            "rank_error",
            "abs_rank_error",
            "minutes",
            "usage",
            "true_shooting_pct",
            "assists_per40",
            "turnovers_per40",
            "rebounds_total_per40",
            "blocks_per40",
            "steals_per40",
            "points_per40",
            "nba_rapm_current",
            "nba_rapm_rank_current",
        ],
    )

    top60 = df[df["pick_number"].notna() & (df["pick_number"] <= 60)].copy()
    if args.rapm_csv.exists():
        rapm = pd.read_csv(args.rapm_csv, low_memory=False)
        for c in ["player_name", "rapm", "rapm_rank"]:
            if c not in rapm.columns:
                rapm = pd.DataFrame()
                break
        if not rapm.empty:
            rapm = rapm[["player_name", "rapm", "rapm_rank"]].copy()
            rapm["name_key"] = rapm["player_name"].map(_name_key)
            rapm = rapm.sort_values(["rapm_rank", "rapm"], kind="stable").drop_duplicates(
                subset=["name_key"], keep="first"
            )
            rapm = rapm.rename(
                columns={"rapm": "nba_rapm_current", "rapm_rank": "nba_rapm_rank_current"}
            )
            top60["name_key"] = top60["name"].map(_name_key)
            top60 = top60.merge(
                rapm[["name_key", "nba_rapm_current", "nba_rapm_rank_current"]],
                on="name_key",
                how="left",
            )

    driver_df = _build_driver_and_miss_text(top60, args.success_threshold)
    top60 = top60.join(driver_df)

    covered = top60[top60["pred_rank"].notna()].copy()
    hits = covered[covered["abs_rank_error"] <= args.success_threshold].copy()

    coverage = f"{len(covered):,}/{len(top60):,}"
    success_rate = f"{(100.0 * len(hits) / len(covered)):.1f}%" if len(covered) else "n/a"

    metrics = {
        "real_top60_total": f"{len(top60):,}",
        "coverage": coverage,
        "success_count": f"{len(hits):,}",
        "eval_covered": f"{len(covered):,}",
        "success_rate": success_rate,
        "mae_covered": f"{covered['abs_rank_error'].mean():.2f}" if len(covered) else "n/a",
        "within5": f"{int((covered['abs_rank_error'] <= 5).sum()):,}" if len(covered) else "n/a",
    }

    keep_cols = [
        "name",
        "team",
        "pick_number",
        "actual_rank",
        "pred_rank",
        "pred_rank_drafted",
        "pred_rank_all_players",
        "rank_error",
        "abs_rank_error",
        "actual_rapm_best2_mean",
        "pred_rapm_best2_mean",
        "rapm_best2_mean",
        "minutes",
        "usage",
        "true_shooting_pct",
        "points_per40",
        "assists_per40",
        "turnovers_per40",
        "rebounds_total_per40",
        "blocks_per40",
        "steals_per40",
        "nba_rapm_current",
        "nba_rapm_rank_current",
        "driver_positive_short",
        "driver_negative_short",
        "miss_explain_short",
    ]
    keep_cols = [c for c in keep_cols if c in top60.columns]
    records = (
        top60[keep_cols]
        .sort_values(["actual_rank", "pick_number", "name"], kind="stable", na_position="last")
        .to_dict(orient="records")
    )

    holdout_year = (
        str(int(pd.to_numeric(top60["season"], errors="coerce").dropna().iloc[0]))
        if ("season" in top60.columns and not top60.empty)
        else "n/a"
    )

    html = build_html(
        records_json=json.dumps(records, ensure_ascii=True, separators=(",", ":")),
        metrics_json=json.dumps(metrics, ensure_ascii=True, separators=(",", ":")),
        success_threshold=args.success_threshold,
        holdout_year=holdout_year,
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
