from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Dashboard 1 HTML for real 2025 picks vs model predictions."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nba_draft_holdout_2025_actual_top60_with_model_coverage.csv"),
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("data/processed/viz/nba_draft_dashboard1_real_vs_predicted_2025.html"),
    )
    parser.add_argument(
        "--miss-debug-csv",
        type=Path,
        default=Path("data/processed/nba_draft_holdout_2025_miss_root_causes.csv"),
    )
    return parser.parse_args()


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_html(records_json: str, metrics_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Draft Dashboard 1: Real vs Predicted Picks (2025)</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
    :root {{
      --bg0:#f3efe7; --bg1:#e1d7c5; --ink:#12292f; --muted:#5b6d70;
      --card:rgba(255,253,247,0.78); --stroke:rgba(18,41,47,0.18); --good:#0b7a75; --warn:#c27d2a;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); font-family:"Space Grotesk",sans-serif;
      background:radial-gradient(1200px 480px at 5% -10%, rgba(11,122,117,.12), transparent 60%),
                 radial-gradient(900px 420px at 95% -10%, rgba(194,125,42,.16), transparent 58%),
                 linear-gradient(160deg,var(--bg0) 0%,var(--bg1) 100%);
      min-height:100vh;
    }}
    .wrap {{ max-width:1240px; margin:0 auto; padding:24px; }}
    .hero,.metric,.panel {{
      border:1px solid var(--stroke); border-radius:16px; background:var(--card);
    }}
    .hero {{ padding:18px 20px; margin-bottom:12px; }}
    .hero h1 {{ margin:0; font-size:clamp(1.4rem,2.4vw,2.05rem); font-family:"Fraunces",serif; }}
    .hero p {{ margin:8px 0 0; color:var(--muted); }}
    .back {{
      display:inline-block; margin-bottom:12px; text-decoration:none; color:var(--ink);
      border:1px solid var(--stroke); border-radius:10px; padding:7px 11px; background:rgba(255,255,255,.72);
      font-size:.88rem;
    }}
    .metrics {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin-bottom:12px; }}
    .metric {{ padding:10px 11px; }}
    .metric .k {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.03em; }}
    .metric .v {{ margin-top:3px; font-weight:700; font-size:1.08rem; font-variant-numeric:tabular-nums; }}
    .controls {{
      display:grid; grid-template-columns:1.2fr .9fr .9fr .9fr; gap:10px; margin-bottom:12px;
    }}
    .controls label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }}
    .controls input,.controls select {{
      width:100%; padding:8px 10px; border-radius:10px; border:1px solid var(--stroke);
      background:#fffdf8; color:var(--ink); font-family:inherit;
    }}
    .grid {{ display:grid; grid-template-columns:1.05fr 1fr; gap:12px; }}
    .panel {{ padding:14px; }}
    table {{ width:100%; border-collapse:collapse; font-size:.89rem; }}
    th {{
      text-align:left; color:var(--muted); font-size:.73rem; text-transform:uppercase; letter-spacing:.03em;
      border-bottom:1px solid var(--stroke); padding:7px 6px;
    }}
    td {{ border-bottom:1px dashed rgba(18,41,47,.12); padding:7px 6px; vertical-align:middle; }}
    .mono {{ font-variant-numeric:tabular-nums; }}
    .ok {{ color:var(--good); font-weight:600; }}
    .miss {{ color:#9f5a1a; font-weight:600; }}
    .success-banner {{
      margin-top:10px; padding:10px 12px; border-radius:12px;
      border:1px solid rgba(11,122,117,.28); background:rgba(11,122,117,.12);
      font-size:.92rem;
    }}
    .success-banner b {{ font-size:1.02rem; }}
    tr.row-success td {{ background:rgba(11,122,117,.09); }}
    tr.row-fail td {{ background:rgba(161,75,67,.10); }}
    tr.row-missing td {{ background:rgba(148,163,184,.12); }}
    .reason {{ color:#445b61; font-size:.82rem; line-height:1.25; max-width:340px; }}
    .note {{ margin-top:8px; color:var(--muted); font-size:.8rem; }}
    @media (max-width:1020px) {{
      .metrics {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
      .controls {{ grid-template-columns:1fr; }}
      .grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <a class="back" href="index.html">Back to Index</a>
    <section class="hero">
      <h1>Draft Dashboard 1: Real vs Predicted Picks (2025 Holdout)</h1>
      <p>This compares real 2025 draft picks to model outputs. X-axis is real pick; Y-axis is model expected pick. The diagonal is perfect prediction.</p>
      <div id="successSummary" class="success-banner"></div>
    </section>
    <section class="metrics" id="metrics"></section>
    <section class="panel">
      <div class="controls">
        <div><label for="q">Search player/college/team</label><input id="q" placeholder="e.g. Flagg, Duke, Rutgers" /></div>
        <div><label for="coverage">Coverage filter</label><select id="coverage"><option value="all">All</option><option value="covered">Model covered</option><option value="missing">Model missing</option></select></div>
        <div><label for="maxpick">Max real pick shown</label><input id="maxpick" type="number" min="1" max="60" value="60" /></div>
        <div><label for="rowsN">Rows shown</label><input id="rowsN" type="number" min="10" max="120" value="60" /></div>
      </div>
      <div class="grid">
        <div class="panel">
          <svg id="plot" width="100%" viewBox="0 0 940 620" preserveAspectRatio="xMidYMid meet"></svg>
          <div id="tip" class="note">Hover point for details.</div>
        </div>
        <div class="panel">
          <table>
            <thead>
              <tr>
                <th>Pick</th><th>Player</th><th>College</th><th>Covered</th><th>Outcome</th><th>P(D)</th><th>Expected</th><th>Error</th><th>Rank</th><th>Miss Tag</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
      <div class="note">Success means a covered real top-60 pick is ranked in the model's top-60 board (`pred_rank_expected_pick <= 60`). Missing rows are mostly international/non-college prospects not present in the NCAA feature universe.</div>
    </section>
  </div>
  <script>
  const DATA = {records_json};
  const MET = {metrics_json};
  const fmt = (v,d=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "n/a";
  const pct = (v) => Number.isFinite(Number(v)) ? `${{(100*Number(v)).toFixed(1)}}%` : "n/a";
  const esc = (s) => String(s ?? "");
  const q = document.getElementById("q");
  const coverage = document.getElementById("coverage");
  const maxpick = document.getElementById("maxpick");
  const rowsN = document.getElementById("rowsN");
  const rows = document.getElementById("rows");
  const plot = document.getElementById("plot");
  const tip = document.getElementById("tip");
  const successSummary = document.getElementById("successSummary");

  const cards = [
    ["Real Picks In File", MET.real_total],
    ["Model Coverage", MET.coverage],
    ["Top-60 Success (covered)", `${{MET.success_count}}/${{MET.eval_covered}} (${{MET.success_rate}})`],
    ["Covered MAE", MET.mae_covered],
    ["Covered Median Error", MET.medae_covered],
    ["Within +/-5 Picks", MET.within5],
    ["Within +/-10 Picks", MET.within10]
  ];
  document.getElementById("metrics").innerHTML = cards.map(([k,v]) =>
    `<div class="metric"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`
  ).join("");

  function scale(v, lo, hi, a, b) {{
    if (!Number.isFinite(v) || hi <= lo) return (a+b)/2;
    return a + ((v - lo) * (b - a)) / (hi - lo);
  }}

  function isCoveredForEval(d) {{
    return Number(d.model_found) === 1 && Number.isFinite(Number(d.pred_rank_expected_pick));
  }}

  function isSuccess(d) {{
    return isCoveredForEval(d) && Number(d.pred_rank_expected_pick) <= 60;
  }}

  function reasonText(d) {{
    const found = Number(d.model_found) === 1;
    const rank = Number(d.pred_rank_expected_pick);
    const prob = Number(d.p_drafted_pred);
    const pickIf = Number(d.pred_pick_if_drafted);
    const expected = Number(d.expected_pick_pred);
    const mins = Number(d.minutes);
    const probPct = Number.isFinite(prob) ? 100.0 * prob : NaN;
    const undraftedPct = Number.isFinite(probPct) ? 100.0 - probPct : NaN;
    const minTxt = Number.isFinite(mins) ? `${{Math.round(mins)}} minutes` : "minutes n/a";

    if (!found) {{
      return "We do not have enough college data for this player in this model, so we cannot score or rank him here.";
    }}
    if (!isCoveredForEval(d)) {{
      return `We found this player, but the final board rank is missing. Current model output: draft chance ${{Number.isFinite(probPct) ? probPct.toFixed(1) + "%" : "n/a"}}, expected pick ${{Number.isFinite(expected) ? "#" + expected.toFixed(1) : "n/a"}}, playing time ${{minTxt}}.`;
    }}
    if (isSuccess(d)) {{
      return `This is a hit. The player is on our top-60 board at #${{Math.round(rank)}}. Model view: draft chance ${{Number.isFinite(probPct) ? probPct.toFixed(1) + "%" : "n/a"}}, likely pick if drafted ${{Number.isFinite(pickIf) ? "#" + pickIf.toFixed(1) : "n/a"}}, expected pick ${{Number.isFinite(expected) ? "#" + expected.toFixed(1) : "n/a"}}, playing time ${{minTxt}}.`;
    }}
    const rankGap = Number.isFinite(rank) ? Math.round(rank - 60) : NaN;
    const expectedFormula = (Number.isFinite(probPct) && Number.isFinite(pickIf))
      ? `${{probPct.toFixed(1)}}%*${{pickIf.toFixed(1)}} + ${{undraftedPct.toFixed(1)}}%*61`
      : "n/a";
    const lowSample = Number.isFinite(mins) && mins < 300 ? " (low sample)" : "";
    return `This is a miss. We ranked this player at #${{Number.isFinite(rank) ? Math.round(rank) : "n/a"}}, which is ${{Number.isFinite(rankGap) ? rankGap : "n/a"}} spots below the top-60 line. Model view: draft chance ${{Number.isFinite(probPct) ? probPct.toFixed(1) + "%" : "n/a"}}, likely pick if drafted ${{Number.isFinite(pickIf) ? "#" + pickIf.toFixed(1) : "n/a"}}, expected pick ${{Number.isFinite(expected) ? "#" + expected.toFixed(1) : "n/a"}}. In simple terms, expected pick uses ${{expectedFormula}}. Playing time: ${{minTxt}}${{lowSample}}.`;
  }}

  function missTag(d) {{
    if (!isCoveredForEval(d) || isSuccess(d)) {{
      return "";
    }}
    const dbgRaw = d.miss_tag_debug;
    const dbg = (dbgRaw === null || dbgRaw === undefined || Number.isNaN(dbgRaw)) ? "" : String(dbgRaw).trim();
    if (dbg && dbg.toLowerCase() !== "nan") {{
      return dbg;
    }}
    const mins = Number(d.minutes);
    const prob = Number(d.p_drafted_pred);
    const pickIf = Number(d.pred_pick_if_drafted);
    const expected = Number(d.expected_pick_pred);
    const rank = Number(d.pred_rank_expected_pick);
    if (Number.isFinite(mins) && mins < 300) {{
      return `Low play time: ${{Math.round(mins)}}m`;
    }}
    if (Number.isFinite(prob) && prob < 0.30) {{
      return `Low draft chance: ${{(100 * prob).toFixed(1)}}%`;
    }}
    if (Number.isFinite(pickIf) && pickIf > 45) {{
      return `Late pick if drafted: #${{pickIf.toFixed(1)}}`;
    }}
    if (Number.isFinite(prob)) {{
      return `Low draft chance: ${{(100 * prob).toFixed(1)}}%`;
    }}
    return "";
  }}

  function filtered() {{
    const qq = q.value.trim().toLowerCase();
    const cov = coverage.value;
    const mp = Math.max(1, Math.min(60, parseInt(maxpick.value || "60", 10)));
    return DATA.filter(d => {{
      if (Number(d.pick_overall) > mp) return false;
      const hitText = `${{d.player_name || ""}} ${{d.college_name || ""}} ${{d.team || ""}}`.toLowerCase();
      const okQ = !qq || hitText.includes(qq);
      const found = Number(d.model_found) === 1;
      const okC = cov === "all" || (cov === "covered" ? found : !found);
      return okQ && okC;
    }});
  }}

  function render() {{
    const n = Math.max(10, Math.min(120, parseInt(rowsN.value || "60", 10)));
    const data = filtered().sort((a,b) => Number(a.pick_overall) - Number(b.pick_overall));
    const shown = data.slice(0, n);
    const evalCovered = data.filter(isCoveredForEval);
    const successCount = evalCovered.filter(isSuccess).length;
    const successRate = evalCovered.length ? `${{(100 * successCount / evalCovered.length).toFixed(1)}}%` : "n/a";
    successSummary.innerHTML = evalCovered.length
      ? `<b>${{successCount}} / ${{evalCovered.length}}</b> successful predictions under current filters (${{successRate}}). Missing rows are excluded.`
      : `No covered rows under current filters. Missing rows are excluded from success/unsuccessful counts.`;

    rows.innerHTML = shown.map(d => {{
      const found = Number(d.model_found) === 1;
      const coveredEval = isCoveredForEval(d);
      const success = isSuccess(d);
      const err = found ? Math.abs(Number(d.expected_pick_pred) - Number(d.pick_overall)) : NaN;
      const rowClass = !coveredEval ? "row-missing" : (success ? "row-success" : "row-fail");
      const outcome = !coveredEval ? "n/a" : (success ? "Success" : "Unsuccessful");
      const reason = reasonText(d);
      const tag = missTag(d);
      return `<tr class="${{rowClass}}">
        <td class="mono">${{Number(d.pick_overall)}}</td>
        <td>${{esc(d.player_name)}}</td>
        <td>${{esc(d.college_name)}}</td>
        <td class="${{found ? "ok" : "miss"}}">${{found ? "Yes" : "No"}}</td>
        <td class="${{success ? "ok" : "miss"}}">${{outcome}}</td>
        <td class="mono">${{pct(d.p_drafted_pred)}}</td>
        <td class="mono">${{fmt(d.expected_pick_pred,1)}}</td>
        <td class="mono">${{Number.isFinite(err) ? fmt(err,1) : "n/a"}}</td>
        <td class="mono">${{Number.isFinite(Number(d.pred_rank_expected_pick)) ? Number(d.pred_rank_expected_pick) : "n/a"}}</td>
        <td class="reason">${{esc(tag)}}</td>
      </tr>`;
    }}).join("");

    const W = 940, H = 620, M = {{l:62,r:24,t:24,b:54}}, PW = W - M.l - M.r, PH = H - M.t - M.b;
    const covered = data.filter(d => Number(d.model_found) === 1 && Number.isFinite(Number(d.expected_pick_pred)));
    if (!covered.length) {{
      plot.innerHTML = `<text x="${{W/2}}" y="${{H/2}}" text-anchor="middle" fill="#5b6d70">No covered rows under current filters</text>`;
      return;
    }}
    const xLo = 1, xHi = Math.max(...covered.map(d => Number(d.pick_overall)), 60);
    const yLo = 1, yHi = 60;
    let svg = `<rect x="0" y="0" width="${{W}}" height="${{H}}" fill="rgba(255,255,255,.72)" />`;
    for (let i=0; i<=5; i++) {{
      const x = M.l + (i/5)*PW;
      const y = M.t + (i/5)*PH;
      svg += `<line x1="${{x}}" y1="${{M.t}}" x2="${{x}}" y2="${{M.t+PH}}" stroke="rgba(18,41,47,.07)" />`;
      svg += `<line x1="${{M.l}}" y1="${{y}}" x2="${{M.l+PW}}" y2="${{y}}" stroke="rgba(18,41,47,.07)" />`;
    }}
    const x1 = scale(1, xLo, xHi, M.l, M.l + PW);
    const y1 = scale(1, yLo, yHi, M.t, M.t + PH);
    const x2 = scale(60, xLo, xHi, M.l, M.l + PW);
    const y2 = scale(60, yLo, yHi, M.t, M.t + PH);
    svg += `<line x1="${{x1}}" y1="${{y1}}" x2="${{x2}}" y2="${{y2}}" stroke="rgba(11,122,117,.5)" stroke-width="1.4" stroke-dasharray="6 5" />`;
    covered.forEach(d => {{
      const x = scale(Number(d.pick_overall), xLo, xHi, M.l, M.l + PW);
      const y = scale(Number(d.expected_pick_pred), yLo, yHi, M.t, M.t + PH);
      const err = Math.abs(Number(d.expected_pick_pred) - Number(d.pick_overall));
      const success = isSuccess(d);
      const c = success ? "#0b7a75" : "#a14b43";
      const status = success ? "success" : "unsuccessful";
      const reason = reasonText(d);
      svg += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="5.1" fill="${{c}}" stroke="rgba(18,41,47,.34)" stroke-width="0.9"
        data-name="${{esc(d.player_name)}}" data-college="${{esc(d.college_name)}}" data-real="${{Number(d.pick_overall)}}" data-exp="${{fmt(d.expected_pick_pred,1)}}" data-prob="${{pct(d.p_drafted_pred)}}" data-err="${{fmt(err,1)}}" data-status="${{status}}" data-rank="${{Number(d.pred_rank_expected_pick)}}" data-reason="${{esc(reason)}}" />`;
    }});
    svg += `<text x="${{W/2}}" y="${{H-14}}" text-anchor="middle" font-size="13" fill="#5b6d70">Real pick number (2025 draft)</text>`;
    svg += `<text x="18" y="${{H/2}}" transform="rotate(-90,18,${{H/2}})" text-anchor="middle" font-size="13" fill="#5b6d70">Model expected pick</text>`;
    plot.innerHTML = svg;

    plot.querySelectorAll("circle").forEach(c => c.addEventListener("mouseenter", () => {{
      tip.textContent = `${{c.dataset.name}} (${{c.dataset.college || "n/a"}}) | real #${{c.dataset.real}} | expected #${{c.dataset.exp}} | rank=${{c.dataset.rank}} | ${{c.dataset.status}} | error=${{c.dataset.err}} | p(drafted)=${{c.dataset.prob}} | reason: ${{c.dataset.reason}}`;
    }}));
  }}

  [q, coverage, maxpick, rowsN].forEach(el => el.addEventListener("input", render));
  render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input CSV: {args.input_csv}")

    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.miss_debug_csv.exists():
        dbg = pd.read_csv(args.miss_debug_csv, low_memory=False)
        if "pick_overall" in dbg.columns and "miss_tag_debug" in dbg.columns:
            dbg = dbg[["pick_overall", "miss_tag_debug"]].copy()
            dbg["pick_overall"] = _safe_num(dbg["pick_overall"])
            # Each draft pick is unique in this holdout table.
            dbg = dbg.dropna(subset=["pick_overall"]).drop_duplicates(subset=["pick_overall"], keep="first")
            df["pick_overall"] = _safe_num(df.get("pick_overall"))
            df = df.merge(dbg, how="left", on="pick_overall")
    for c in [
        "pick_overall",
        "model_found",
        "p_drafted_pred",
        "expected_pick_pred",
        "pred_rank_expected_pick",
    ]:
        if c in df.columns:
            df[c] = _safe_num(df[c])

    df["abs_pick_error"] = (df["expected_pick_pred"] - df["pick_overall"]).abs()

    covered = df[df["model_found"] == 1].copy()
    eval_covered = covered[covered["pred_rank_expected_pick"].notna()].copy()
    success_count = int((eval_covered["pred_rank_expected_pick"] <= 60).sum()) if not eval_covered.empty else 0
    within5 = int((covered["abs_pick_error"] <= 5).sum()) if not covered.empty else 0
    within10 = int((covered["abs_pick_error"] <= 10).sum()) if not covered.empty else 0
    metrics = {
        "real_total": f"{len(df):,}",
        "coverage": f"{int((df['model_found'] == 1).sum()):,}/{len(df):,}",
        "success_count": f"{success_count:,}",
        "eval_covered": f"{len(eval_covered):,}",
        "success_rate": f"{(100.0 * success_count / len(eval_covered)):.1f}%" if len(eval_covered) else "n/a",
        "mae_covered": f"{covered['abs_pick_error'].mean():.2f}" if not covered.empty else "n/a",
        "medae_covered": f"{covered['abs_pick_error'].median():.2f}" if not covered.empty else "n/a",
        "within5": f"{within5:,}",
        "within10": f"{within10:,}",
    }

    keep = [
        "pick_overall",
        "player_name",
        "college_name",
        "model_found",
        "team",
        "minutes",
        "p_drafted_pred",
        "pred_pick_if_drafted",
        "expected_pick_pred",
        "pred_rank_expected_pick",
        "miss_tag_debug",
    ]
    keep = [c for c in keep if c in df.columns]
    records = df[keep].copy().sort_values("pick_overall").to_dict(orient="records")

    html = build_html(
        records_json=json.dumps(records, ensure_ascii=True, separators=(",", ":")),
        metrics_json=json.dumps(metrics, ensure_ascii=True, separators=(",", ":")),
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
