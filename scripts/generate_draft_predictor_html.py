from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML dashboard for NBA draft predictor results.")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("data/processed/nba_draft_predictions_season_2026.csv"),
    )
    parser.add_argument(
        "--report-txt",
        type=Path,
        default=Path("data/processed/nba_draft_model_report.txt"),
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("data/processed/viz/nba_draft_predictor_season_2026.html"),
    )
    parser.add_argument("--top-n", type=int, default=60)
    return parser.parse_args()


def extract_metric(text: str, label: str, cast=float) -> str:
    m = re.search(rf"{re.escape(label)}:\s*([0-9\.,]+)", text)
    if not m:
        return "n/a"
    raw = m.group(1).replace(",", "")
    try:
        v = cast(raw)
    except Exception:
        return "n/a"
    if cast is int:
        return f"{v:,}"
    if isinstance(v, float):
        if label in {"MAE", "RMSE"}:
            return f"{v:.3f}"
        return f"{v:.4f}"
    return str(v)


def build_html(records_json: str, metrics_json: str, top_n: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NBA Draft Predictor Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
    :root {{
      --bg-0:#f4efe6; --bg-1:#e7dcc9; --ink:#132a2f; --muted:#5e6f71;
      --card:rgba(255,252,245,0.78); --stroke:rgba(22,55,58,0.18); --teal:#0b7a75;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; font-family:"Space Grotesk",sans-serif; color:var(--ink);
      background:radial-gradient(1200px 500px at 10% -10%, rgba(11,122,117,.15), transparent 60%),
                 radial-gradient(900px 450px at 90% -20%, rgba(199,127,43,.18), transparent 58%),
                 linear-gradient(165deg,var(--bg-0) 0%,var(--bg-1) 100%);
      min-height:100vh;
    }}
    .wrap {{ max-width:1240px; margin:0 auto; padding:24px; }}
    .top-nav {{ margin-bottom:12px; }}
    .back-link {{
      display:inline-block; text-decoration:none; font-size:.86rem; color:var(--ink);
      border:1px solid var(--stroke); background:rgba(255,255,255,.72); padding:7px 11px; border-radius:10px;
    }}
    .hero,.card,.metric {{
      border:1px solid var(--stroke); background:var(--card); border-radius:16px;
    }}
    .hero {{ padding:18px 20px; margin-bottom:12px; }}
    .hero h1 {{ margin:0; font-family:"Fraunces",serif; font-size:clamp(1.45rem,2.4vw,2.15rem); }}
    .hero p {{ margin:8px 0 0; color:var(--muted); }}
    .narrative {{
      margin-top:10px; padding:10px 12px; border-radius:12px;
      border:1px solid var(--stroke); background:rgba(255,255,255,.62); color:var(--muted);
      font-size:.88rem; line-height:1.45;
    }}
    .narrative p {{ margin: 0 0 8px; }}
    .narrative ul {{ margin: 6px 0 0 18px; padding: 0; }}
    .narrative li {{ margin: 2px 0; }}
    .metrics {{ display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:10px; margin-bottom:12px; }}
    .metric {{ padding:10px 11px; }}
    .metric .k {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.03em; }}
    .metric .v {{ font-size:1.08rem; font-weight:700; margin-top:3px; font-variant-numeric:tabular-nums; }}
    .card {{ padding:14px; }}
    .summary-list {{ margin:8px 0 0; padding-left:18px; color:var(--muted); }}
    .controls {{
      display:grid; grid-template-columns:1.2fr 1fr 1fr .8fr .8fr; gap:10px; margin-bottom:12px;
    }}
    .controls label {{ display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }}
    .controls input,.controls select {{
      width:100%; padding:8px 10px; border-radius:10px; border:1px solid var(--stroke);
      background:#fffdf8; color:var(--ink); font-family:inherit;
    }}
    .grid-2 {{ display:grid; grid-template-columns:1.2fr 1fr; gap:12px; }}
    .canvas-wrap {{ border:1px solid var(--stroke); border-radius:14px; background:rgba(255,255,255,.72); overflow:hidden; }}
    table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
    thead th {{
      text-align:left; color:var(--muted); font-size:.74rem; font-weight:600; padding:8px 6px;
      border-bottom:1px solid var(--stroke); text-transform:uppercase; letter-spacing:.02em;
    }}
    tbody td {{ padding:8px 6px; border-bottom:1px dashed rgba(19,42,47,.12); vertical-align:middle; }}
    .mono {{ font-variant-numeric:tabular-nums; }}
    .footer-note {{ margin-top:10px; font-size:.79rem; color:var(--muted); }}
    @media (max-width:1050px) {{
      .metrics {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
      .controls {{ grid-template-columns:1fr; }}
      .grid-2 {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top-nav"><a class="back-link" href="index.html">Back to Index</a></div>
    <section class="hero">
      <h1>NBA Draft Predictor (2026)</h1>
      <p>Two-stage hurdle model. Stage 1 predicts <strong>drafted probability</strong>; stage 2 predicts <strong>pick number if drafted</strong>. Final ranking uses expected pick.</p>
      <div class="narrative" id="narrative"></div>
      <ul class="summary-list" id="topSummary"></ul>
    </section>
    <section class="metrics" id="metrics"></section>
    <section class="card">
      <div class="controls">
        <div><label for="q">Search player/team</label><input id="q" placeholder="e.g. Boozer, Duke" /></div>
        <div><label for="conf">Conference</label><select id="conf"></select></div>
        <div><label for="pos">Position</label><select id="pos"></select></div>
        <div><label for="prob">Min p(drafted)</label><input id="prob" type="number" min="0" max="1" step="0.01" value="0.10" /></div>
        <div><label for="topn">Rows shown</label><input id="topn" type="number" min="10" max="300" value="{int(top_n)}" /></div>
      </div>
      <div class="grid-2">
        <div class="card canvas-wrap">
          <svg id="plot" width="100%" viewBox="0 0 980 620" preserveAspectRatio="xMidYMid meet"></svg>
          <div id="tip" class="footer-note">Hover a point for details.</div>
        </div>
        <div class="card">
          <table>
            <thead><tr><th>Rank</th><th>Player</th><th>Team</th><th>P(D)</th><th>Pick If</th><th>Expected</th><th>Pts/40</th><th>Ast/40</th><th>Reb/40</th><th>TS%</th></tr></thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
  <script>
  const DATA = {records_json};
  const MET = {metrics_json};
  const q = document.getElementById("q"), conf = document.getElementById("conf"), pos = document.getElementById("pos");
  const prob = document.getElementById("prob"), topn = document.getElementById("topn");
  const rows = document.getElementById("rows"), plot = document.getElementById("plot"), tip = document.getElementById("tip");

  const cardItems = [
    ["Draft vs Not Drafted (separation)", MET.roc_auc], ["Draft Board Precision", MET.pr_auc], ["Probability Error", MET.brier],
    ["Average Pick Miss", MET.mae], ["Pick Miss Spread", MET.rmse], ["Real Drafted (2025)", MET.actual_drafted], ["Found in Top-60 Board", MET.captured_top60]
  ];
  document.getElementById("metrics").innerHTML = cardItems.map(([k,v]) => `<div class="metric"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");
  const narrative = document.getElementById("narrative");
  narrative.innerHTML = `
    <p><strong>Executive Summary (Plain Language):</strong> This page ranks prospects using a two-step model built from college statistics only. First, it estimates each player's <strong>chance of being drafted</strong>. Second, for players likely to be drafted, it estimates <strong>where they would be picked</strong>. Those two pieces are combined into one number: <strong>expected draft slot</strong> (lower is better).</p>
    <p><strong>How good was it on the 2025 test year?</strong> The model was strong at telling apart drafted vs non-drafted players (separation score ${{MET.roc_auc}}). It was directionally useful for building a board, but still missed many late and surprise picks: it found <strong>${{MET.captured_top60}} of ${{MET.actual_drafted}}</strong> real draftees in its top-60 probability board. For exact pick placement, the average miss was about <strong>${{MET.mae}} picks</strong>.</p>
    <ul>
      <li><strong>Use it for:</strong> tiering prospects, comparing players, and finding undervalued candidates.</li>
      <li><strong>Do not use it as:</strong> a final mock draft with exact pick certainty.</li>
      <li><strong>Why:</strong> team fit, interviews, medicals, role context, and late-cycle info are not in this pure-stats model.</li>
    </ul>
  `;
  const top10 = [...DATA].sort((a,b)=>(a.expected_pick-b.expected_pick)||(b.p_drafted-a.p_drafted)).slice(0,10);
  document.getElementById("topSummary").innerHTML = top10.map(d => `<li>${{d.name}} (${{d.team}}): p(drafted)=${{(100*d.p_drafted).toFixed(1)}}%, expected pick=${{d.expected_pick.toFixed(1)}}</li>`).join("");

  conf.innerHTML = ["All", ...Array.from(new Set(DATA.map(d => d.conference || "Unknown"))).sort()].map(x => `<option>${{x}}</option>`).join("");
  pos.innerHTML = ["All", ...Array.from(new Set(DATA.map(d => d.position_group || "Unknown"))).sort()].map(x => `<option>${{x}}</option>`).join("");

  const f = (v,d=3)=>Number.isFinite(Number(v))?Number(v).toFixed(d):"n/a";
  const pct = v=>Number.isFinite(Number(v))?`${{(100*Number(v)).toFixed(1)}}%`:"n/a";
  const clr = p=>{{ p=Math.max(0,Math.min(1,Number(p)||0)); const r=Math.round((1-p)*177+p*11), g=Math.round((1-p)*78+p*122), b=Math.round((1-p)*65+p*117); return `rgb(${{r}},${{g}},${{b}})`; }};
  const scale=(v,lo,hi,a,b)=>hi<=lo?(a+b)/2:a+(v-lo)*(b-a)/(hi-lo);

  function filtered() {{
    const qq=q.value.trim().toLowerCase(), c=conf.value, p=pos.value, minP=Math.max(0,Math.min(1,Number(prob.value)||0));
    return DATA.filter(d => {{
      const okQ=!qq || d.name.toLowerCase().includes(qq) || d.team.toLowerCase().includes(qq);
      const okC=c==="All" || (d.conference||"Unknown")===c;
      const okP=p==="All" || (d.position_group||"Unknown")===p;
      return okQ && okC && okP && Number(d.p_drafted)>=minP;
    }});
  }}

  function render() {{
    const n=Math.max(10,Math.min(300,parseInt(topn.value||"60",10)));
    const data=filtered().sort((a,b)=>(a.expected_pick-b.expected_pick)||(b.p_drafted-a.p_drafted));
    const show=data.slice(0,n);
    rows.innerHTML = show.map((d,i)=>`
      <tr><td class="mono">${{i+1}}</td><td>${{d.name}}</td><td>${{d.team}}</td>
      <td class="mono">${{pct(d.p_drafted)}}</td><td class="mono">${{f(d.pred_pick_if_drafted,1)}}</td><td class="mono">${{f(d.expected_pick,1)}}</td>
      <td class="mono">${{f(d.points_per40,1)}}</td><td class="mono">${{f(d.assists_per40,1)}}</td><td class="mono">${{f(d.rebounds_total_per40,1)}}</td><td class="mono">${{pct(d.true_shooting_pct)}}</td></tr>`).join("");

    const W=980,H=620,M={{l:72,r:26,t:24,b:58}},PW=W-M.l-M.r,PH=H-M.t-M.b;
    if(!data.length){{ plot.innerHTML=`<text x="${{W/2}}" y="${{H/2}}" text-anchor="middle" fill="#5e6f71">No players for current filters</text>`; return; }}
    const yVals=data.map(d=>Number(d.expected_pick)||61), mVals=data.map(d=>Number(d.minutes)||0);
    const yLo=Math.min(1,...yVals), yHi=Math.max(61,...yVals), mLo=Math.min(...mVals), mHi=Math.max(...mVals);
    const labels=show.slice(0,Math.min(12,show.length));
    let html=`<rect x="0" y="0" width="${{W}}" height="${{H}}" fill="rgba(255,255,255,.72)" />`;
    for(let i=0;i<=5;i++){{ const x=M.l+(i/5)*PW,y=M.t+(i/5)*PH; html+=`<line x1="${{x}}" y1="${{M.t}}" x2="${{x}}" y2="${{M.t+PH}}" stroke="rgba(19,42,47,.06)" />`; html+=`<line x1="${{M.l}}" y1="${{y}}" x2="${{M.l+PW}}" y2="${{y}}" stroke="rgba(19,42,47,.06)" />`; }}
    data.forEach(d=>{{ const x=scale(Number(d.p_drafted)||0,0,1,M.l,M.l+PW), y=scale(Number(d.expected_pick)||61,yLo,yHi,M.t,M.t+PH), r=scale(Number(d.minutes)||0,mLo,mHi,3.5,10.5); html+=`<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{r.toFixed(2)}}" fill="${{clr(d.p_drafted)}}" stroke="rgba(19,42,47,.35)" stroke-width="0.9" data-name="${{d.name}}" data-team="${{d.team}}" data-prob="${{d.p_drafted}}" data-pickif="${{d.pred_pick_if_drafted}}" data-exp="${{d.expected_pick}}" />`; }});
    labels.forEach(d=>{{ const x=scale(Number(d.p_drafted)||0,0,1,M.l,M.l+PW), y=scale(Number(d.expected_pick)||61,yLo,yHi,M.t,M.t+PH); html+=`<text x="${{(x+8).toFixed(2)}}" y="${{(y-8).toFixed(2)}}" font-size="11" fill="#132a2f">${{d.name}}</text>`; }});
    html+=`<text x="${{W/2}}" y="${{H-14}}" text-anchor="middle" font-size="13" fill="#5e6f71">Drafted Probability</text>`;
    html+=`<text x="18" y="${{H/2}}" transform="rotate(-90,18,${{H/2}})" text-anchor="middle" font-size="13" fill="#5e6f71">Expected Pick (lower is better)</text>`;
    plot.innerHTML=html;
    plot.querySelectorAll("circle").forEach(c=>c.addEventListener("mouseenter",()=>{{ tip.textContent=`${{c.dataset.name}} · ${{c.dataset.team}} | p(drafted): ${{(100*Number(c.dataset.prob)).toFixed(1)}}% | pick_if: ${{Number(c.dataset.pickif).toFixed(1)}} | expected: ${{Number(c.dataset.exp).toFixed(1)}}`; }}));
  }}
  [q,conf,pos,prob,topn].forEach(el=>el.addEventListener("input",render));
  render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if not args.predictions_csv.exists():
        raise SystemExit(f"Missing predictions CSV: {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv, low_memory=False)
    for c in ["p_drafted", "pred_pick_if_drafted", "expected_pick", "minutes", "points_per40", "assists_per40", "rebounds_total_per40", "true_shooting_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["name", "team", "conference", "position_group", "p_drafted", "pred_pick_if_drafted", "expected_pick", "minutes", "points_per40", "assists_per40", "rebounds_total_per40", "true_shooting_pct"]
    keep = [c for c in keep if c in df.columns]
    records = df[keep].copy().to_dict(orient="records")

    report_text = args.report_txt.read_text(encoding="utf-8", errors="ignore") if args.report_txt.exists() else ""
    metrics = {
        "roc_auc": extract_metric(report_text, "ROC-AUC"),
        "pr_auc": extract_metric(report_text, "PR-AUC"),
        "brier": extract_metric(report_text, "Brier"),
        "mae": extract_metric(report_text, "MAE"),
        "rmse": extract_metric(report_text, "RMSE"),
        "actual_drafted": extract_metric(report_text, "Actual drafted in test year", int),
        "captured_top60": extract_metric(report_text, "Actual drafted captured in top-60 predicted probs", int),
    }

    html = build_html(
        records_json=json.dumps(records, ensure_ascii=True, separators=(",", ":")),
        metrics_json=json.dumps(metrics, ensure_ascii=True, separators=(",", ":")),
        top_n=args.top_n,
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
