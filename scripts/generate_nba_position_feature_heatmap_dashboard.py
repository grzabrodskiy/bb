from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate position-by-feature coefficient heatmap dashboard for NBA success."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nba_impact_targets_window2.csv"),
    )
    p.add_argument(
        "--out-html",
        type=Path,
        default=Path("data/processed/viz/nba_position_feature_heatmap_dashboard.html"),
    )
    p.add_argument("--min-rows", type=int, default=70)
    p.add_argument("--alpha", type=float, default=200.0)
    p.add_argument(
        "--min-feature-coverage",
        type=float,
        default=0.40,
        help="Minimum non-null share for absolute/trend features in each position subset.",
    )
    p.add_argument(
        "--min-measurement-coverage",
        type=float,
        default=0.10,
        help="Minimum non-null share for measurement features in each position subset.",
    )
    p.add_argument(
        "--top-highlight-count",
        type=int,
        default=12,
        help="Top-N features per position to mark as most significant.",
    )
    return p.parse_args()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _position_bucket(v: object) -> str:
    s = "" if v is None else str(v).strip().upper()
    if not s or s == "NAN":
        return "Other"
    if s.startswith("G") or " PG" in s or "SG" in s:
        return "Guard"
    if s.startswith("F") or "SF" in s or "PF" in s:
        return "Forward"
    if s.startswith("C") or s == "C":
        return "Center"
    if "G" in s and "F" in s:
        return "Forward"
    if "G" in s:
        return "Guard"
    if "F" in s:
        return "Forward"
    if "C" in s:
        return "Center"
    return "Other"


def _feature_label(col: str) -> str:
    labels = {
        "minutes": "Minutes",
        "points_per40": "Points per 40",
        "assists_per40": "Assists per 40",
        "rebounds_total_per40": "Rebounds per 40",
        "steals_per40": "Steals per 40",
        "blocks_per40": "Blocks per 40",
        "stocks_per40": "Steals + Blocks per 40",
        "usage": "Usage",
        "true_shooting_pct": "True Shooting %",
        "three_point_pct": "3PT%",
        "three_point_attempt_rate": "3PA share",
        "assist_to_turnover": "Assist / Turnover",
        "turnovers_per40": "Turnovers per 40",
        "net_rating": "Net Rating",
        "offensive_rating": "Offensive Rating",
        "defensive_rating": "Defensive Rating",
        "measurement_height_in": "Height (in)",
        "measurement_weight_lb": "Weight (lb)",
        "measurement_wingspan_in": "Wingspan (in)",
        "measurement_standing_reach_in": "Standing Reach (in)",
        "measurement_reach_minus_height": "Reach - Height",
        "career_trim1_avg_points_per40": "Multi-year points profile",
        "career_trim1_avg_assists_per40": "Multi-year assists profile",
        "career_trim1_avg_rebounds_total_per40": "Multi-year rebounds profile",
        "career_trim1_avg_true_shooting_pct": "Multi-year TS% profile",
        "career_trim1_avg_three_point_pct": "Multi-year 3PT% profile",
        "career_trim1_avg_three_point_attempt_rate": "Multi-year 3PA share profile",
        "career_trim1_avg_turnovers_per40": "Multi-year turnover profile",
        "career_trim1_avg_stocks_per40": "Multi-year stocks profile",
        "best_points_per40": "Best points season",
        "best_assists_per40": "Best assists season",
        "best_rebounds_total_per40": "Best rebounds season",
        "best_stocks_per40": "Best stocks season",
        "best_true_shooting_pct": "Best TS% season",
        "best_three_point_pct": "Best 3PT% season",
        "best_three_point_attempt_rate": "Best 3PA share season",
        "best_net_rating": "Best net rating season",
    }
    return labels.get(col, col.replace("_", " "))


def _feature_bucket(col: str) -> str:
    if col.startswith("measurement_"):
        return "measurement"
    if col.startswith(("trend_", "peak_gap_")):
        return "performance_trend"
    return "performance_absolute"


def _coverage_floor(n_rows: int, share: float) -> int:
    return max(20, int(round(float(n_rows) * float(share))))


def _cap_first(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def _build_html(payload_json: str, meta_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NBA Success Feature Heatmap by Position</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Archivo+Black&family=Manrope:wght@400;600;700&display=swap");
    :root {{
      --bg0:#f7f1e7; --bg1:#e8ddd0; --ink:#122327; --muted:#5b6b6f;
      --card:rgba(255,253,248,0.86); --stroke:rgba(18,35,39,0.16);
      --line:rgba(18,35,39,0.10); --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); font-family:"Manrope",sans-serif;
      background:
        radial-gradient(1200px 420px at 6% -8%, rgba(11, 122, 117, 0.13), transparent 60%),
        radial-gradient(900px 460px at 90% -10%, rgba(201, 128, 47, 0.12), transparent 58%),
        linear-gradient(165deg, var(--bg0) 0%, var(--bg1) 100%);
      min-height:100vh;
    }}
    .wrap {{ max-width:1320px; margin:0 auto; padding:22px 18px 34px; }}
    .top-nav {{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      margin-bottom:10px;
    }}
    .back {{
      display:inline-block; margin-bottom:10px; text-decoration:none; color:var(--ink);
      border:1px solid var(--stroke); border-radius:10px; padding:7px 11px;
      background:rgba(255,255,255,0.78); font-size:0.87rem; font-weight:600;
    }}
    .hero,.panel,.meta-card {{
      border:1px solid var(--stroke); border-radius:16px; background:var(--card);
    }}
    .hero {{ padding:16px; }}
    h1 {{
      margin:0; font-family:"Archivo Black",sans-serif;
      text-transform:uppercase; letter-spacing:0.02em;
      font-size:clamp(1.2rem,2.2vw,1.85rem); line-height:1.18;
    }}
    .hero p {{ margin:8px 0 0; color:var(--muted); max-width:1050px; }}
    .meta {{
      margin-top:11px; display:grid; gap:10px; grid-template-columns:repeat(5,minmax(0,1fr));
    }}
    .meta-card {{ padding:10px 11px; }}
    .meta-card .k {{
      color:var(--muted); text-transform:uppercase; letter-spacing:0.03em;
      font-size:0.7rem;
    }}
    .meta-card .v {{
      margin-top:3px; font-weight:700; font-size:1.04rem; font-variant-numeric:tabular-nums;
    }}
    .legend {{
      margin-top:11px; border:1px dashed var(--stroke); border-radius:12px;
      background:rgba(255,255,255,0.6); padding:9px 10px; color:var(--muted); font-size:0.82rem;
    }}
    .legend strong {{ color:var(--ink); }}
    .legend .swatch {{
      display:inline-block; width:14px; height:14px; border-radius:4px; vertical-align:middle; margin:0 4px;
      border:1px solid rgba(0,0,0,0.1);
    }}
    .modebar {{
      margin-top:10px;
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      align-items:center;
    }}
    .mode-btn {{
      border:1px solid var(--stroke);
      border-radius:10px;
      background:rgba(255,255,255,0.78);
      color:var(--ink);
      font:inherit;
      font-size:0.82rem;
      font-weight:700;
      padding:7px 10px;
      cursor:pointer;
    }}
    .mode-btn.active {{
      background:rgba(11,122,117,0.16);
      border-color:rgba(11,122,117,0.45);
    }}
    .panel {{ margin-top:12px; padding:12px; }}
    .cat-title {{
      margin:0 0 8px; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted);
    }}
    .table-wrap {{
      overflow:auto; border:1px solid var(--line); border-radius:12px; background:rgba(255,255,255,0.7);
    }}
    table {{ width:100%; border-collapse:separate; border-spacing:0; min-width:980px; }}
    thead th {{
      position:sticky; top:0; z-index:3; background:rgba(255,251,245,0.96);
      border-bottom:1px solid var(--line);
      font-size:0.76rem; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted);
      padding:9px 10px; text-align:left;
    }}
    thead th.pos {{ text-align:center; min-width:140px; }}
    th.sortable {{ cursor:pointer; user-select:none; }}
    .th-wrap {{
      display:inline-flex;
      align-items:center;
      gap:6px;
    }}
    .th-arr {{
      font-size:0.82rem;
      color:var(--muted);
    }}
    th.feat, td.feat {{
      position:sticky; left:0; z-index:2; background:rgba(255,251,245,0.98);
      border-right:1px solid var(--line);
    }}
    tbody td {{
      border-bottom:1px solid var(--line); border-right:1px solid var(--line);
      padding:8px 9px; vertical-align:top;
    }}
    tbody tr:last-child td {{ border-bottom:0; }}
    td.feat {{
      min-width:360px;
      border-right:1px solid var(--line);
    }}
    tr.cat-divider td {{
      background:rgba(248, 243, 234, 0.98);
      color:var(--muted);
      font-size:0.75rem;
      text-transform:uppercase;
      letter-spacing:0.04em;
      font-weight:700;
      padding:6px 9px;
    }}
    .feat-label {{ font-weight:700; font-size:0.84rem; line-height:1.2; }}
    .feat-key {{
      margin-top:2px; color:var(--muted); font-size:0.72rem; font-family:var(--mono);
      word-break:break-word;
    }}
    td.cell {{
      text-align:center; min-width:140px; font-variant-numeric:tabular-nums;
      border-right:1px solid rgba(18,35,39,0.13);
      box-shadow:inset 0 0 0 1px transparent;
    }}
    td.cell.top {{
      box-shadow:inset 0 0 0 2px rgba(18,35,39,0.33);
    }}
    td.cell .coef {{ font-weight:700; font-size:0.84rem; line-height:1.2; }}
    td.cell .aux {{ margin-top:2px; font-size:0.7rem; opacity:0.86; }}
    td.cell.empty {{
      background:rgba(255,255,255,0.72); color:var(--muted);
    }}
    .status {{
      margin-top:7px; color:var(--muted); font-size:0.78rem;
    }}
    @media (max-width:1160px) {{
      .meta {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      td.feat, th.feat {{ min-width:280px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="top-nav">
      <a class="back" href="index.html">Back to Index</a>
    </div>
    <section class="hero">
      <h1>NBA Success Dashboard: Feature Heatmap by Position</h1>
      <p>
        Rows are features grouped by category. Columns are position models (Guard, Forward, Center) trained on pooled historical data.
        Use Impact Mode to switch between partial impact (model coefficient) and marginal impact (feature correlation).
        Cell color follows sign (green positive, red negative), with intensity scaled to the active metric magnitude.
      </p>
      <div class="meta" id="meta"></div>
      <div class="legend">
        <span class="swatch" style="background:hsl(152 62% 76%);"></span><strong>Green:</strong> positive driver
        <span class="swatch" style="background:hsl(4 72% 76%);"></span><strong>Red:</strong> negative driver
        <span id="legendModeText" style="margin-left:8px;">Darker shade = larger |partial impact (coef)|. Outlined cells are top significant features per position.</span>
      </div>
      <div class="modebar">
        <button type="button" id="modePartial" class="mode-btn">Partial Impact (coef)</button>
        <button type="button" id="modeMarginal" class="mode-btn active">Marginal Impact (corr)</button>
      </div>
    </section>
    <div id="content"></div>
  </main>
  <script>
    const DATA = {payload_json};
    const META = {meta_json};

    function esc(s) {{
      return String(s ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}
    function fmt(v, d=3) {{
      const x = Number(v);
      return Number.isFinite(x) ? x.toFixed(d) : "n/a";
    }}
    function fmtSigned(v, d=3) {{
      const x = Number(v);
      if (!Number.isFinite(x)) return "n/a";
      return `${{x >= 0 ? "+" : ""}}${{x.toFixed(d)}}`;
    }}
    function fmtPct(v, d=1) {{
      const x = Number(v);
      return Number.isFinite(x) ? `${{(x * 100).toFixed(d)}}%` : "n/a";
    }}
    function fmtPctSigned(v, d=1) {{
      const x = Number(v);
      if (!Number.isFinite(x)) return "n/a";
      return `${{x >= 0 ? "+" : ""}}${{(x * 100).toFixed(d)}}%`;
    }}

    const metaRows = [
      ["Model rows", META.n_model_rows],
      ["Rows with target", META.n_target_rows],
      ["Source seasons", META.season_span],
      ["Ridge alpha", META.alpha],
      ["Top highlight", `Top ${{META.top_highlight_count}} / position`],
    ];
    document.getElementById("meta").innerHTML = metaRows
      .map(([k, v]) => `<div class="meta-card"><div class="k">${{esc(k)}}</div><div class="v">${{esc(v)}}</div></div>`)
      .join("");

    let sortKey = "feature";
    let sortDir = "asc";
    let impactMode = "marginal";

    const modePartialBtn = document.getElementById("modePartial");
    const modeMarginalBtn = document.getElementById("modeMarginal");
    const legendModeText = document.getElementById("legendModeText");

    function activeMetricKey() {{
      return impactMode === "partial" ? "coef" : "corr";
    }}

    function otherMetricKey() {{
      return impactMode === "partial" ? "corr" : "coef";
    }}

    function modeLabel() {{
      return impactMode === "partial" ? "partial impact (coef)" : "marginal impact (corr)";
    }}

    function syncModeUi() {{
      modePartialBtn.classList.toggle("active", impactMode === "partial");
      modeMarginalBtn.classList.toggle("active", impactMode === "marginal");
      if (legendModeText) {{
        legendModeText.textContent = `Darker shade = larger |${{modeLabel()}}|. Outlined cells are top significant features per position.`;
      }}
    }}

    function toneStyle(val) {{
      const mKey = activeMetricKey();
      const mRaw = Number(val?.[mKey]);
      if (!Number.isFinite(mRaw)) return "";
      const signPos = mRaw >= 0;
      const mShown = Number((Number(mRaw) || 0).toFixed(3));
      const pct = Math.abs(mShown) * 100.0;

      // Fixed intensity buckets requested by product:
      // <5%, 5-10%, 10-15%, 15-20%, 20%+
      let bucket = 0;
      if (pct >= 20.0) bucket = 4;
      else if (pct >= 15.0) bucket = 3;
      else if (pct >= 10.0) bucket = 2;
      else if (pct >= 5.0) bucket = 1;

      const hue = signPos ? 152 : 4;
      const sat = signPos ? 62 : 72;
      const lightByBucket = [97, 90, 82, 74, 64];
      const borderByBucket = [84, 72, 60, 48, 36];
      const light = lightByBucket[bucket];
      const borderLight = borderByBucket[bucket];
      const textColor = bucket >= 3 ? "#081218" : "#0f1f22";
      return `background:hsl(${{hue}} ${{sat}}% ${{light}}%);color:${{textColor}};border-color:hsl(${{hue}} ${{sat}}% ${{borderLight}}%);`;
    }}

    function renderCell(val) {{
      const mKey = activeMetricKey();
      const oKey = otherMetricKey();
      const mVal = Number(val?.[mKey]);
      const oVal = Number(val?.[oKey]);
      if (!Number.isFinite(mVal)) {{
        return '<td class="cell empty"><div class="coef">-</div></td>';
      }}
      const topClass = Number(val.rank || 999999) <= Number(META.top_highlight_count || 0) ? "top" : "";
      const title = `coef=${{fmtPctSigned(val.coef, 2)}}, corr=${{fmtPctSigned(val.corr, 2)}}, sig=${{fmt(val.sig, 4)}}, rank=${{Number(val.rank || 0)}}`;
      return `<td class="cell ${{topClass}}" style="${{toneStyle(val)}}" title="${{esc(title)}}">
        <div class="coef">${{fmtPctSigned(mVal, 1)}}</div>
        <div class="aux">${{fmtPctSigned(oVal, 1)}} ${{oKey}}</div>
      </td>`;
    }}

    function normalizeCategoryKey(v) {{
      const key = String(v || "");
      return DATA.category_labels[key] || key;
    }}

    function headerArrow(key) {{
      if (sortKey !== key) return "↕";
      return sortDir === "asc" ? "↑" : "↓";
    }}

    function rowCmpByLabel(a, b, dir) {{
      const an = String(a.label || a.feature || "").toLowerCase();
      const bn = String(b.label || b.feature || "").toLowerCase();
      if (an < bn) return -1 * dir;
      if (an > bn) return 1 * dir;
      return String(a.feature || "").localeCompare(String(b.feature || "")) * dir;
    }}

    function rowCmpByPosition(a, b, pos, dir) {{
      const mKey = activeMetricKey();
      const av = Number((a.values || {{}})[pos]?.[mKey]);
      const bv = Number((b.values || {{}})[pos]?.[mKey]);
      const aOk = Number.isFinite(av);
      const bOk = Number.isFinite(bv);
      if (!aOk && !bOk) return rowCmpByLabel(a, b, 1);
      if (!aOk) return 1;
      if (!bOk) return -1;
      if (av === bv) return rowCmpByLabel(a, b, 1);
      return (av < bv ? -1 : 1) * dir;
    }}

    function renderDataRow(r) {{
        const feat = `<td class="feat"><div class="feat-label">${{esc(r.label)}}</div><div class="feat-key">${{esc(r.feature)}}</div></td>`;
        const cells = DATA.positions.map((p) => renderCell((r.values || {{}})[p])).join("");
        return `<tr>${{feat}}${{cells}}</tr>`;
    }}

    function renderTable() {{
      const dir = sortDir === "asc" ? 1 : -1;
      const allRows = [...DATA.rows];
      let body = "";
      if (sortKey === "feature") {{
        DATA.category_order.forEach((cat) => {{
          const part = allRows.filter((r) => String(r.category) === String(cat));
          if (!part.length) return;
          part.sort((a, b) => rowCmpByLabel(a, b, dir));
          body += `<tr class="cat-divider"><td colspan="${{1 + DATA.positions.length}}">${{esc(normalizeCategoryKey(cat))}}</td></tr>`;
          body += part.map((r) => renderDataRow(r)).join("");
        }});
      }} else {{
        allRows.sort((a, b) => rowCmpByPosition(a, b, sortKey, dir));
        body = allRows.map((r) => renderDataRow(r)).join("");
      }}

      const featureHeader = `<th class="feat sortable" data-sort="feature"><span class="th-wrap">Feature <span class="th-arr">${{headerArrow("feature")}}</span></span></th>`;
      const positionHeader = DATA.positions
        .map((p) => `<th class="pos sortable" data-sort="${{esc(p)}}"><span class="th-wrap">${{esc(p)}} <span class="th-arr">${{headerArrow(p)}}</span></span></th>`)
        .join("");
      document.getElementById("content").innerHTML = `<section class="panel">
        <h2 class="cat-title">Sortable Feature Heatmap</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                ${{featureHeader}}
                ${{positionHeader}}
              </tr>
            </thead>
            <tbody>${{body}}</tbody>
          </table>
        </div>
        <div class="status">${{DATA.rows.length}} features shown. Sort key: ${{esc(sortKey)}} (${{esc(sortDir)}}), mode: ${{esc(modeLabel())}}.</div>
      </section>`;
    }}

    document.addEventListener("click", (ev) => {{
      const th = ev.target.closest("th.sortable");
      if (!th) return;
      const key = String(th.dataset.sort || "");
      if (!key) return;
      if (sortKey === key) {{
        sortDir = sortDir === "asc" ? "desc" : "asc";
      }} else {{
        sortKey = key;
        sortDir = key === "feature" ? "asc" : "desc";
      }}
      renderTable();
    }});

    modePartialBtn.addEventListener("click", () => {{
      impactMode = "partial";
      syncModeUi();
      renderTable();
    }});
    modeMarginalBtn.addEventListener("click", () => {{
      impactMode = "marginal";
      syncModeUi();
      renderTable();
    }});

    syncModeUi();
    renderTable();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    if "nba_impact_target_z" not in df.columns:
        raise SystemExit("Input is missing nba_impact_target_z.")

    args.min_feature_coverage = min(1.0, max(0.0, float(args.min_feature_coverage)))
    args.min_measurement_coverage = min(1.0, max(0.0, float(args.min_measurement_coverage)))
    args.top_highlight_count = max(1, int(args.top_highlight_count))

    if "stocks_per40" not in df.columns:
        df["stocks_per40"] = _num(df.get("steals_per40")) + _num(df.get("blocks_per40"))
    if "assist_to_turnover" not in df.columns:
        df["assist_to_turnover"] = _num(df.get("assists_per40")) / _num(df.get("turnovers_per40")).replace(0, np.nan)

    def _coalesce(cols: list[str]) -> pd.Series:
        out = pd.Series([float("nan")] * len(df), index=df.index, dtype=float)
        for c in cols:
            if c not in df.columns:
                continue
            x = _num(df[c])
            out = out.where(out.notna(), x)
        return out

    # One source policy: combine first, crafted or legacy fallbacks.
    df["measurement_height_in"] = _coalesce(
        ["combine_height_wo_shoes_in", "measurement_height_in", "crafted_height_in", "height_in", "ext_height_in_modeldb"]
    )
    df["measurement_weight_lb"] = _coalesce(["combine_weight_lb", "measurement_weight_lb", "weight_lb", "ext_weight_lb_modeldb"])
    df["measurement_wingspan_in"] = _coalesce(["combine_wingspan_in", "measurement_wingspan_in", "crafted_wingspan_in"])
    df["measurement_standing_reach_in"] = _coalesce(["combine_standing_reach_in", "measurement_standing_reach_in"])
    df["measurement_wingspan_minus_height"] = _coalesce(
        ["combine_wingspan_minus_height", "measurement_wingspan_minus_height", "crafted_wingspan_minus_height"]
    )
    df["measurement_wingspan_minus_height"] = _num(df["measurement_wingspan_minus_height"]).where(
        _num(df["measurement_wingspan_minus_height"]).notna(),
        _num(df["measurement_wingspan_in"]) - _num(df["measurement_height_in"]),
    )
    reach_minus = _num(
        df.get("measurement_reach_minus_height", pd.Series([float("nan")] * len(df), index=df.index))
    )
    df["measurement_reach_minus_height"] = reach_minus.where(
        reach_minus.notna(),
        _num(df["measurement_standing_reach_in"]) - _num(df["measurement_height_in"]),
    )

    if "position_group" in df.columns:
        pos_source = df["position_group"]
    elif "position" in df.columns:
        pos_source = df["position"]
    else:
        pos_source = pd.Series(["Other"] * len(df), index=df.index)
    df["position_bucket"] = pos_source.map(_position_bucket)

    df["season"] = _num(df.get("season"))
    df["nba_impact_target_z"] = _num(df["nba_impact_target_z"])
    model = df[df["nba_impact_target_z"].notna()].copy()
    if model.empty:
        raise SystemExit("No rows with known target in input CSV.")

    candidate_features = [
        "minutes",
        "points_per40",
        "assists_per40",
        "rebounds_total_per40",
        "steals_per40",
        "blocks_per40",
        "stocks_per40",
        "usage",
        "true_shooting_pct",
        "three_point_pct",
        "three_point_attempt_rate",
        "assist_to_turnover",
        "turnovers_per40",
        "net_rating",
        "offensive_rating",
        "defensive_rating",
        "measurement_height_in",
        "measurement_weight_lb",
        "measurement_wingspan_in",
        "measurement_standing_reach_in",
        "measurement_reach_minus_height",
        "career_trim1_avg_points_per40",
        "career_trim1_avg_assists_per40",
        "career_trim1_avg_rebounds_total_per40",
        "career_trim1_avg_true_shooting_pct",
        "career_trim1_avg_three_point_pct",
        "career_trim1_avg_three_point_attempt_rate",
        "career_trim1_avg_turnovers_per40",
        "career_trim1_avg_stocks_per40",
        "best_points_per40",
        "best_assists_per40",
        "best_rebounds_total_per40",
        "best_stocks_per40",
        "best_true_shooting_pct",
        "best_three_point_pct",
        "best_three_point_attempt_rate",
        "best_net_rating",
    ]
    for pref in ("career_avg_", "career_trim1_avg_", "best_"):
        for c in sorted([x for x in model.columns if x.startswith(pref)]):
            if c not in candidate_features:
                candidate_features.append(c)

    candidate_features = [c for c in candidate_features if c in model.columns]
    for c in candidate_features:
        model[c] = _num(model[c])

    positions = ["Guard", "Forward", "Center"]
    by_position: dict[str, dict[str, object]] = {
        p: {"n_rows": int(len(model[model["position_bucket"] == p])), "fit_corr": float("nan"), "features": {}}
        for p in positions
    }

    for pos in positions:
        sub = model[model["position_bucket"] == pos].copy().reset_index(drop=True)
        if len(sub) < int(args.min_rows):
            continue
        y = _num(sub["nba_impact_target_z"]).reset_index(drop=True)
        y_np = y.to_numpy(dtype=float)

        use_features: list[str] = []
        for c in candidate_features:
            x = _num(sub[c])
            bucket = _feature_bucket(c)
            min_non_na = _coverage_floor(
                len(sub),
                args.min_measurement_coverage if bucket == "measurement" else args.min_feature_coverage,
            )
            if int(x.notna().sum()) < min_non_na:
                continue
            if float(x.std(ddof=0)) <= 1e-9:
                continue
            use_features.append(c)
        if len(use_features) < 6:
            continue

        X = sub[use_features].copy()
        med = X.median(axis=0, skipna=True)
        X = X.fillna(med)
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0).replace(0, 1.0)
        Xz = (X - mu) / sd

        mdl = Ridge(alpha=float(args.alpha), random_state=42)
        Xz_np = Xz.to_numpy(dtype=float)
        mdl.fit(Xz_np, y_np)
        pred = mdl.predict(Xz_np)
        if len(pred) >= 2:
            by_position[pos]["fit_corr"] = float(np.corrcoef(pred, y_np)[0, 1])

        coefs = pd.Series(mdl.coef_, index=use_features, dtype=float)
        for c in use_features:
            x = _num(sub[c]).to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y_np)
            corr = float(np.corrcoef(x[m], y_np[m])[0, 1]) if int(m.sum()) >= 3 else float("nan")
            corr_abs = abs(corr) if pd.notna(corr) else 0.0
            sig = float(abs(coefs[c]) * corr_abs)
            by_position[pos]["features"][c] = {
                "coef": float(coefs[c]),
                "corr": float(corr) if pd.notna(corr) else 0.0,
                "sig": sig,
                "coverage": float(_num(sub[c]).notna().mean()),
            }

    # Rank feature significance within each position model.
    rank_map: dict[str, dict[str, int]] = {}
    for pos in positions:
        feats = by_position[pos]["features"]
        ordered = sorted(feats.items(), key=lambda kv: kv[1]["sig"], reverse=True)
        rank_map[pos] = {feat: i + 1 for i, (feat, _) in enumerate(ordered)}

    all_features = sorted({f for pos in positions for f in by_position[pos]["features"].keys()})
    if not all_features:
        raise SystemExit("No usable position features found. Try lowering coverage thresholds.")

    cat_order = ["measurement", "performance_absolute", "performance_trend"]
    cat_labels = {
        "measurement": "Measurement",
        "performance_absolute": "Performance (absolute)",
        "performance_trend": "Performance (trend)",
    }

    rows: list[dict[str, object]] = []
    max_sig = 0.0
    max_abs_coef = 0.0
    max_abs_corr = 0.0
    for feat in all_features:
        values: dict[str, dict[str, float] | None] = {}
        row_max_sig = 0.0
        for pos in positions:
            m = by_position[pos]["features"].get(feat)
            if m is None:
                values[pos] = None
                continue
            row_max_sig = max(row_max_sig, float(m["sig"]))
            max_sig = max(max_sig, float(m["sig"]))
            max_abs_coef = max(max_abs_coef, round(abs(float(m["coef"])), 3))
            max_abs_corr = max(max_abs_corr, round(abs(float(m["corr"])), 3))
            values[pos] = {
                "coef": float(m["coef"]),
                "corr": float(m["corr"]),
                "sig": float(m["sig"]),
                "coverage": float(m["coverage"]),
                "rank": int(rank_map.get(pos, {}).get(feat, 999999)),
            }

        rows.append(
            {
                "feature": feat,
                "label": _cap_first(_feature_label(feat)),
                "category": _feature_bucket(feat),
                "max_sig": float(row_max_sig),
                "values": values,
            }
        )

    cat_rank = {c: i for i, c in enumerate(cat_order)}
    rows.sort(
        key=lambda r: (
            cat_rank.get(str(r["category"]), 999),
            -float(r.get("max_sig", 0.0)),
            str(r.get("label", "")),
        )
    )

    min_season = int(_num(model["season"]).dropna().min()) if _num(model["season"]).notna().any() else None
    max_season = int(_num(model["season"]).dropna().max()) if _num(model["season"]).notna().any() else None
    season_span = f"{min_season}-{max_season}" if min_season and max_season else "n/a"

    payload = {
        "positions": positions,
        "category_order": cat_order,
        "category_labels": cat_labels,
        "rows": rows,
    }
    meta = {
        "n_model_rows": f"{len(df):,}",
        "n_target_rows": f"{len(model):,}",
        "season_span": season_span,
        "alpha": f"{float(args.alpha):.2f}",
        "top_highlight_count": int(args.top_highlight_count),
        "max_sig": float(max_sig),
        "max_abs_coef": float(max_abs_coef),
        "max_abs_corr": float(max_abs_corr),
    }

    html = _build_html(
        payload_json=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        meta_json=json.dumps(meta, ensure_ascii=True, separators=(",", ":")),
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
