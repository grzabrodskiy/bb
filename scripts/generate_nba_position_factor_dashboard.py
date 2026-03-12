from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate position-level factor dashboard for NBA success (RAPM proxy)."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nba_impact_targets_window2.csv"),
    )
    p.add_argument(
        "--out-html",
        type=Path,
        default=Path("data/processed/viz/nba_position_factor_dashboard.html"),
    )
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--min-rows", type=int, default=70)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument(
        "--min-abs-corr",
        type=float,
        default=0.05,
        help="Hide weak factors below this absolute correlation unless a section would be empty.",
    )
    p.add_argument(
        "--min-abs-coef",
        type=float,
        default=0.05,
        help="Show factors only when absolute standardized coefficient meets this floor.",
    )
    p.add_argument(
        "--min-feature-coverage",
        type=float,
        default=0.40,
        help="Minimum non-null share for core/improvement features in each position subset.",
    )
    p.add_argument(
        "--min-measurement-coverage",
        type=float,
        default=0.10,
        help="Minimum non-null share for measurement features in each position subset.",
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
    if s.startswith("C") or "C" == s:
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
        "stocks_per40": "Steals+Blocks per 40",
        "usage": "Usage",
        "true_shooting_pct": "True Shooting %",
        "three_point_pct": "3PT%",
        "three_point_attempt_rate": "3PA share",
        "assist_to_turnover": "Assist/Turnover",
        "turnovers_per40": "Turnovers per 40",
        "net_rating": "Net Rating",
        "offensive_rating": "Offensive Rating",
        "defensive_rating": "Defensive Rating",
        "measurement_height_in": "Height (in)",
        "measurement_weight_lb": "Weight (lb)",
        "measurement_wingspan_in": "Wingspan (in)",
        "measurement_standing_reach_in": "Standing Reach (in)",
        "measurement_wingspan_minus_height": "Wingspan - Height",
        "measurement_reach_minus_height": "Reach - Height",
        "combine_height_wo_shoes_in": "Combine Height (in)",
        "combine_weight_lb": "Combine Weight (lb)",
        "combine_wingspan_in": "Combine Wingspan (in)",
        "combine_standing_reach_in": "Combine Standing Reach (in)",
        "combine_wingspan_minus_height": "Combine Wingspan - Height",
        "crafted_height_in": "Crafted Height (in)",
        "crafted_wingspan_in": "Crafted Wingspan (in)",
        "crafted_length_in": "Crafted Length (in)",
        "crafted_wingspan_minus_height": "Crafted Wingspan - Height",
        "trend_points_per40": "YOY points growth",
        "trend_assists_per40": "YOY assists growth",
        "trend_rebounds_total_per40": "YOY rebounds growth",
        "trend_true_shooting_pct": "YOY TS% growth",
        "trend_three_point_pct": "YOY 3PT% growth",
        "trend_three_point_attempt_rate": "YOY 3PA share growth",
        "trend_net_rating": "YOY net rating growth",
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
        "peak_gap_points_per40": "Peak-over-average points gap",
        "peak_gap_assists_per40": "Peak-over-average assists gap",
        "peak_gap_rebounds_total_per40": "Peak-over-average rebounds gap",
        "peak_gap_true_shooting_pct": "Peak-over-average TS% gap",
        "peak_gap_three_point_pct": "Peak-over-average 3PT% gap",
        "peak_gap_three_point_attempt_rate": "Peak-over-average 3PA share gap",
        "peak_gap_net_rating": "Peak-over-average net rating gap",
    }
    return labels.get(col, col.replace("_", " "))


def _feature_bucket(col: str) -> str:
    if col.startswith("measurement_"):
        return "measurement"
    if col.startswith(("trend_", "peak_gap_")):
        return "improvement"
    return "core"


def _cap_first(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def _coverage_floor(n_rows: int, share: float) -> int:
    return max(20, int(round(float(n_rows) * float(share))))


def _build_html(payload_json: str, meta_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NBA Success by Position Factors</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
    :root {{
      --bg0:#f4efe4; --bg1:#e7ddcc; --ink:#10262a; --muted:#5c6e72;
      --card:rgba(255,253,247,0.84); --stroke:rgba(16,38,42,0.18);
      --plus:#0b7a75; --minus:#a25044; --line:#dde4e4;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); font-family:"Space Grotesk",sans-serif;
      background:radial-gradient(1200px 420px at 6% -10%, rgba(11,122,117,.12), transparent 58%),
                 radial-gradient(900px 380px at 94% -8%, rgba(162,80,68,.10), transparent 58%),
                 linear-gradient(165deg,var(--bg0) 0%,var(--bg1) 100%);
      min-height:100vh;
    }}
    .wrap {{ max-width:1260px; margin:0 auto; padding:24px; }}
    .hero,.panel,.meta-card {{
      border:1px solid var(--stroke); border-radius:16px; background:var(--card);
    }}
    .hero {{ padding:18px 20px; }}
    .hero h1 {{ margin:0; font-family:"Fraunces",serif; font-size:clamp(1.35rem,2.5vw,2rem); }}
    .hero p {{ margin:8px 0 0; color:var(--muted); max-width:980px; }}
    .top-nav {{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      margin-bottom:12px;
    }}
    .back {{
      display:inline-block; margin-bottom:12px; text-decoration:none; color:var(--ink);
      border:1px solid var(--stroke); border-radius:10px; padding:7px 11px; background:rgba(255,255,255,.76);
      font-size:.88rem;
    }}
    .meta {{
      display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:12px;
    }}
    .meta-card {{ padding:10px 11px; }}
    .meta-card .k {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.03em; }}
    .meta-card .v {{ margin-top:3px; font-weight:700; font-size:1.06rem; font-variant-numeric:tabular-nums; }}
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
      background:rgba(255,255,255,.78);
      color:var(--ink);
      font:inherit;
      font-size:.82rem;
      font-weight:700;
      padding:7px 10px;
      cursor:pointer;
    }}
    .mode-btn.active {{
      background:rgba(11,122,117,.16);
      border-color:rgba(11,122,117,.45);
    }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:12px; }}
    .panel {{ padding:12px; }}
    .panel h2 {{ margin:0; font-family:"Fraunces",serif; font-size:1.05rem; }}
    .sub {{
      margin-top:6px; margin-bottom:10px; color:var(--muted); font-size:.82rem;
      display:flex; gap:10px; flex-wrap:wrap;
    }}
    .section-title {{
      margin:10px 0 6px; font-size:.76rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em;
    }}
    .pn-grid {{
      display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px;
    }}
    .pn-col {{
      border:1px dashed rgba(16,38,42,.16);
      border-radius:10px;
      padding:8px;
      background:rgba(255,255,255,.58);
    }}
    .pn-col .section-title {{ margin-top:0; }}
    .factor-row {{
      display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center;
      border-bottom:1px dashed var(--line); padding:7px 0;
      border-left:3px solid transparent;
      padding-left:8px;
    }}
    .factor-row.bucket-core {{ border-left-color:#6b7280; }}
    .factor-row.bucket-measurement {{ border-left-color:#0b7a75; }}
    .factor-row.bucket-improvement {{ border-left-color:#c27d2a; }}
    .factor-row:last-child {{ border-bottom:0; }}
    .label {{ font-size:.81rem; font-weight:600; line-height:1.25; }}
    .meta-line {{ font-size:.74rem; color:var(--muted); margin-top:2px; }}
    .empty {{
      border:1px dashed rgba(16,38,42,.18); border-radius:10px; padding:8px 9px; color:var(--muted); font-size:.78rem;
      background:rgba(255,255,255,.6);
    }}
    .note {{ margin-top:10px; color:var(--muted); font-size:.8rem; }}
    @media (max-width:1120px) {{
      .meta {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .grid {{ grid-template-columns:1fr; }}
      .pn-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="top-nav">
      <a class="back" href="index.html">Back to Index</a>
      <a class="back" href="nba_position_feature_heatmap_dashboard.html?v=20260308d">Open Heatmap View</a>
    </div>
    <section class="hero">
      <h1>NBA Success Dashboard: Position Factor Explorer</h1>
      <p>
        This view uses one pooled multi-year model per position (guard, forward, center), so these are global drivers rather than year-specific.
        Positive factors are features where higher values push projected NBA impact up; negative factors push it down.
        To address scouting needs directly, the page also breaks out dedicated Measurement and Improvement (year-over-year / peak-gap) factors.
        Effect size is shown as standardized coefficient and simple correlation for context.
      </p>
      <div class="meta" id="meta"></div>
      <div class="modebar">
        <button type="button" id="modePartial" class="mode-btn active">Partial Impact (coef)</button>
        <button type="button" id="modeMarginal" class="mode-btn">Marginal Impact (corr)</button>
      </div>
    </section>
    <section class="grid" id="grid"></section>
  </main>
  <script>
  const DATA = {payload_json};
  const META = {meta_json};

  function fmt(v, d=2) {{
    const x = Number(v);
    return Number.isFinite(x) ? x.toFixed(d) : "n/a";
  }}
  function esc(s) {{
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }}

  document.getElementById("meta").innerHTML = [
    ["Model Rows", META.n_model_rows],
    ["Rows with Target", META.n_target_rows],
    ["Source Seasons", META.season_span],
    ["Min |coef| shown", META.min_abs_coef],
    ["Min |corr| shown", META.min_abs_corr],
  ].map(([k, v]) => `<div class="meta-card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");

  let impactMode = "partial";
  const modePartialBtn = document.getElementById("modePartial");
  const modeMarginalBtn = document.getElementById("modeMarginal");

  function metricKey() {{
    return impactMode === "partial" ? "coef" : "corr";
  }}

  function otherMetricKey() {{
    return impactMode === "partial" ? "corr" : "coef";
  }}

  function syncModeUi() {{
    modePartialBtn.classList.toggle("active", impactMode === "partial");
    modeMarginalBtn.classList.toggle("active", impactMode === "marginal");
  }}

  function renderFactors(items) {{
    if (!items.length) return '<div class="empty">Not enough stable signal in this section.</div>';
    const k = metricKey();
    const ok = otherMetricKey();
    const sorted = [...items].sort((a, b) => {{
      const av = Math.abs(Number(a?.[k]) || 0);
      const bv = Math.abs(Number(b?.[k]) || 0);
      if (av === bv) return String(a?.label || "").localeCompare(String(b?.label || ""));
      return bv - av;
    }});
    return sorted.map((x) => {{
      const b = String(x.bucket || "core");
      return `
        <div class="factor-row bucket-${{esc(b)}}">
          <div>
            <div class="label">${{esc(x.label)}}</div>
            <div class="meta-line">${{fmt(x[k],3)}} ${{k}} | ${{fmt(x[ok],2)}} ${{ok}}</div>
          </div>
        </div>`;
    }}).join("");
  }}

  function renderPosition(pos) {{
    const k = metricKey();
    const factors = Array.isArray(pos.factors) && pos.factors.length
      ? pos.factors
      : [...(pos.positive || []), ...(pos.negative || [])];
    const positive = factors.filter((x) => Number(x?.[k]) > 0);
    const negative = factors.filter((x) => Number(x?.[k]) < 0);
    return `
      <article class="panel">
        <h2>${{esc(pos.position)}}</h2>
        <div class="sub">
          <span>Rows: <strong>${{pos.n_rows}}</strong></span>
          <span>Mean target z: <strong>${{fmt(pos.mean_target_z,2)}}</strong></span>
          <span>Model fit corr: <strong>${{fmt(pos.fit_corr,2)}}</strong></span>
          <span>Mode: <strong>${{esc(impactMode === "partial" ? "partial" : "marginal")}}</strong></span>
        </div>
        <div class="pn-grid">
          <div class="pn-col">
            <h3 class="section-title">Positive Factors</h3>
            ${{renderFactors(positive)}}
          </div>
          <div class="pn-col">
            <h3 class="section-title">Negative Factors</h3>
            ${{renderFactors(negative)}}
          </div>
        </div>
        <div class="note">All feature types are pooled together here (performance, measurements, and development signals).</div>
      </article>`;
  }}

  function renderGrid() {{
    const grid = document.getElementById("grid");
    grid.innerHTML = DATA.map((pos) => renderPosition(pos)).join("");
  }}

  modePartialBtn.addEventListener("click", () => {{
    impactMode = "partial";
    syncModeUi();
    renderGrid();
  }});
  modeMarginalBtn.addEventListener("click", () => {{
    impactMode = "marginal";
    syncModeUi();
    renderGrid();
  }});

  syncModeUi();
  renderGrid();
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
    args.min_abs_corr = max(0.0, float(args.min_abs_corr))
    args.min_abs_coef = max(0.0, float(args.min_abs_coef))
    args.min_feature_coverage = min(1.0, max(0.0, float(args.min_feature_coverage)))
    args.min_measurement_coverage = min(1.0, max(0.0, float(args.min_measurement_coverage)))

    if "stocks_per40" not in df.columns:
        df["stocks_per40"] = _num(df.get("steals_per40")) + _num(df.get("blocks_per40"))
    if "assist_to_turnover" not in df.columns:
        df["assist_to_turnover"] = _num(df.get("assists_per40")) / _num(df.get("turnovers_per40")).replace(0, np.nan)

    # One measurement source policy: combine is primary, crafted is fallback.
    def _coalesce(cols: list[str]) -> pd.Series:
        out = pd.Series([float("nan")] * len(df), index=df.index, dtype=float)
        for c in cols:
            if c not in df.columns:
                continue
            x = _num(df[c])
            out = out.where(out.notna(), x)
        return out

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
    df["measurement_reach_minus_height"] = _coalesce(["measurement_reach_minus_height"])
    df["measurement_reach_minus_height"] = _num(df["measurement_reach_minus_height"]).where(
        _num(df["measurement_reach_minus_height"]).notna(),
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
        "measurement_wingspan_minus_height",
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
    # Pull all engineered trend/peak and profile variants from the table so
    # improvement drivers are not silently omitted when new columns are added.
    for pref in ("trend_", "peak_gap_", "career_avg_", "career_trim1_avg_", "best_"):
        for c in sorted([x for x in model.columns if x.startswith(pref)]):
            if c not in candidate_features:
                candidate_features.append(c)

    candidate_features = [c for c in candidate_features if c in model.columns]
    for c in candidate_features:
        model[c] = _num(model[c])

    out_rows: list[dict[str, object]] = []
    for pos in ["Guard", "Forward", "Center"]:
        sub = model[model["position_bucket"] == pos].copy().reset_index(drop=True)
        if len(sub) < args.min_rows:
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

        mdl = Ridge(alpha=args.alpha, random_state=42)
        Xz_np = Xz.to_numpy(dtype=float)
        mdl.fit(Xz_np, y_np)
        pred = mdl.predict(Xz_np)
        if len(pred) >= 2:
            fit_corr = float(np.corrcoef(pred, y_np)[0, 1])
        else:
            fit_corr = float("nan")
        coefs = pd.Series(mdl.coef_, index=use_features, dtype=float)
        corr_map: dict[str, float] = {}
        for c in use_features:
            x = _num(sub[c]).to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y_np)
            if int(m.sum()) >= 3:
                corr_map[c] = float(np.corrcoef(x[m], y_np[m])[0, 1])
            else:
                corr_map[c] = float("nan")
        corrs = pd.Series(corr_map, dtype=float).fillna(0.0)
        score = coefs.abs() * corrs.abs()

        tmp = pd.DataFrame(
            {
                "feature": use_features,
                "coef": coefs.values,
                "corr": corrs.reindex(use_features).values,
                "score": score.reindex(use_features).values,
            }
        )
        tmp["abs_coef"] = tmp["coef"].abs()
        tmp["bucket"] = tmp["feature"].map(_feature_bucket)
        coef_keep = tmp[tmp["abs_coef"] >= args.min_abs_coef].copy()
        if not coef_keep.empty:
            tmp = coef_keep
        corr_keep = tmp[tmp["corr"].abs() >= args.min_abs_corr].copy()
        if not corr_keep.empty:
            tmp = corr_keep
        pos_rows = (
            tmp[tmp["coef"] > 0]
            .sort_values(["abs_coef", "coef"], ascending=[False, False], kind="stable")
        )
        neg_rows = (
            tmp[tmp["coef"] < 0]
            .sort_values(["abs_coef", "coef"], ascending=[False, True], kind="stable")
        )
        all_rows = tmp.sort_values(["abs_coef", "coef"], ascending=[False, False], kind="stable")
        out_rows.append(
            {
                "position": pos,
                "n_rows": int(len(sub)),
                "mean_target_z": float(y.mean()),
                "fit_corr": fit_corr if pd.notna(fit_corr) else float("nan"),
                "factors": [
                    {
                        "feature": str(r["feature"]),
                        "label": _cap_first(_feature_label(str(r["feature"]))),
                        "bucket": str(r["bucket"]),
                        "coef": float(r["coef"]),
                        "corr": float(r["corr"]),
                        "score": float(r["score"]),
                    }
                    for _, r in all_rows.iterrows()
                ],
                "positive": [
                    {
                        "feature": str(r["feature"]),
                        "label": _cap_first(_feature_label(str(r["feature"]))),
                        "bucket": str(r["bucket"]),
                        "coef": float(r["coef"]),
                        "corr": float(r["corr"]),
                        "score": float(r["score"]),
                    }
                    for _, r in pos_rows.iterrows()
                ],
                "negative": [
                    {
                        "feature": str(r["feature"]),
                        "label": _cap_first(_feature_label(str(r["feature"]))),
                        "bucket": str(r["bucket"]),
                        "coef": float(r["coef"]),
                        "corr": float(r["corr"]),
                        "score": float(r["score"]),
                    }
                    for _, r in neg_rows.iterrows()
                ],
            }
        )

    if not out_rows:
        raise SystemExit("No position group had enough rows to build factors.")

    min_season = int(_num(model["season"]).dropna().min()) if _num(model["season"]).notna().any() else None
    max_season = int(_num(model["season"]).dropna().max()) if _num(model["season"]).notna().any() else None
    season_span = f"{min_season}-{max_season}" if min_season and max_season else "n/a"
    meta = {
        "n_model_rows": f"{len(df):,}",
        "n_target_rows": f"{len(model):,}",
        "season_span": season_span,
        "top_k": str(int(args.top_k)),
        "min_abs_coef": f"{100.0 * float(args.min_abs_coef):.1f}%",
        "min_abs_corr": f"{100.0 * float(args.min_abs_corr):.1f}%",
    }

    html = _build_html(
        payload_json=json.dumps(out_rows, ensure_ascii=True, separators=(",", ":")),
        meta_json=json.dumps(meta, ensure_ascii=True, separators=(",", ":")),
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
