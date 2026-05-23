#!/usr/bin/env python3
"""Render a static HTML report for the final dns3 clustering result.

The report intentionally contains only the two views requested for manual
inspection: a 2D scatter plot and a cluster feature profile table. It reads
the selected candidate metadata from final_selection.json and the labels from
submission/answers/dns3.csv, then projects the traffic-only feature matrix
into two dimensions with TruncatedSVD.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_score_experiments as experiments
import solve_dns_homework as solver


REPORT_PATH = solver.REPORTS_DIR / "dns3_cluster_visualization.html"
LABEL_PATH = solver.PROJECT_ROOT / "submission" / "answers" / "dns3.csv"
FINAL_SELECTION_PATH = (
    solver.PROJECT_ROOT
    / ".sisyphus"
    / "evidence"
    / "score_improvement"
    / "dns3_interpretable"
    / "final_selection.json"
)
SVG_WIDTH = 1180
SVG_HEIGHT = 760
PLOT_MARGIN_X = 78
PLOT_MARGIN_Y = 64


PALETTE = [
    "#d95f02",
    "#1b9e77",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#1f78b4",
    "#b2df8a",
    "#fb9a99",
    "#6a3d9a",
    "#ff7f00",
    "#b15928",
    "#8dd3c7",
    "#bc80bd",
    "#bebada",
]


@dataclass(frozen=True)
class ClusterProfile:
    label: int
    count: int
    share: float
    x_mean: float
    y_mean: float
    access_signature: list[tuple[str, float]]
    flint_signature: list[tuple[str, float]]
    overall_signature: list[tuple[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render static dns3 cluster visualization HTML.")
    parser.add_argument(
        "--output",
        default=str(REPORT_PATH),
        help=f"HTML output path. Default: {REPORT_PATH}",
    )
    parser.add_argument(
        "--labels",
        default=str(LABEL_PATH),
        help=f"dns3 label CSV path. Default: {LABEL_PATH}",
    )
    parser.add_argument(
        "--context-note",
        default=(
            "route/algorithm metadata is sourced from the source final_selection.json "
            "and source feature pipeline."
        ),
        help="Plain-language note explaining where route/algorithm visualization context comes from.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_to_2d(matrix: solver.NDArray) -> solver.NDArray:
    reducer = solver.TruncatedSVD(n_components=2, random_state=solver.RANDOM_STATE)
    projected = reducer.fit_transform(matrix)
    return solver.StandardScaler().fit_transform(projected)


def scale(values: solver.NDArray | list[float], *, low: float, high: float, invert: bool = False) -> list[float]:
    np = solver.np
    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or abs(float(hi - lo)) < 1e-12:
        midpoint = (low + high) / 2.0
        return [midpoint for _ in arr]
    clipped = np.clip(arr, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    if invert:
        norm = 1.0 - norm
    return [float(low + item * (high - low)) for item in norm]


def feature_signature(
    means: solver.NDArray,
    feature_names: list[str],
    prefixes: tuple[str, ...],
    *,
    limit: int = 5,
) -> list[tuple[str, float]]:
    np = solver.np
    indices = [idx for idx, name in enumerate(feature_names) if name.startswith(prefixes)]
    ranked = sorted(indices, key=lambda idx: float(means[idx]), reverse=True)
    return [(feature_names[idx], float(means[idx])) for idx in ranked[:limit] if float(means[idx]) > 0.05]


def overall_signature(means: solver.NDArray, feature_names: list[str], *, limit: int = 5) -> list[tuple[str, float]]:
    ranked = sorted(range(len(feature_names)), key=lambda idx: float(means[idx]), reverse=True)
    return [(feature_names[idx], float(means[idx])) for idx in ranked[:limit] if float(means[idx]) > 0.05]


def format_signature(items: list[tuple[str, float]]) -> str:
    if not items:
        return "<span class=\"muted\">无明显高于整体均值的特征</span>"
    chips = []
    for name, value in items:
        chips.append(
            "<span class=\"feature-chip\">"
            f"{html.escape(name)} <strong>+{value:.2f}σ</strong>"
            "</span>"
        )
    return "".join(chips)


def build_profiles(
    labels: solver.NDArray,
    matrix: solver.NDArray,
    embedding: solver.NDArray,
    feature_names: list[str],
) -> list[ClusterProfile]:
    np = solver.np
    pd = solver.pd
    labels_arr = np.asarray(labels, dtype=int)
    profiles: list[ClusterProfile] = []
    total = len(labels_arr)
    for label in sorted(pd.unique(labels_arr)):
        mask = labels_arr == int(label)
        cluster_matrix = matrix[mask]
        means = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        cluster_embedding = embedding[mask]
        profiles.append(
            ClusterProfile(
                label=int(label),
                count=int(mask.sum()),
                share=float(mask.sum() / total),
                x_mean=float(cluster_embedding[:, 0].mean()),
                y_mean=float(cluster_embedding[:, 1].mean()),
                access_signature=feature_signature(means, feature_names, ("dns3_access",)),
                flint_signature=feature_signature(means, feature_names, ("dns3_flint",)),
                overall_signature=overall_signature(means, feature_names),
            )
        )
    return profiles


def render_svg(labels: solver.NDArray, embedding: solver.NDArray, profiles: list[ClusterProfile], meta: dict[str, Any]) -> str:
    np = solver.np
    labels_arr = np.asarray(labels, dtype=int)
    x_values = np.asarray(embedding[:, 0], dtype=float)
    y_values = np.asarray(embedding[:, 1], dtype=float)
    xs = scale(x_values, low=PLOT_MARGIN_X, high=SVG_WIDTH - PLOT_MARGIN_X)
    ys = scale(y_values, low=PLOT_MARGIN_Y, high=SVG_HEIGHT - PLOT_MARGIN_Y, invert=True)

    circles = []
    for idx, label in enumerate(labels_arr):
        color = PALETTE[int(label) % len(PALETTE)]
        circles.append(
            f'<circle cx="{xs[idx]:.2f}" cy="{ys[idx]:.2f}" r="2.05" fill="{color}" '
            f'fill-opacity="0.48"><title>fqdn index {idx}; cluster {int(label)}</title></circle>'
        )

    label_nodes = []
    legend_nodes = []
    profile_by_label = {profile.label: profile for profile in profiles}
    cluster_center_x = scale(
        [profile.x_mean for profile in profiles],
        low=PLOT_MARGIN_X,
        high=SVG_WIDTH - PLOT_MARGIN_X,
    )
    cluster_center_y = scale(
        [profile.y_mean for profile in profiles],
        low=PLOT_MARGIN_Y,
        high=SVG_HEIGHT - PLOT_MARGIN_Y,
        invert=True,
    )
    for pos, profile in enumerate(profiles):
        color = PALETTE[profile.label % len(PALETTE)]
        label_nodes.append(
            f'<g class="cluster-label"><circle cx="{cluster_center_x[pos]:.2f}" cy="{cluster_center_y[pos]:.2f}" '
            f'r="15" fill="{color}" fill-opacity="0.88" />'
            f'<text x="{cluster_center_x[pos]:.2f}" y="{cluster_center_y[pos] + 5:.2f}" text-anchor="middle">{profile.label}</text></g>'
        )
        legend_y = 30 + pos * 24
        legend_nodes.append(
            f'<g class="legend-item" transform="translate(956 {legend_y})">'
            f'<rect width="13" height="13" rx="3" fill="{color}" />'
            f'<text x="21" y="11">簇 {profile.label}: {profile.count:,} ({profile.share:.1%})</text></g>'
        )

    largest = max(profiles, key=lambda item: item.count)
    smallest = min(profiles, key=lambda item: item.count)
    return f"""
<section class="panel scatter-panel" aria-labelledby="scatter-title">
  <div class="section-kicker">2D scatter · TruncatedSVD projection</div>
  <h2 id="scatter-title">二维散点图</h2>
  <p class="section-note">每个点代表一个 <code>fqdn_no</code>；颜色来自标签文件 <code>{html.escape(meta['labels_path'])}</code> 中的聚类标签（SHA256: <code>{html.escape(meta['labels_sha256'])}</code>）。二维坐标由 source feature pipeline 的 {meta['selected_feature_count']} 维 <code>{html.escape(meta['route'])}</code> 特征经 TruncatedSVD 投影得到，仅用于直观观察；{html.escape(meta['context_note'])}</p>
  <svg class="scatter" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-label="dns3 final clustering two-dimensional scatter plot">
    <defs>
      <linearGradient id="paperGlow" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0%" stop-color="#fff7e1"/>
        <stop offset="100%" stop-color="#eef7ff"/>
      </linearGradient>
      <pattern id="grid" width="42" height="42" patternUnits="userSpaceOnUse">
        <path d="M 42 0 L 0 0 0 42" fill="none" stroke="#17324d" stroke-opacity="0.075" stroke-width="1"/>
      </pattern>
    </defs>
    <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" rx="28" fill="url(#paperGlow)" />
    <rect x="24" y="24" width="{SVG_WIDTH - 48}" height="{SVG_HEIGHT - 48}" rx="20" fill="url(#grid)" />
    <line x1="{PLOT_MARGIN_X}" y1="{SVG_HEIGHT - PLOT_MARGIN_Y}" x2="{SVG_WIDTH - PLOT_MARGIN_X}" y2="{SVG_HEIGHT - PLOT_MARGIN_Y}" class="axis" />
    <line x1="{PLOT_MARGIN_X}" y1="{PLOT_MARGIN_Y}" x2="{PLOT_MARGIN_X}" y2="{SVG_HEIGHT - PLOT_MARGIN_Y}" class="axis" />
    <text x="{SVG_WIDTH / 2:.0f}" y="{SVG_HEIGHT - 22}" class="axis-label" text-anchor="middle">SVD-1</text>
    <text x="28" y="{SVG_HEIGHT / 2:.0f}" class="axis-label" transform="rotate(-90 28 {SVG_HEIGHT / 2:.0f})" text-anchor="middle">SVD-2</text>
    <g class="points">{''.join(circles)}</g>
    <g class="cluster-centers">{''.join(label_nodes)}</g>
    <g class="legend">{''.join(legend_nodes)}</g>
  </svg>
  <div class="callout-row">
    <div><strong>最大簇</strong><span>簇 {profile_by_label[largest.label].label} · {largest.count:,} 个域名</span></div>
    <div><strong>最小簇</strong><span>簇 {profile_by_label[smallest.label].label} · {smallest.count:,} 个域名</span></div>
    <div><strong>总样本</strong><span>{len(labels_arr):,} 个域名 · {len(profiles)} 个簇</span></div>
  </div>
</section>
"""


def render_profile_table(profiles: list[ClusterProfile]) -> str:
    rows = []
    for profile in profiles:
        color = PALETTE[profile.label % len(PALETTE)]
        rows.append(
            "<tr>"
            f"<td><span class=\"swatch\" style=\"--swatch:{color}\"></span><strong>簇 {profile.label}</strong></td>"
            f"<td class=\"num\">{profile.count:,}</td>"
            f"<td class=\"num\">{profile.share:.2%}</td>"
            f"<td>{format_signature(profile.access_signature)}</td>"
            f"<td>{format_signature(profile.flint_signature)}</td>"
            f"<td>{format_signature(profile.overall_signature)}</td>"
            "</tr>"
        )
    return f"""
<section class="panel profile-panel" aria-labelledby="profile-title">
  <div class="section-kicker">cluster feature profile · standardized mean lift</div>
  <h2 id="profile-title">簇特征画像表</h2>
  <p class="section-note">表中 <strong>+xσ</strong> 表示该簇在对应标准化特征上的均值比全体均值高多少个标准差；仅展示每簇最突出的访问/解析特征。</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>簇</th>
          <th>数量</th>
          <th>占比</th>
          <th>访问行为画像</th>
          <th>解析行为画像</th>
          <th>整体最突出特征</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</section>
"""


def render_html(scatter_svg: str, profile_table: str, meta: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>dns3 聚类结果静态展示</title>
  <link rel="icon" href="data:," />
  <style>
    :root {{
      --ink: #18212d;
      --muted: #607087;
      --paper: #fbf6ea;
      --panel: rgba(255, 255, 255, 0.82);
      --line: rgba(24, 33, 45, 0.16);
      --shadow: 0 24px 70px rgba(26, 39, 61, 0.16);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 10%, rgba(245, 160, 72, 0.22), transparent 32rem),
        radial-gradient(circle at 90% 2%, rgba(69, 140, 202, 0.20), transparent 30rem),
        linear-gradient(135deg, #f5ecd8 0%, #f9fbff 48%, #edf4ed 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1280px, calc(100vw - 48px));
      margin: 42px auto 64px;
    }}
    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 28px;
      margin-bottom: 24px;
      padding: 0 6px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2.4rem, 5vw, 5.4rem);
      line-height: 0.92;
      letter-spacing: -0.07em;
      font-weight: 900;
    }}
    .subtitle {{
      max-width: 460px;
      color: var(--muted);
      font-size: 1.02rem;
      text-align: right;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 32px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
      padding: 28px;
      margin-top: 22px;
    }}
    h2 {{
      margin: 4px 0 8px;
      font-size: clamp(1.7rem, 3vw, 2.7rem);
      letter-spacing: -0.045em;
      line-height: 1;
    }}
    .section-kicker {{
      color: #a25c0f;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font: 700 0.76rem "Avenir Next", "Trebuchet MS", sans-serif;
    }}
    .section-note {{
      color: var(--muted);
      max-width: 980px;
      margin: 0 0 20px;
      font-size: 0.98rem;
    }}
    code {{
      padding: 0.08rem 0.34rem;
      border-radius: 8px;
      background: rgba(24, 33, 45, 0.08);
      font-family: "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.9em;
    }}
    .scatter {{ width: 100%; height: auto; display: block; }}
    .axis {{ stroke: rgba(24, 33, 45, 0.38); stroke-width: 1.5; }}
    .axis-label {{ fill: rgba(24, 33, 45, 0.58); font: 700 15px "Avenir Next", "Trebuchet MS", sans-serif; }}
    .legend text {{ fill: #243041; font: 700 13px "Avenir Next", "Trebuchet MS", sans-serif; }}
    .cluster-label text {{ fill: #fff; font: 900 15px "Avenir Next", "Trebuchet MS", sans-serif; pointer-events: none; }}
    .callout-row {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .callout-row div {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(255, 255, 255, 0.62);
    }}
    .callout-row strong {{ display: block; font-size: 0.8rem; color: #a25c0f; text-transform: uppercase; letter-spacing: 0.08em; }}
    .callout-row span {{ display: block; margin-top: 4px; font-size: 1.1rem; font-weight: 800; }}
    .table-wrap {{ overflow-x: auto; border-radius: 20px; border: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; background: rgba(255, 255, 255, 0.74); min-width: 980px; }}
    th, td {{ padding: 14px 16px; border-bottom: 1px solid rgba(24, 33, 45, 0.11); vertical-align: top; }}
    th {{
      text-align: left;
      color: #263449;
      background: rgba(251, 239, 211, 0.78);
      font: 800 0.82rem "Avenir Next", "Trebuchet MS", sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      white-space: nowrap;
    }}
    tbody tr:hover {{ background: rgba(255, 248, 231, 0.78); }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 800; }}
    .swatch {{ display: inline-block; width: 13px; height: 13px; border-radius: 4px; background: var(--swatch); margin-right: 8px; vertical-align: -1px; }}
    .feature-chip {{
      display: inline-block;
      margin: 0 7px 7px 0;
      padding: 5px 8px;
      border-radius: 999px;
      background: rgba(24, 33, 45, 0.075);
      font: 600 0.78rem "Avenir Next", "Trebuchet MS", sans-serif;
      white-space: nowrap;
    }}
    .feature-chip strong {{ color: #9b4f00; }}
    .muted {{ color: var(--muted); font-style: italic; }}
    @media (max-width: 860px) {{
      main {{ width: min(100vw - 22px, 1280px); margin-top: 20px; }}
      header {{ display: block; }}
      .subtitle {{ text-align: left; margin-top: 12px; }}
      .panel {{ padding: 18px; border-radius: 24px; }}
      .callout-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="section-kicker">dns3 final clustering</div>
        <h1>聚类结果静态展示</h1>
      </div>
      <p class="subtitle">标签来源：<code>{html.escape(meta['labels_path'])}</code>（SHA256: <code>{html.escape(meta['labels_sha256'])}</code>），当前标签分布为 {meta['label_cluster_count']} 簇 / {meta['sample_count']:,} 个域名。二维投影与画像使用 source feature pipeline 的 {meta['selected_feature_count']} 维 <code>{html.escape(meta['route'])}</code> 特征；source selection context: <code>{html.escape(meta['variant'])}</code> / {html.escape(meta['algorithm'])}。{html.escape(meta['context_note'])} 页面只保留二维散点与簇特征画像表。</p>
    </header>
    {scatter_svg}
    {profile_table}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    labels_sha256 = sha256_file(labels_path)

    selection = json.loads(FINAL_SELECTION_PATH.read_text(encoding="utf-8"))
    selected = selection["selected_candidate"]

    bundle = experiments.build_dns3_feature_bundle()
    routes = experiments.build_dns3_interpretable_routes(bundle)
    matrix, route_meta = routes[selected["route"]]
    feature_names = route_meta["feature_names"]

    meta = {
        "variant": selected["variant"],
        "algorithm": selected["algorithm"],
        "route": selected["route"],
        "source_selection_k": selected["k"],
        "selected_feature_count": route_meta["selected_feature_count"],
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha256,
        "context_note": args.context_note,
        "local_only": True,
    }
    labels_df = solver.pd.read_csv(labels_path)
    labels_df["fqdn_no"] = labels_df["fqdn_no"].astype(str)
    fqdn_order = bundle.fqdn["fqdn_no"].astype(str).tolist()
    ordered_labels = labels_df.set_index("fqdn_no").loc[fqdn_order, "label"].astype(int).to_numpy()

    embedding = project_to_2d(matrix)
    profiles = build_profiles(ordered_labels, matrix, embedding, feature_names)
    meta["label_cluster_count"] = len(profiles)
    meta["sample_count"] = len(ordered_labels)
    scatter_svg = render_svg(ordered_labels, embedding, profiles, meta)
    profile_table = render_profile_table(profiles)
    report_html = render_html(scatter_svg, profile_table, meta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding="utf-8")
    print(f"Wrote {output_path}")
    print(
        f"rows={len(labels_df)} clusters={len(profiles)} features={route_meta['selected_feature_count']} "
        f"route={selected['route']} labels_sha256={labels_sha256}"
    )


if __name__ == "__main__":
    main()
