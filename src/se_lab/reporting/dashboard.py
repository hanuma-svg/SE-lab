from __future__ import annotations

import html
import json
from pathlib import Path

from se_lab.experiments.contracts import ExperimentResult


def write_experiment_dashboard(result: ExperimentResult, output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = result.report
    aggregation = result.aggregation.get("variants", {})
    rows = []
    for variant, values in sorted(aggregation.items()):
        ci = values.get("pass_rate_ci95", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(variant)}</td>"
            f"<td>{values.get('sample_count', 0)}</td>"
            f"<td>{float(values.get('pass_rate', 0.0)):.1%}</td>"
            f"<td>{float(ci.get('low', 0.0)):.1%} – {float(ci.get('high', 0.0)):.1%}</td>"
            f"<td>{values.get('mean_model_calls', '—')}</td>"
            f"<td>{values.get('mean_tool_calls', '—')}</td>"
            f"<td>{values.get('mean_wall_clock_duration_seconds', '—')}</td>"
            "</tr>"
        )
    status = "PASS" if not result.mismatches else "MISMATCH"
    payload = json.dumps(result.model_dump(mode="json"), sort_keys=True).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SE-Lab · {html.escape(result.experiment_id)}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background:#07111f; color:#e5edf7; }}
body {{ margin:0; background:radial-gradient(circle at top right,#17365d 0,#07111f 42rem); min-height:100vh; }}
main {{ max-width:1180px; margin:auto; padding:42px 24px 72px; }}
.eyebrow {{ color:#6ee7d8; text-transform:uppercase; letter-spacing:.14em; font-size:12px; font-weight:700; }}
h1 {{ font-size:42px; margin:8px 0 10px; letter-spacing:-.04em; }}
.muted {{ color:#93a7bf; }} .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:26px 0; }}
.card {{ background:rgba(13,28,49,.86); border:1px solid #28425f; border-radius:16px; padding:18px; box-shadow:0 14px 42px rgba(0,0,0,.2); }}
.metric {{ font-size:28px; font-weight:750; margin-top:8px; }} table {{ width:100%; border-collapse:collapse; margin-top:14px; }} th,td {{ text-align:left; padding:13px 10px; border-bottom:1px solid #203852; }} th {{ color:#94b6d8; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.badge {{ display:inline-block; border-radius:999px; padding:5px 11px; background:{'#123f3a' if status == 'PASS' else '#4b3211'}; color:{'#6ee7d8' if status == 'PASS' else '#f5c56b'}; font-weight:700; }}
pre {{ white-space:pre-wrap; background:#050b14; padding:16px; border-radius:12px; overflow:auto; color:#bed0e5; }}
@media(max-width:800px) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} h1 {{font-size:32px}} }}
</style></head><body><main>
<div class="eyebrow">Software Engineering Evaluation Lab</div>
<h1>{html.escape(result.experiment_id)}</h1><p class="muted">Matched baseline/treatment experiment · descriptive evidence only</p>
<span class="badge">{status}</span>
<section class="grid">
<div class="card"><div class="muted">Runs</div><div class="metric">{len(result.runs)}</div></div>
<div class="card"><div class="muted">Variants</div><div class="metric">{len(aggregation)}</div></div>
<div class="card"><div class="muted">Budget matched</div><div class="metric">{'YES' if result.has_matched_budgets else 'NO'}</div></div>
<div class="card"><div class="muted">Mismatches</div><div class="metric">{len(result.mismatches)}</div></div>
</section>
<div class="card"><h2>Outcome and efficiency</h2><table><thead><tr><th>Variant</th><th>n</th><th>Pass@1</th><th>95% Wilson CI</th><th>Model calls</th><th>Tool calls</th><th>Wall clock (s)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class="card" style="margin-top:16px"><h2>Research caveats</h2><p class="muted">{html.escape(report.get('scientific_note', 'Descriptive measurements only.'))}</p><p class="muted">This dashboard intentionally exposes task-level evidence and does not claim superiority from a small sample.</p></div>
<details style="margin-top:16px"><summary>Machine-readable report</summary><pre id="data"></pre></details>
<script>document.getElementById('data').textContent = JSON.stringify({payload}, null, 2);</script>
</main></body></html>"""
    target.write_text(document, encoding="utf-8")
    return target
