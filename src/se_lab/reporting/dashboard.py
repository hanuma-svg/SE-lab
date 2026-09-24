from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from se_lab.experiments.contracts import ExperimentResult
from se_lab.experiments.statistics import wilson_interval


def write_experiment_dashboard(result: ExperimentResult, output: str | Path) -> Path:
    """Write a self-contained, evidence-oriented report with no fabricated metrics."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = result.report
    aggregation = {
        variant: _normalize_variant(variant, values, result)
        for variant, values in result.aggregation.get("variants", {}).items()
    }
    status = "PASS" if not result.mismatches else "MISMATCH"
    status_class = "pass" if status == "PASS" else "warning"
    scope = [
        "deterministic/mock provider",
        "curated smoke benchmark",
        "bounded execution",
        "descriptive statistics",
    ]
    payload = json.dumps(result.model_dump(mode="json"), sort_keys=True).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SE-Lab / {html.escape(result.experiment_id)}</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: #f7f7f5;
      --surface: #ffffff;
      --ink: #202124;
      --muted: #6b7075;
      --line: #d9dcdf;
      --accent: #2457a6;
      --pass: #247447;
      --warning: #9a6500;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: var(--sans); line-height: 1.5; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px 64px; }}
    header {{ display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; padding-bottom: 24px; border-bottom: 1px solid var(--line); }}
    .brand {{ font-weight: 750; letter-spacing: -.02em; }}
    .eyebrow, .label {{ color: var(--muted); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; font-weight: 700; }}
    h1 {{ font-size: clamp(28px, 4vw, 42px); line-height: 1.08; letter-spacing: -.04em; margin: 10px 0 8px; font-weight: 760; }}
    h2 {{ font-size: 18px; letter-spacing: -.015em; margin: 0 0 14px; }}
    h3 {{ font-size: 14px; margin: 0 0 8px; }}
    p {{ margin: 0 0 12px; }}
    .muted {{ color: var(--muted); }}
    .mono {{ font-family: var(--mono); font-size: .92em; }}
    .status {{ white-space: nowrap; border: 1px solid currentColor; border-radius: 3px; padding: 4px 9px; font-size: 12px; font-weight: 750; letter-spacing: .06em; }}
    .status.pass {{ color: var(--pass); }} .status.warning {{ color: var(--warning); }}
    .section {{ padding: 26px 0; border-bottom: 1px solid var(--line); }}
    .section-header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 14px; }}
    .architecture {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; align-items: center; color: var(--muted); font-size: 12px; text-align: center; }}
    .architecture span {{ border: 1px solid var(--line); background: var(--surface); padding: 10px 6px; }}
    .architecture b {{ color: var(--accent); font-size: 14px; }}
    table {{ border-collapse: collapse; width: 100%; background: var(--surface); border: 1px solid var(--line); }}
    th, td {{ text-align: left; vertical-align: top; padding: 11px 12px; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--muted); font-size: 11px; letter-spacing: .07em; text-transform: uppercase; background: #f1f2f1; }}
    tr:last-child td {{ border-bottom: 0; }}
    .comparison td:first-child {{ font-weight: 650; }}
    .comparison td:not(:first-child) {{ font-family: var(--mono); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }}
    .definition {{ display: grid; grid-template-columns: 170px 1fr; border-top: 1px solid var(--line); }}
    .definition dt, .definition dd {{ margin: 0; padding: 9px 0; border-bottom: 1px solid var(--line); }}
    .definition dt {{ color: var(--muted); }}
    .definition dd {{ font-family: var(--mono); font-size: 13px; }}
    .scope {{ margin: 0; padding-left: 18px; color: var(--muted); }}
    .scope li {{ margin: 4px 0; }}
    .note {{ border-left: 3px solid var(--accent); padding: 10px 14px; background: #eef3fb; color: #33465e; }}
    details {{ margin-top: 18px; }} summary {{ cursor: pointer; color: var(--accent); font-weight: 650; }}
    pre {{ overflow: auto; background: #202124; color: #eef0f2; padding: 16px; font: 12px/1.5 var(--mono); }}
    footer {{ padding-top: 20px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 760px) {{
      main {{ padding: 22px 16px 48px; }} header, .section-header {{ display: block; }} header .status {{ display: inline-block; margin-top: 16px; }}
      .architecture {{ grid-template-columns: 1fr; }} .architecture b {{ transform: rotate(90deg); }} .grid {{ grid-template-columns: 1fr; }}
      .definition {{ grid-template-columns: 1fr; }} .definition dt {{ padding-bottom: 2px; border-bottom: 0; }} .definition dd {{ padding-top: 2px; }}
      table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="brand">SE-Lab</div>
      <div class="eyebrow" style="margin-top: 20px">Experiment report</div>
      <h1>{html.escape(result.experiment_id)}</h1>
      <p class="muted">Controlled comparison of a baseline and a bounded multi-agent software-engineering workflow.</p>
    </div>
    <div class="status {status_class}">● {status}</div>
  </header>

  <section class="section">
    <div class="section-header"><h2>Execution path</h2><span class="label">Observed architecture</span></div>
    <div class="architecture">
      <span>Benchmark task</span><b>→</b><span>Baseline / multi-agent</span><b>→</b><span>Policy + execution</span><b>→</b><span>Evidence + evaluation</span>
    </div>
    <div class="architecture" style="margin-top: 8px; grid-template-columns: repeat(3, 1fr)">
      <span>Experiment result</span><b>→</b><span>Statistics + SQLite registry</span>
    </div>
  </section>

  <section class="section">
    <div class="section-header"><h2>Experiment metadata</h2><span class="label">No synthetic values</span></div>
    <dl class="definition">
      <dt>Task</dt><dd>{html.escape(str(report.get('task_id', '—')))}</dd>
      <dt>Provider</dt><dd>{html.escape(str(report.get('provider', {}).get('name', '—')))} / {html.escape(str(report.get('provider', {}).get('model', '—')))}</dd>
      <dt>Runs</dt><dd>{len(result.runs)}</dd>
      <dt>Budget valid</dt><dd>{'yes' if result.has_matched_budgets else 'no'}</dd>
      <dt>Mismatch count</dt><dd>{len(result.mismatches)}</dd>
    </dl>
  </section>

  <section class="section">
    <div class="section-header"><h2>Baseline vs multi-agent</h2><span class="label">Descriptive comparison</span></div>
    <table class="comparison">
      <thead><tr><th>Metric</th>{''.join(f'<th>{html.escape(variant)}</th>' for variant in sorted(aggregation))}</tr></thead>
      <tbody>
        {_comparison_row('Pass@1', aggregation, lambda values: f"{int(values.get('successes', 0))} / {int(values.get('sample_count', 0))}")}
        {_comparison_row('95% Wilson CI', aggregation, lambda values: _interval(values))}
        {_comparison_row('Mean wall time', aggregation, lambda values: _number(values.get('mean_wall_clock_duration_seconds')) + ' s')}
        {_comparison_row('Median wall time', aggregation, lambda values: _number(values.get('median_wall_clock_duration_seconds')) + ' s')}
        {_comparison_row('Mean model calls', aggregation, lambda values: _number(values.get('mean_model_calls')))}
        {_comparison_row('Mean tool calls', aggregation, lambda values: _number(values.get('mean_tool_calls')))}
        {_comparison_row('Budget valid', aggregation, lambda values: 'yes' if result.has_matched_budgets else 'no')}
      </tbody>
    </table>
  </section>

  <section class="section grid">
    <div><h2>Failure classes</h2><table><thead><tr><th>Variant</th><th>Observed classes</th></tr></thead><tbody>{''.join(_failure_row(variant, values) for variant, values in sorted(aggregation.items()))}</tbody></table></div>
    <div><h2>Scope</h2><ul class="scope">{''.join(f'<li>{html.escape(item)}</li>' for item in scope)}</ul><p class="note" style="margin-top: 16px">This report demonstrates evaluation infrastructure. It does not establish multi-agent superiority or statistical significance.</p></div>
  </section>

  <section class="section">
    <h2>Technical notes</h2>
    <p class="muted">Runs preserve model calls, tool calls, retries, wall-clock duration, failure class, budget validity, evidence digest, and evidence directory. The evaluator consumes externally observable evidence rather than agent self-reports.</p>
    <details><summary>Machine-readable report</summary><pre id="data"></pre></details>
  </section>
  <footer>Generated from the recorded experiment result. No metrics are hard-coded into this report.</footer>
</main>
<script>document.getElementById('data').textContent = JSON.stringify({payload}, null, 2);</script>
</body>
</html>"""
    target.write_text(document, encoding="utf-8")
    return target


def _comparison_row(label: str, aggregation: dict[str, Any], formatter: Any) -> str:
    cells = "".join(f"<td>{html.escape(str(formatter(values)))}</td>" for values in aggregation.values())
    return f"<tr><td>{html.escape(label)}</td>{cells}</tr>"


def _normalize_variant(variant: str, values: dict[str, Any], result: ExperimentResult) -> dict[str, Any]:
    """Backfill display fields from recorded runs for historical result JSON."""
    normalized = dict(values)
    runs = [run for run in result.runs if run.variant == variant]
    sample_count = len(runs)
    successes = sum(run.pass_at_1 for run in runs)
    normalized["sample_count"] = sample_count
    normalized["successes"] = successes
    normalized["pass_rate"] = successes / sample_count if sample_count else 0.0
    low, high = wilson_interval(successes, sample_count)
    normalized["pass_rate_ci95"] = {"low": low, "high": high}
    failure_classes: dict[str, int] = {}
    for run in runs:
        failure_classes[run.failure_class] = failure_classes.get(run.failure_class, 0) + 1
    normalized["failure_classes"] = dict(sorted(failure_classes.items()))
    return normalized


def _failure_row(variant: str, values: dict[str, Any]) -> str:
    failures = values.get("failure_classes", {})
    classes = ", ".join(f"{key}: {value}" for key, value in failures.items()) or "none recorded"
    return f"<tr><td class=\"mono\">{html.escape(variant)}</td><td class=\"mono\">{html.escape(classes)}</td></tr>"


def _interval(values: dict[str, Any]) -> str:
    interval = values.get("pass_rate_ci95", {})
    return f"[{_number(interval.get('low'))}, {_number(interval.get('high'))}]"


def _number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
