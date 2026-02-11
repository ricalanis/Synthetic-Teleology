"""Export utilities for metrics reports.

Supports JSON, CSV, and self-contained HTML output formats.
All functions use only the Python standard library -- no extra
dependencies are required.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
from pathlib import Path
from typing import Any

from synthetic_teleology.measurement.report import MetricsReport

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def export_json(report: MetricsReport, path: str) -> None:
    """Export a single metrics report to a JSON file.

    Parameters
    ----------
    report:
        The report to export.
    path:
        File path for the JSON output.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_csv(reports: list[MetricsReport], path: str) -> None:
    """Export multiple metrics reports to a CSV file.

    Each row represents one agent.  Columns are: ``agent_id``,
    ``timestamp``, then one column per metric name.

    Parameters
    ----------
    reports:
        List of reports to export.
    path:
        File path for the CSV output.
    """
    if not reports:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")
        return

    # Gather all metric names across all reports (in order of first appearance)
    all_metrics: list[str] = []
    seen: set[str] = set()
    for rpt in reports:
        for name in rpt.metric_names:
            if name not in seen:
                all_metrics.append(name)
                seen.add(name)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # Header
        writer.writerow(["agent_id", "timestamp"] + all_metrics)
        # Data rows
        for rpt in reports:
            row: list[Any] = [rpt.agent_id, rpt.timestamp]
            values = rpt.values
            for metric_name in all_metrics:
                row.append(values.get(metric_name, ""))
            writer.writerow(row)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Metrics Report: {agent_id}</title>
<style>
  :root {{
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --border: #dee2e6;
    --text: #212529;
    --muted: #6c757d;
    --accent: #0d6efd;
    --good: #198754;
    --warn: #ffc107;
    --bad: #dc3545;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.6;
  }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{
    font-size: 1.75rem;
    margin-bottom: 0.25rem;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
  }}
  th, td {{
    text-align: left;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }}
  td.value {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }}
  .bar-container {{
    width: 120px;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  }}
  .bar-fill.excellent {{ background: var(--good); }}
  .bar-fill.good {{ background: #20c997; }}
  .bar-fill.fair {{ background: var(--warn); }}
  .bar-fill.poor {{ background: var(--bad); }}
  .rating {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }}
  .rating.excellent {{ background: #d1e7dd; color: #0f5132; }}
  .rating.good {{ background: #d1ecf1; color: #055160; }}
  .rating.fair {{ background: #fff3cd; color: #664d03; }}
  .rating.poor {{ background: #f8d7da; color: #842029; }}
  .metadata-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.75rem;
  }}
  .meta-item {{
    font-size: 0.9rem;
  }}
  .meta-item .label {{
    color: var(--muted);
    font-size: 0.8rem;
  }}
  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Metrics Report</h1>
  <p class="subtitle">Agent: <strong>{agent_id}</strong> &mdash; {timestamp_iso}</p>

  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th style="text-align:right">Value</th>
          <th>Progress</th>
          <th>Rating</th>
          <th>Explanation</th>
        </tr>
      </thead>
      <tbody>
{metric_rows}
      </tbody>
    </table>
  </div>

{metadata_section}

  <footer>
    Generated by Synthetic Teleology Framework v{version}
  </footer>
</div>
</body>
</html>"""


def _rating(value: float) -> tuple[str, str]:
    """Return ``(css_class, label)`` for a metric value in [0, 1]."""
    if value >= 0.8:
        return "excellent", "excellent"
    if value >= 0.6:
        return "good", "good"
    if value >= 0.4:
        return "fair", "fair"
    return "poor", "poor"


def _html_escape(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def export_html(report: MetricsReport, path: str) -> None:
    """Export a single metrics report to a self-contained HTML file.

    The generated file includes inline CSS and needs no external
    resources.

    Parameters
    ----------
    report:
        The report to export.
    path:
        File path for the HTML output.
    """
    from synthetic_teleology import __version__

    iso_ts = datetime.datetime.fromtimestamp(
        report.timestamp, tz=datetime.UTC
    ).isoformat()

    # Build metric rows
    rows_buf = io.StringIO()
    for r in report.results:
        css, label = _rating(r.value)
        pct = max(0.0, min(100.0, r.value * 100))
        explanation = _html_escape(r.explanation or "")
        rows_buf.write(
            f'        <tr>\n'
            f'          <td>{_html_escape(r.name)}</td>\n'
            f'          <td class="value">{r.value:.4f}</td>\n'
            f'          <td>'
            f'<div class="bar-container">'
            f'<div class="bar-fill {css}" style="width:{pct:.1f}%"></div>'
            f'</div></td>\n'
            f'          <td><span class="rating {css}">{label}</span></td>\n'
            f'          <td>{explanation}</td>\n'
            f'        </tr>\n'
        )

    # Build metadata section
    meta_buf = io.StringIO()
    if report.metadata:
        meta_buf.write('  <div class="card">\n')
        meta_buf.write('    <h3 style="margin-bottom:0.75rem">Metadata</h3>\n')
        meta_buf.write('    <div class="metadata-grid">\n')
        for key, val in report.metadata.items():
            meta_buf.write(
                f'      <div class="meta-item">'
                f'<div class="label">{_html_escape(str(key))}</div>'
                f'{_html_escape(str(val))}</div>\n'
            )
        meta_buf.write("    </div>\n")
        meta_buf.write("  </div>\n")

    html = _HTML_TEMPLATE.format(
        agent_id=_html_escape(report.agent_id),
        timestamp_iso=iso_ts,
        metric_rows=rows_buf.getvalue(),
        metadata_section=meta_buf.getvalue(),
        version=__version__,
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def export_all(
    reports: list[MetricsReport],
    output_dir: str,
    formats: list[str] | None = None,
) -> dict[str, list[str]]:
    """Export reports in multiple formats at once.

    Parameters
    ----------
    reports:
        Reports to export.
    output_dir:
        Directory to write output files into.
    formats:
        List of format strings: ``"json"``, ``"csv"``, ``"html"``.
        Defaults to ``["json", "csv"]``.

    Returns
    -------
    dict[str, list[str]]
        Mapping of format name to list of generated file paths.
    """
    if formats is None:
        formats = ["json", "csv"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result: dict[str, list[str]] = {}

    if "json" in formats:
        json_paths: list[str] = []
        for rpt in reports:
            fname = out / f"report_{rpt.agent_id}.json"
            export_json(rpt, str(fname))
            json_paths.append(str(fname))
        result["json"] = json_paths

    if "csv" in formats:
        csv_path = str(out / "reports.csv")
        export_csv(reports, csv_path)
        result["csv"] = [csv_path]

    if "html" in formats:
        html_paths: list[str] = []
        for rpt in reports:
            fname = out / f"report_{rpt.agent_id}.html"
            export_html(rpt, str(fname))
            html_paths.append(str(fname))
        result["html"] = html_paths

    return result
