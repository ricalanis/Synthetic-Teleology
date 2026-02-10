"""Rich-based console dashboard with plain-text fallback.

If the ``rich`` library is installed (part of the ``viz`` optional extra),
:class:`ConsoleDashboard` renders tables and sparklines with colour and
formatting.  Otherwise it falls back to simple ``print()``-based output
that works in any terminal.
"""

from __future__ import annotations

import sys
from typing import Any

from synthetic_teleology.measurement.report import MetricsReport

# ---------------------------------------------------------------------------
# Graceful rich import
# ---------------------------------------------------------------------------

try:
    from rich.console import Console as RichConsole
    from rich.table import Table as RichTable
    from rich.text import Text as RichText

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# Sparkline helpers
# ---------------------------------------------------------------------------

_SPARK_CHARS = " " + "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def _sparkline(values: list[float], width: int = 60) -> str:
    """Return an ASCII/Unicode sparkline string for *values*.

    Parameters
    ----------
    values:
        Sequence of numeric values to visualise.
    width:
        Maximum character width.  If ``len(values) > width``, the
        values are down-sampled by averaging adjacent bins.
    """
    if not values:
        return ""

    # Down-sample if needed
    if len(values) > width:
        bin_size = len(values) / width
        sampled: list[float] = []
        for i in range(width):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            chunk = values[start:end]
            sampled.append(sum(chunk) / len(chunk) if chunk else 0.0)
        values = sampled

    lo = min(values)
    hi = max(values)
    span = hi - lo if hi != lo else 1.0
    n_chars = len(_SPARK_CHARS) - 1

    chars: list[str] = []
    for v in values:
        idx = int(((v - lo) / span) * n_chars)
        idx = max(0, min(n_chars, idx))
        chars.append(_SPARK_CHARS[idx])

    return "".join(chars)


# ---------------------------------------------------------------------------
# ConsoleDashboard
# ---------------------------------------------------------------------------

class ConsoleDashboard:
    """Console presentation layer for metrics reports and trajectories.

    Automatically selects rich-based rendering when ``rich`` is installed
    or falls back to plain text.

    Parameters
    ----------
    use_rich:
        Explicitly enable (``True``) or disable (``False``) rich output.
        ``None`` (default) auto-detects based on availability.
    file:
        Output stream.  Defaults to ``sys.stdout``.
    """

    def __init__(
        self,
        use_rich: bool | None = None,
        file: Any = None,
    ) -> None:
        self._file = file or sys.stdout
        if use_rich is None:
            self._use_rich = _HAS_RICH
        else:
            self._use_rich = use_rich and _HAS_RICH

        if self._use_rich:
            self._console = RichConsole(file=self._file)
        else:
            self._console = None

    # -- helpers -----------------------------------------------------------

    def _plain_print(self, *args: Any, **kwargs: Any) -> None:
        """Print to the configured output stream."""
        kwargs.setdefault("file", self._file)
        print(*args, **kwargs)

    # -- public API --------------------------------------------------------

    def print_report(self, report: MetricsReport) -> None:
        """Print a single metrics report as a formatted table.

        Parameters
        ----------
        report:
            The :class:`MetricsReport` to display.
        """
        if self._use_rich and self._console is not None:
            self._print_report_rich(report)
        else:
            self._print_report_plain(report)

    def print_comparison(self, reports: list[MetricsReport]) -> None:
        """Print a side-by-side comparison of multiple reports.

        Each metric appears as a row; each report as a column.

        Parameters
        ----------
        reports:
            List of :class:`MetricsReport` instances to compare.
        """
        if not reports:
            self._plain_print("[no reports to compare]")
            return

        if self._use_rich and self._console is not None:
            self._print_comparison_rich(reports)
        else:
            self._print_comparison_plain(reports)

    def print_trajectory(self, scores: list[float], width: int = 60) -> None:
        """Print an ASCII sparkline of evaluation scores over time.

        Parameters
        ----------
        scores:
            List of score values (typically from :meth:`AgentLog.get_scores`).
        width:
            Maximum character width for the sparkline.
        """
        if not scores:
            self._plain_print("[no scores to display]")
            return

        spark = _sparkline(scores, width)
        lo, hi = min(scores), max(scores)
        avg = sum(scores) / len(scores)

        if self._use_rich and self._console is not None:
            self._console.print()
            self._console.print("[bold]Score Trajectory[/bold]")
            self._console.print(spark)
            self._console.print(
                f"  min={lo:.4f}  max={hi:.4f}  avg={avg:.4f}  steps={len(scores)}"
            )
            self._console.print()
        else:
            self._plain_print()
            self._plain_print("Score Trajectory")
            self._plain_print(spark)
            self._plain_print(
                f"  min={lo:.4f}  max={hi:.4f}  avg={avg:.4f}  steps={len(scores)}"
            )
            self._plain_print()

    def print_benchmark_results(
        self, results: dict[str, list[MetricsReport]]
    ) -> None:
        """Print a summary table of benchmark results.

        Parameters
        ----------
        results:
            Mapping of ``suite_name -> list[MetricsReport]``.  Each list
            contains reports from multiple runs of the same benchmark
            configuration.
        """
        if not results:
            self._plain_print("[no benchmark results]")
            return

        if self._use_rich and self._console is not None:
            self._print_benchmark_rich(results)
        else:
            self._print_benchmark_plain(results)

    # ======================================================================
    # Rich implementations
    # ======================================================================

    def _print_report_rich(self, report: MetricsReport) -> None:
        assert self._console is not None
        table = RichTable(
            title=f"Metrics Report: {report.agent_id}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Rating", justify="center")

        for r in report.results:
            if r.value >= 0.8:
                colour = "green"
                rating = "excellent"
            elif r.value >= 0.6:
                colour = "yellow"
                rating = "good"
            elif r.value >= 0.4:
                colour = "orange3"
                rating = "fair"
            else:
                colour = "red"
                rating = "poor"

            table.add_row(
                r.name,
                f"[{colour}]{r.value:.4f}[/{colour}]",
                f"[{colour}]{rating}[/{colour}]",
            )

        self._console.print()
        self._console.print(table)

        if report.metadata:
            self._console.print()
            for key, val in report.metadata.items():
                self._console.print(f"  [dim]{key}:[/dim] {val}")
        self._console.print()

    def _print_comparison_rich(self, reports: list[MetricsReport]) -> None:
        assert self._console is not None

        # Collect all unique metric names in order of first appearance
        all_metrics: list[str] = []
        seen: set[str] = set()
        for rpt in reports:
            for name in rpt.metric_names:
                if name not in seen:
                    all_metrics.append(name)
                    seen.add(name)

        table = RichTable(
            title="Metric Comparison",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        for rpt in reports:
            table.add_column(rpt.agent_id, justify="right")

        for metric_name in all_metrics:
            row: list[str] = [metric_name]
            values_for_metric: list[float] = []
            for rpt in reports:
                m = rpt.get_metric(metric_name)
                if m is not None:
                    values_for_metric.append(m.value)
                else:
                    values_for_metric.append(float("nan"))

            # Find the best value for highlighting
            valid_vals = [v for v in values_for_metric if v == v]  # filter NaN
            best_val = max(valid_vals) if valid_vals else None

            for v in values_for_metric:
                if v != v:  # NaN
                    row.append("[dim]N/A[/dim]")
                elif best_val is not None and v == best_val and len(valid_vals) > 1:
                    row.append(f"[bold green]{v:.4f}[/bold green]")
                else:
                    row.append(f"{v:.4f}")

            table.add_row(*row)

        self._console.print()
        self._console.print(table)
        self._console.print()

    def _print_benchmark_rich(
        self, results: dict[str, list[MetricsReport]]
    ) -> None:
        assert self._console is not None

        # Collect all metric names from all reports
        all_metrics: list[str] = []
        seen: set[str] = set()
        for suite_reports in results.values():
            for rpt in suite_reports:
                for name in rpt.metric_names:
                    if name not in seen:
                        all_metrics.append(name)
                        seen.add(name)

        table = RichTable(
            title="Benchmark Results",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Suite", style="bold")
        table.add_column("Runs", justify="right")
        for metric_name in all_metrics:
            table.add_column(metric_name, justify="right")

        for suite_name, suite_reports in results.items():
            row: list[str] = [suite_name, str(len(suite_reports))]
            for metric_name in all_metrics:
                values: list[float] = []
                for rpt in suite_reports:
                    m = rpt.get_metric(metric_name)
                    if m is not None:
                        values.append(m.value)
                if values:
                    avg = sum(values) / len(values)
                    std = (
                        (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
                        if len(values) > 1
                        else 0.0
                    )
                    row.append(f"{avg:.3f} +/- {std:.3f}")
                else:
                    row.append("[dim]N/A[/dim]")
            table.add_row(*row)

        self._console.print()
        self._console.print(table)
        self._console.print()

    # ======================================================================
    # Plain-text implementations
    # ======================================================================

    def _print_report_plain(self, report: MetricsReport) -> None:
        self._plain_print()
        self._plain_print(report.summary())
        self._plain_print()

    def _print_comparison_plain(self, reports: list[MetricsReport]) -> None:
        # Collect all unique metric names
        all_metrics: list[str] = []
        seen: set[str] = set()
        for rpt in reports:
            for name in rpt.metric_names:
                if name not in seen:
                    all_metrics.append(name)
                    seen.add(name)

        # Column widths
        metric_w = max(len("Metric"), *(len(m) for m in all_metrics)) if all_metrics else 10
        agent_w = max(10, *(len(rpt.agent_id) for rpt in reports)) if reports else 10

        self._plain_print()
        self._plain_print("=== Metric Comparison ===")

        # Header
        header = f"{'Metric':<{metric_w}}"
        for rpt in reports:
            header += f"  {rpt.agent_id:>{agent_w}}"
        self._plain_print(header)
        self._plain_print("-" * len(header))

        # Rows
        for metric_name in all_metrics:
            row = f"{metric_name:<{metric_w}}"
            for rpt in reports:
                m = rpt.get_metric(metric_name)
                if m is not None:
                    row += f"  {m.value:>{agent_w}.4f}"
                else:
                    row += f"  {'N/A':>{agent_w}}"
            self._plain_print(row)

        self._plain_print("-" * len(header))
        self._plain_print()

    def _print_benchmark_plain(
        self, results: dict[str, list[MetricsReport]]
    ) -> None:
        # Collect all metric names
        all_metrics: list[str] = []
        seen: set[str] = set()
        for suite_reports in results.values():
            for rpt in suite_reports:
                for name in rpt.metric_names:
                    if name not in seen:
                        all_metrics.append(name)
                        seen.add(name)

        suite_w = max(10, *(len(s) for s in results)) if results else 10
        col_w = 16

        self._plain_print()
        self._plain_print("=== Benchmark Results ===")

        # Header
        header = f"{'Suite':<{suite_w}}  {'Runs':>5}"
        for metric_name in all_metrics:
            header += f"  {metric_name:>{col_w}}"
        self._plain_print(header)
        self._plain_print("-" * len(header))

        for suite_name, suite_reports in results.items():
            row = f"{suite_name:<{suite_w}}  {len(suite_reports):>5}"
            for metric_name in all_metrics:
                values: list[float] = []
                for rpt in suite_reports:
                    m = rpt.get_metric(metric_name)
                    if m is not None:
                        values.append(m.value)
                if values:
                    avg = sum(values) / len(values)
                    std = (
                        (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
                        if len(values) > 1
                        else 0.0
                    )
                    row += f"  {avg:>{col_w - 8}.3f} +/- {std:.3f}"
                else:
                    row += f"  {'N/A':>{col_w}}"
            self._plain_print(row)

        self._plain_print("-" * len(header))
        self._plain_print()
