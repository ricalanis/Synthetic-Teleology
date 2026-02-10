"""Matplotlib-based plotting utilities for the Synthetic Teleology framework.

All functions return ``matplotlib.figure.Figure`` objects so the caller
decides whether to ``show()``, ``savefig()``, or embed them.  If
``matplotlib`` is not installed the module still imports; each function
raises :class:`ImportError` with a clear message at call time.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from synthetic_teleology.measurement.report import MetricsReport

# ---------------------------------------------------------------------------
# Graceful matplotlib import
# ---------------------------------------------------------------------------

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False
    Figure = Any  # type: ignore[assignment,misc]


def _require_matplotlib() -> None:
    """Raise a clear error if matplotlib is not available."""
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting.  "
            "Install it with: pip install synthetic-teleology[viz]"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_score_trajectory(
    scores: list[float],
    title: str = "",
) -> Figure:
    """Plot evaluation scores over time as a line chart.

    Parameters
    ----------
    scores:
        List of evaluation scores (one per step).
    title:
        Optional title for the figure.

    Returns
    -------
    Figure
        A matplotlib Figure ready for display or saving.
    """
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 4))
    steps = list(range(1, len(scores) + 1))
    ax.plot(steps, scores, linewidth=1.5, color="#2196F3", label="score")

    # Moving average if enough data
    if len(scores) >= 10:
        window = max(3, len(scores) // 10)
        ma: list[float] = []
        for i in range(len(scores)):
            start = max(0, i - window + 1)
            chunk = scores[start : i + 1]
            ma.append(sum(chunk) / len(chunk))
        ax.plot(
            steps,
            ma,
            linewidth=2.0,
            color="#FF9800",
            alpha=0.8,
            linestyle="--",
            label=f"MA({window})",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title(title or "Score Trajectory")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_goal_revisions(
    revisions: list[int],
    scores: list[float],
) -> Figure:
    """Plot scores with goal-revision points highlighted.

    Parameters
    ----------
    revisions:
        List of step indices where goal revisions occurred.
    scores:
        List of evaluation scores (one per step).

    Returns
    -------
    Figure
        A matplotlib Figure.
    """
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 4))
    steps = list(range(1, len(scores) + 1))
    ax.plot(steps, scores, linewidth=1.5, color="#2196F3", label="score")

    # Mark revision points
    for rev_step in revisions:
        if 1 <= rev_step <= len(scores):
            ax.axvline(
                x=rev_step,
                color="#F44336",
                linestyle="--",
                alpha=0.6,
                linewidth=1.0,
            )
            ax.plot(
                rev_step,
                scores[rev_step - 1],
                marker="D",
                markersize=8,
                color="#F44336",
                zorder=5,
            )

    # Add legend entry for revisions if any exist
    if revisions:
        ax.plot([], [], "D", color="#F44336", markersize=8, label="goal revision")

    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title("Score Trajectory with Goal Revisions")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metric_comparison(
    reports: list[MetricsReport],
) -> Figure:
    """Plot a radar (spider) chart comparing metrics across agents.

    Each agent is drawn as a polygon on the radar chart, with one axis
    per metric.  Metrics are assumed to be in [0, 1].

    Parameters
    ----------
    reports:
        List of :class:`MetricsReport` instances to compare.

    Returns
    -------
    Figure
        A matplotlib Figure with the radar chart.
    """
    _require_matplotlib()

    if not reports:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No reports", ha="center", va="center")
        return fig

    # Collect unique metric names in order
    all_metrics: list[str] = []
    seen: set[str] = set()
    for rpt in reports:
        for name in rpt.metric_names:
            if name not in seen:
                all_metrics.append(name)
                seen.add(name)

    n_metrics = len(all_metrics)
    if n_metrics < 3:
        # Fall back to bar chart if fewer than 3 metrics
        return _bar_comparison(reports, all_metrics)

    # Radar chart angles
    angles = [i * 2 * math.pi / n_metrics for i in range(n_metrics)]
    angles.append(angles[0])  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    # Colour cycle
    colours = [
        "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5",
    ]

    for idx, rpt in enumerate(reports):
        values: list[float] = []
        for metric_name in all_metrics:
            m = rpt.get_metric(metric_name)
            values.append(m.value if m is not None else 0.0)
        values.append(values[0])  # close polygon

        colour = colours[idx % len(colours)]
        ax.plot(angles, values, linewidth=2, color=colour, label=rpt.agent_id)
        ax.fill(angles, values, color=colour, alpha=0.1)

    ax.set_thetagrids(
        [a * 180 / math.pi for a in angles[:-1]],
        all_metrics,
        fontsize=8,
    )
    ax.set_ylim(0, 1.0)
    ax.set_title("Metric Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def _bar_comparison(
    reports: list[MetricsReport],
    metric_names: list[str],
) -> Figure:
    """Fallback bar chart for fewer than 3 metrics."""
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 5))
    n_metrics = len(metric_names)
    n_agents = len(reports)

    if n_agents == 0 or n_metrics == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    bar_width = 0.8 / n_agents
    colours = [
        "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5",
    ]

    for agent_idx, rpt in enumerate(reports):
        x_positions = [i + agent_idx * bar_width for i in range(n_metrics)]
        values = []
        for metric_name in metric_names:
            m = rpt.get_metric(metric_name)
            values.append(m.value if m is not None else 0.0)

        ax.bar(
            x_positions,
            values,
            width=bar_width,
            color=colours[agent_idx % len(colours)],
            label=rpt.agent_id,
            alpha=0.85,
        )

    ax.set_xticks([i + bar_width * (n_agents - 1) / 2 for i in range(n_metrics)])
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Metric Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_phase_portrait(
    states: list[tuple[float, ...]],
    dims: tuple[int, int] = (0, 1),
) -> Figure:
    """Plot a 2-D phase portrait of agent state trajectories.

    Projects the state history onto two selected dimensions and draws
    the trajectory with time-encoded colour (early steps are light,
    late steps are dark).

    Parameters
    ----------
    states:
        List of state-value tuples (one per step).
    dims:
        Which two dimensions to project onto.  Defaults to ``(0, 1)``.

    Returns
    -------
    Figure
        A matplotlib Figure with the phase portrait.
    """
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 8))

    if not states:
        ax.text(0.5, 0.5, "No states", ha="center", va="center")
        return fig

    dim_x, dim_y = dims
    max_dim = max(dim_x, dim_y)

    xs: list[float] = []
    ys: list[float] = []
    for s in states:
        if len(s) <= max_dim:
            # Pad with zeros if state has fewer dimensions
            padded = s + (0.0,) * (max_dim + 1 - len(s))
            xs.append(padded[dim_x])
            ys.append(padded[dim_y])
        else:
            xs.append(s[dim_x])
            ys.append(s[dim_y])

    # Plot trajectory with time-colour gradient
    n = len(xs)
    cmap = plt.cm.viridis  # type: ignore[attr-defined]
    for i in range(n - 1):
        colour = cmap(i / max(n - 1, 1))
        ax.plot(
            [xs[i], xs[i + 1]],
            [ys[i], ys[i + 1]],
            color=colour,
            linewidth=1.5,
            alpha=0.7,
        )

    # Mark start and end
    ax.plot(xs[0], ys[0], "o", markersize=10, color="#4CAF50", label="start", zorder=5)
    if n > 1:
        ax.plot(xs[-1], ys[-1], "s", markersize=10, color="#F44336", label="end", zorder=5)

    # Add colourbar to indicate time progression
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=1, vmax=n),  # type: ignore[attr-defined]
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Step", shrink=0.8)

    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")
    ax.set_title("Phase Portrait")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_plots(
    figures: list[Figure],
    output_dir: str,
    prefix: str = "plot",
    fmt: str = "png",
    dpi: int = 150,
) -> list[str]:
    """Save a list of figures to disk.

    Parameters
    ----------
    figures:
        Figures to save.
    output_dir:
        Directory to write the image files into.
    prefix:
        Filename prefix.  Files are named ``{prefix}_{i}.{fmt}``.
    fmt:
        Image format (e.g. ``"png"``, ``"pdf"``, ``"svg"``).
    dpi:
        Resolution in dots-per-inch.

    Returns
    -------
    list[str]
        Paths to the saved files.
    """
    _require_matplotlib()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for i, fig in enumerate(figures):
        filename = out / f"{prefix}_{i}.{fmt}"
        fig.savefig(str(filename), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(filename))

    return saved
