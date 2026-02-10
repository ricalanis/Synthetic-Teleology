"""Presentation layer for the Synthetic Teleology framework.

Provides console output, plotting, and export utilities for metrics
reports and agent run data.  All external dependencies (``rich``,
``matplotlib``) are optional -- modules degrade gracefully when they
are not installed.

Public API
----------
- :class:`ConsoleDashboard` -- Rich-based (or plain-text) console output
- :func:`plot_score_trajectory`, :func:`plot_goal_revisions`,
  :func:`plot_metric_comparison`, :func:`plot_phase_portrait`,
  :func:`save_plots` -- matplotlib plotting helpers
- :func:`export_json`, :func:`export_csv`, :func:`export_html`,
  :func:`export_all` -- serialisation utilities
"""

from synthetic_teleology.presentation.console import ConsoleDashboard
from synthetic_teleology.presentation.export import (
    export_all,
    export_csv,
    export_html,
    export_json,
)
from synthetic_teleology.presentation.plots import (
    plot_goal_revisions,
    plot_metric_comparison,
    plot_phase_portrait,
    plot_score_trajectory,
    save_plots,
)

__all__ = [
    # Console
    "ConsoleDashboard",
    # Plots
    "plot_score_trajectory",
    "plot_goal_revisions",
    "plot_metric_comparison",
    "plot_phase_portrait",
    "save_plots",
    # Export
    "export_json",
    "export_csv",
    "export_html",
    "export_all",
]
