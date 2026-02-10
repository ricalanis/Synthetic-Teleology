"""Command-line interface for the Synthetic Teleology framework.

Provides subcommands for running agents, benchmarks, reports, and
querying framework information.  Each subcommand imports its
dependencies lazily so that ``synthetic-teleology info`` works even
when heavy optional dependencies are missing.

Entry point
-----------
The ``main()`` function is registered as a console script in
``pyproject.toml``::

    [project.scripts]
    synthetic-teleology = "synthetic_teleology.cli:main"

Usage examples::

    synthetic-teleology run --agent-type simple --env-type numeric --steps 50
    synthetic-teleology benchmark --suite default --runs 5 --output ./results
    synthetic-teleology report --input report_agent-1.json --format table
    synthetic-teleology info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="synthetic-teleology",
        description=(
            "Synthetic Teleology Framework -- CLI for running teleological "
            "agents, benchmarks, and generating reports."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="Show framework version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # -- run ---------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single agent loop.",
        description="Run a teleological agent in a specified environment.",
    )
    run_parser.add_argument(
        "--agent-type",
        type=str,
        default="simple",
        help=(
            "Agent type to create.  'simple' uses AgentFactory.create_simple_agent "
            "with default settings.  (default: simple)"
        ),
    )
    run_parser.add_argument(
        "--env-type",
        type=str,
        default="numeric",
        choices=["numeric", "resource", "research"],
        help="Environment type to use. (default: numeric)",
    )
    run_parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of loop steps to run. (default: 50)",
    )
    run_parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help=(
            "Goal target values as comma-separated floats "
            "(e.g. '1.0,2.0,3.0').  Defaults to a 2-D target of (1.0, 1.0)."
        ),
    )
    run_parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Environment noise standard deviation. (default: 0.0)",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the report files.  If omitted, prints to stdout.",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    # -- benchmark ---------------------------------------------------------
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks.",
        description="Run a benchmark suite and collect metrics across multiple runs.",
    )
    bench_parser.add_argument(
        "--suite",
        type=str,
        default="default",
        help="Benchmark suite name. (default: default)",
    )
    bench_parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs. (default: 5)",
    )
    bench_parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Steps per run. (default: 50)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. (default: 42)",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default="./benchmark_results",
        help="Output directory for benchmark results. (default: ./benchmark_results)",
    )

    # -- report ------------------------------------------------------------
    report_parser = subparsers.add_parser(
        "report",
        help="Load and display a saved report.",
        description="Load a previously exported JSON report and display it.",
    )
    report_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the JSON report file.",
    )
    report_parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "json", "summary"],
        help="Display format. (default: table)",
    )

    # -- info --------------------------------------------------------------
    subparsers.add_parser(
        "info",
        help="Show framework version and registered components.",
        description="Display version, registered components, and optional dependency status.",
    )

    return parser


# =========================================================================
# Subcommand handlers
# =========================================================================

def _cmd_run(args: argparse.Namespace) -> int:
    """Handle the ``run`` subcommand."""
    import time

    import numpy as np

    from synthetic_teleology.agents.factory import AgentFactory
    from synthetic_teleology.environments.numeric import NumericEnvironment
    from synthetic_teleology.infrastructure.event_bus import EventBus
    from synthetic_teleology.measurement.collector import EventCollector
    from synthetic_teleology.measurement.engine import MetricsEngine
    from synthetic_teleology.presentation.console import ConsoleDashboard

    if args.seed is not None:
        np.random.seed(args.seed)

    # Parse goal
    if args.goal is not None:
        target_values = tuple(float(x.strip()) for x in args.goal.split(","))
    else:
        target_values = (1.0, 1.0)

    dimensions = len(target_values)

    # Create shared event bus
    bus = EventBus()

    # Create environment
    if args.env_type == "numeric":
        env = NumericEnvironment(
            dimensions=dimensions,
            noise_std=args.noise,
            event_bus=bus,
        )
    elif args.env_type == "resource":
        from synthetic_teleology.environments.resource import ResourceEnvironment
        env = ResourceEnvironment(event_bus=bus)
    elif args.env_type == "research":
        from synthetic_teleology.environments.research import ResearchEnvironment
        env = ResearchEnvironment(event_bus=bus)
    else:
        print(f"Error: unknown environment type '{args.env_type}'", file=sys.stderr)
        return 1

    # Create agent
    agent = AgentFactory.create_simple_agent(
        agent_id="cli-agent",
        target_values=target_values,
        event_bus=bus,
    )

    # Set up measurement
    collector = EventCollector(bus)
    engine = MetricsEngine()
    dashboard = ConsoleDashboard()

    # Run the loop
    env.reset()
    print(f"Running {args.steps} steps with {args.agent_type} agent "
          f"in {args.env_type} environment...")
    print(f"  Goal: {target_values}")
    print(f"  Dimensions: {dimensions}")
    print(f"  Noise: {args.noise}")
    print()

    t0 = time.time()
    for step in range(args.steps):
        action = agent.run_cycle(env)
        env.step(action)

    elapsed = time.time() - t0

    # Build report
    log = collector.get_log("cli-agent")
    if log is None:
        print("Warning: no log data collected.", file=sys.stderr)
        return 1

    report = engine.build_report("cli-agent", log)

    # Display
    dashboard.print_report(report)
    dashboard.print_trajectory(log.get_scores())

    print(f"Completed in {elapsed:.2f}s ({args.steps} steps)")

    # Export if output dir specified
    if args.output is not None:
        from synthetic_teleology.presentation.export import export_all
        files = export_all([report], args.output, formats=["json", "csv", "html"])
        print(f"\nExported to {args.output}:")
        for fmt, paths in files.items():
            for p in paths:
                print(f"  [{fmt}] {p}")

    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle the ``benchmark`` subcommand."""
    import time

    import numpy as np

    from synthetic_teleology.agents.factory import AgentFactory
    from synthetic_teleology.environments.numeric import NumericEnvironment
    from synthetic_teleology.infrastructure.event_bus import EventBus
    from synthetic_teleology.measurement.collector import EventCollector
    from synthetic_teleology.measurement.engine import MetricsEngine
    from synthetic_teleology.measurement.report import MetricsReport
    from synthetic_teleology.presentation.console import ConsoleDashboard
    from synthetic_teleology.presentation.export import export_all

    print(f"Running benchmark suite '{args.suite}' with {args.runs} runs, "
          f"{args.steps} steps each...")
    print()

    all_reports: list[MetricsReport] = []
    suite_results: dict[str, list[MetricsReport]] = {}

    # Define benchmark configurations based on suite name
    if args.suite == "default":
        configs = [
            {"name": "2d_noiseless", "dims": 2, "noise": 0.0, "target": (1.0, 1.0)},
            {"name": "2d_noisy", "dims": 2, "noise": 0.05, "target": (1.0, 1.0)},
            {"name": "3d_noiseless", "dims": 3, "noise": 0.0, "target": (1.0, 1.0, 1.0)},
            {"name": "3d_noisy", "dims": 3, "noise": 0.05, "target": (1.0, 1.0, 1.0)},
        ]
    else:
        # Single configuration with default parameters
        configs = [
            {"name": args.suite, "dims": 2, "noise": 0.0, "target": (1.0, 1.0)},
        ]

    t0 = time.time()

    for config in configs:
        config_name: str = config["name"]  # type: ignore[assignment]
        config_reports: list[MetricsReport] = []

        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            np.random.seed(seed)

            bus = EventBus()
            env = NumericEnvironment(
                dimensions=config["dims"],  # type: ignore[arg-type]
                noise_std=config["noise"],  # type: ignore[arg-type]
                event_bus=bus,
            )

            agent = AgentFactory.create_simple_agent(
                agent_id=f"bench-{config_name}-run{run_idx}",
                target_values=config["target"],  # type: ignore[arg-type]
                event_bus=bus,
            )

            collector = EventCollector(bus)
            engine = MetricsEngine()

            env.reset()
            for _ in range(args.steps):
                action = agent.run_cycle(env)
                env.step(action)

            log = collector.get_log(agent.id)
            if log is not None:
                report = engine.build_report(agent.id, log)
                config_reports.append(report)
                all_reports.append(report)

        suite_results[config_name] = config_reports
        print(f"  [{config_name}] {len(config_reports)}/{args.runs} runs completed")

    elapsed = time.time() - t0
    print(f"\nBenchmark completed in {elapsed:.2f}s")
    print()

    # Display summary
    dashboard = ConsoleDashboard()
    dashboard.print_benchmark_results(suite_results)

    # Export results
    files = export_all(all_reports, args.output, formats=["json", "csv"])
    print(f"Exported to {args.output}:")
    for fmt, paths in files.items():
        for p in paths:
            print(f"  [{fmt}] {p}")

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Handle the ``report`` subcommand."""
    from synthetic_teleology.measurement.metrics.base import MetricResult
    from synthetic_teleology.measurement.report import MetricsReport
    from synthetic_teleology.presentation.console import ConsoleDashboard

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error reading {input_path}: {exc}", file=sys.stderr)
        return 1

    # Reconstruct MetricsReport from JSON
    agent_id = data.get("agent_id", "unknown")
    timestamp = data.get("timestamp", 0.0)
    metadata = data.get("metadata", {})
    metrics_dict = data.get("metrics", {})

    results: list[MetricResult] = []
    for name, metric_data in metrics_dict.items():
        results.append(
            MetricResult(
                name=name,
                value=metric_data.get("value", 0.0),
                explanation=metric_data.get("explanation", ""),
                metadata=metric_data.get("metadata", {}),
            )
        )

    report = MetricsReport(
        agent_id=agent_id,
        results=tuple(results),
        timestamp=timestamp,
        metadata=metadata,
    )

    # Display based on requested format
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, default=str))
    elif args.format == "summary":
        print(report.summary())
    else:
        # table (default)
        dashboard = ConsoleDashboard()
        dashboard.print_report(report)

    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Handle the ``info`` subcommand."""
    from synthetic_teleology import __version__

    print(f"Synthetic Teleology Framework v{__version__}")
    print()

    # Check optional dependencies
    optional_deps = {
        "rich": "Console dashboard with colour formatting",
        "matplotlib": "Plotting and visualisation",
        "anthropic": "Anthropic LLM integration",
        "openai": "OpenAI LLM integration",
        "transformers": "HuggingFace LLM integration",
        "httpx": "Generic LLM HTTP client",
        "numpy": "Numerical computation (required)",
    }

    print("Dependencies:")
    for pkg, desc in optional_deps.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [installed] {pkg} {version} -- {desc}")
        except ImportError:
            print(f"  [missing]   {pkg} -- {desc}")

    print()

    # Show registered components via the global registry
    try:
        from synthetic_teleology.infrastructure.registry import registry

        categories = registry.list_categories()
        if categories:
            print("Registered Components:")
            for category in sorted(categories):
                names = registry.list_category(category)
                print(f"  {category}: {', '.join(sorted(names)) if names else '(empty)'}")
        else:
            print("Registered Components: (none)")
    except ImportError:
        print("Component registry not available.")

    print()

    # Show available agent types
    print("Built-in Agent Types:")
    print("  simple -- Minimal teleological agent (AgentFactory.create_simple_agent)")
    print("  teleological -- Full teleological agent with strategy injection")
    print("  constrained -- Teleological agent with explicit constraints")
    print()

    # Show available environments
    print("Built-in Environments:")
    print("  numeric -- N-dimensional continuous state space")
    print("  resource -- Resource allocation environment")
    print("  research -- Research exploration environment")
    print("  shared -- Multi-agent shared environment")
    print()

    # Show available metrics
    print("Canonical Metrics (7):")
    try:
        from synthetic_teleology.measurement.engine import MetricsEngine
        engine = MetricsEngine()
        for name in engine.metric_names:
            print(f"  - {name}")
    except ImportError:
        print("  (metrics engine not available)")

    return 0


# =========================================================================
# Main entry point
# =========================================================================

def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Handle --version at top level
    if args.version:
        from synthetic_teleology import __version__
        print(f"synthetic-teleology {__version__}")
        sys.exit(0)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers: dict[str, Any] = {
        "run": _cmd_run,
        "benchmark": _cmd_benchmark,
        "report": _cmd_report,
        "info": _cmd_info,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        exit_code = handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)
