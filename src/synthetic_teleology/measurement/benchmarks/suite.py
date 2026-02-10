"""Benchmark suite -- Composite pattern for running multiple benchmarks.

:class:`BenchmarkSuite` aggregates named :class:`BaseBenchmark` instances
and provides bulk-run and summary functionality.

Usage::

    suite = BenchmarkSuite()
    suite.add_benchmark("distribution_shift", DistributionShiftBenchmark())
    suite.add_benchmark("conflicting_obj", ConflictingObjectivesBenchmark())
    results = suite.run_all(num_runs=10)
    print(suite.summary())
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

from synthetic_teleology.measurement.benchmarks.base import BaseBenchmark
from synthetic_teleology.measurement.report import MetricsReport

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Composite benchmark runner -- aggregates and runs multiple benchmarks.

    Benchmarks are stored in insertion order and can be addressed by name.

    Parameters
    ----------
    name:
        Optional descriptive name for the suite.
    """

    def __init__(self, name: str = "Teleological Benchmark Suite") -> None:
        self._name = name
        self._benchmarks: OrderedDict[str, BaseBenchmark] = OrderedDict()
        self._results: dict[str, list[MetricsReport]] = {}
        self._run_metadata: dict[str, Any] = {}

    # -- mutation -------------------------------------------------------------

    def add_benchmark(self, name: str, benchmark: BaseBenchmark) -> None:
        """Register a benchmark under the given *name*.

        Parameters
        ----------
        name:
            Unique identifier for this benchmark within the suite.
        benchmark:
            The benchmark instance to register.

        Raises
        ------
        ValueError
            If a benchmark with the same name is already registered.
        """
        if name in self._benchmarks:
            raise ValueError(
                f"Benchmark with name {name!r} is already registered. "
                f"Use remove_benchmark() first if you want to replace it."
            )
        self._benchmarks[name] = benchmark

    def remove_benchmark(self, name: str) -> bool:
        """Remove the benchmark with the given *name*.

        Parameters
        ----------
        name:
            The benchmark name to remove.

        Returns
        -------
        bool
            ``True`` if a benchmark was removed, ``False`` if not found.
        """
        if name in self._benchmarks:
            del self._benchmarks[name]
            self._results.pop(name, None)
            return True
        return False

    # -- query ----------------------------------------------------------------

    @property
    def name(self) -> str:
        """Suite name."""
        return self._name

    @property
    def benchmark_names(self) -> list[str]:
        """Ordered list of registered benchmark names."""
        return list(self._benchmarks.keys())

    @property
    def num_benchmarks(self) -> int:
        """Number of registered benchmarks."""
        return len(self._benchmarks)

    def get_benchmark(self, name: str) -> BaseBenchmark | None:
        """Return the benchmark with the given *name*, or ``None``."""
        return self._benchmarks.get(name)

    @property
    def results(self) -> dict[str, list[MetricsReport]]:
        """Return a shallow copy of the most recent results."""
        return dict(self._results)

    # -- execution ------------------------------------------------------------

    def run_all(
        self,
        num_runs: int = 10,
        base_seed: int = 42,
    ) -> dict[str, list[MetricsReport]]:
        """Run all registered benchmarks and return results.

        Parameters
        ----------
        num_runs:
            Number of runs per benchmark.
        base_seed:
            Starting seed for reproducibility.

        Returns
        -------
        dict[str, list[MetricsReport]]
            Mapping from benchmark name to its list of reports.
        """
        logger.info(
            "%s: running %d benchmarks with %d runs each (base_seed=%d)",
            self._name,
            len(self._benchmarks),
            num_runs,
            base_seed,
        )

        t0 = time.monotonic()
        self._results.clear()

        for bench_name, benchmark in self._benchmarks.items():
            logger.info(
                "%s: starting benchmark %r (%s)",
                self._name,
                bench_name,
                type(benchmark).__name__,
            )

            try:
                reports = benchmark.run(num_runs=num_runs, base_seed=base_seed)
                self._results[bench_name] = reports
                logger.info(
                    "%s: benchmark %r completed -- %d/%d successful runs",
                    self._name,
                    bench_name,
                    len(reports),
                    num_runs,
                )
            except Exception:
                logger.exception(
                    "%s: benchmark %r failed entirely",
                    self._name,
                    bench_name,
                )
                self._results[bench_name] = []

        elapsed = time.monotonic() - t0
        self._run_metadata = {
            "num_runs": num_runs,
            "base_seed": base_seed,
            "elapsed_seconds": elapsed,
            "benchmarks_completed": sum(
                1 for r in self._results.values() if len(r) > 0
            ),
            "total_reports": sum(len(r) for r in self._results.values()),
        }

        logger.info(
            "%s: suite finished in %.2fs -- %d total reports",
            self._name,
            elapsed,
            self._run_metadata["total_reports"],
        )

        return dict(self._results)

    def run_benchmark(
        self,
        name: str,
        num_runs: int = 10,
        base_seed: int = 42,
    ) -> list[MetricsReport]:
        """Run a single benchmark by name.

        Parameters
        ----------
        name:
            Name of the benchmark to run.
        num_runs:
            Number of runs.
        base_seed:
            Starting seed.

        Returns
        -------
        list[MetricsReport]
            Reports from the benchmark.

        Raises
        ------
        KeyError
            If no benchmark with the given name is registered.
        """
        if name not in self._benchmarks:
            raise KeyError(f"Benchmark {name!r} not found in suite")

        reports = self._benchmarks[name].run(num_runs=num_runs, base_seed=base_seed)
        self._results[name] = reports
        return reports

    # -- summary generation ---------------------------------------------------

    def summary(self) -> str:
        """Generate a formatted summary table of all benchmark results.

        Returns
        -------
        str
            A human-readable table string.  Returns a message indicating
            no results if :meth:`run_all` has not been called yet.
        """
        if not self._results:
            return f"=== {self._name} ===\nNo results available. Run the suite first."

        lines: list[str] = []
        lines.append(f"=== {self._name} ===")
        lines.append("")

        if self._run_metadata:
            lines.append(
                f"Runs per benchmark: {self._run_metadata.get('num_runs', '?')} | "
                f"Total reports: {self._run_metadata.get('total_reports', '?')} | "
                f"Elapsed: {self._run_metadata.get('elapsed_seconds', 0):.2f}s"
            )
            lines.append("")

        for bench_name, reports in self._results.items():
            lines.append(f"--- {bench_name} ({len(reports)} runs) ---")

            if not reports:
                lines.append("  (no successful runs)")
                lines.append("")
                continue

            # Aggregate metrics across runs
            metric_values: dict[str, list[float]] = {}
            for report in reports:
                for result in report.results:
                    metric_values.setdefault(result.name, []).append(result.value)

            # Compute statistics
            if metric_values:
                # Header
                header = f"  {'Metric':<30s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}"
                lines.append(header)
                lines.append("  " + "-" * (len(header) - 2))

                for metric_name, values in sorted(metric_values.items()):
                    import numpy as np

                    arr = np.array(values)
                    mean_val = float(arr.mean())
                    std_val = float(arr.std())
                    min_val = float(arr.min())
                    max_val = float(arr.max())

                    lines.append(
                        f"  {metric_name:<30s} "
                        f"{mean_val:>8.4f} "
                        f"{std_val:>8.4f} "
                        f"{min_val:>8.4f} "
                        f"{max_val:>8.4f}"
                    )

            lines.append("")

        # Footer separator
        lines.append("=" * 60)

        return "\n".join(lines)

    # -- dunder helpers -------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BenchmarkSuite("
            f"name={self._name!r}, "
            f"benchmarks={len(self._benchmarks)}, "
            f"has_results={bool(self._results)})"
        )

    def __len__(self) -> int:
        return len(self._benchmarks)

    def __contains__(self, name: str) -> bool:
        return name in self._benchmarks
