"""Base benchmark abstraction for the Synthetic Teleology measurement layer.

:class:`BaseBenchmark` defines the **Template Method** pattern used by all
concrete benchmark scenarios:

1. ``setup()``           -- prepare environment, agents, and infrastructure.
2. ``run_scenario(seed)`` -- execute a single scenario run with a given seed.
3. ``collect_metrics(log)`` -- compute metrics from the resulting agent log.
4. ``teardown()``        -- clean up resources.
5. ``run(num_runs, base_seed)`` -- orchestrate the full benchmark.

Subclasses must implement ``setup``, ``run_scenario``, ``collect_metrics``,
and ``teardown``.  The ``run`` method is the public entry point and should
**not** be overridden.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from synthetic_teleology.measurement.collector import AgentLog
from synthetic_teleology.measurement.report import MetricsReport

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Template Method base class for all teleological benchmarks.

    Subclasses implement the four hook methods that define a specific
    benchmark scenario.  The invariant orchestration lives in :meth:`run`.

    Typical usage::

        class MyBenchmark(BaseBenchmark):
            def setup(self) -> None: ...
            def run_scenario(self, seed: int) -> AgentLog: ...
            def collect_metrics(self, log: AgentLog) -> MetricsReport: ...
            def teardown(self) -> None: ...

        benchmark = MyBenchmark()
        reports = benchmark.run(num_runs=10, base_seed=42)
    """

    # -- abstract hooks (subclass responsibility) ----------------------------

    @abstractmethod
    def setup(self) -> None:
        """Prepare environment, agents, metrics engine, and any other
        infrastructure required by the benchmark.

        Called once at the beginning of :meth:`run`, before the first scenario.
        """

    @abstractmethod
    def run_scenario(self, seed: int) -> AgentLog:
        """Execute a single benchmark scenario with a deterministic *seed*.

        Parameters
        ----------
        seed:
            Random seed for reproducibility.  Implementations should use
            this to seed numpy, random, and any stochastic components.

        Returns
        -------
        AgentLog
            The complete agent log produced by the scenario run.
        """

    @abstractmethod
    def collect_metrics(self, log: AgentLog) -> MetricsReport:
        """Compute all relevant metrics from the agent log.

        Parameters
        ----------
        log:
            The agent log produced by :meth:`run_scenario`.

        Returns
        -------
        MetricsReport
            An immutable report containing all metric results.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Release resources, reset state, and clean up after all runs.

        Called once at the end of :meth:`run`, after the last scenario.
        """

    # -- template method (public API) ----------------------------------------

    def run(
        self,
        num_runs: int = 10,
        base_seed: int = 42,
    ) -> list[MetricsReport]:
        """Execute the benchmark suite using the template method pattern.

        Orchestrates: ``setup`` -> (for each run: ``run_scenario`` ->
        ``collect_metrics``) -> ``teardown``.

        Parameters
        ----------
        num_runs:
            Number of independent scenario runs. Each uses a unique seed
            derived from *base_seed*.
        base_seed:
            Starting seed.  Run *i* uses ``base_seed + i`` as its seed.

        Returns
        -------
        list[MetricsReport]
            One report per run, in order.
        """
        logger.info(
            "%s: starting benchmark with %d runs (base_seed=%d)",
            type(self).__name__,
            num_runs,
            base_seed,
        )

        t0 = time.monotonic()
        reports: list[MetricsReport] = []

        self.setup()

        try:
            for i in range(num_runs):
                seed = base_seed + i
                logger.info(
                    "%s: run %d/%d (seed=%d)",
                    type(self).__name__,
                    i + 1,
                    num_runs,
                    seed,
                )

                try:
                    log = self.run_scenario(seed)
                    report = self.collect_metrics(log)
                    reports.append(report)
                    logger.debug(
                        "%s: run %d completed -- %d metrics",
                        type(self).__name__,
                        i + 1,
                        len(report.results),
                    )
                except Exception:
                    logger.exception(
                        "%s: run %d (seed=%d) failed",
                        type(self).__name__,
                        i + 1,
                        seed,
                    )
        finally:
            self.teardown()

        elapsed = time.monotonic() - t0
        logger.info(
            "%s: benchmark finished -- %d/%d successful runs in %.2fs",
            type(self).__name__,
            len(reports),
            num_runs,
            elapsed,
        )

        return reports

    # -- dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
