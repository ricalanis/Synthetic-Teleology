"""Tests for benchmark infrastructure."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.measurement.benchmarks.base import BaseBenchmark
from synthetic_teleology.measurement.benchmarks.suite import BenchmarkSuite
from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry
from synthetic_teleology.measurement.engine import MetricsEngine
from synthetic_teleology.measurement.metrics.base import MetricResult
from synthetic_teleology.measurement.report import MetricsReport


class _SimpleBenchmark(BaseBenchmark):
    """A trivial benchmark for testing the template-method flow."""

    def __init__(self, fail_on_run: bool = False) -> None:
        self._setup_called = False
        self._teardown_called = False
        self._scenarios_run: list[int] = []
        self._fail_on_run = fail_on_run

    def setup(self) -> None:
        self._setup_called = True

    def run_scenario(self, seed: int) -> AgentLog:
        if self._fail_on_run:
            raise RuntimeError("scenario failed")
        self._scenarios_run.append(seed)
        log = AgentLog(agent_id="bench-agent")
        for i in range(5):
            log.entries.append(
                AgentLogEntry(
                    step=i,
                    timestamp=float(i),
                    eval_score=0.5 + i * 0.05,
                    action_name=f"action_{i % 3}",
                )
            )
        return log

    def collect_metrics(self, log: AgentLog) -> MetricsReport:
        engine = MetricsEngine()
        return engine.build_report("bench-agent", log)

    def teardown(self) -> None:
        self._teardown_called = True


class TestBaseBenchmark:
    """Test the benchmark template method pattern."""

    def test_run_calls_setup_and_teardown(self) -> None:
        bench = _SimpleBenchmark()
        reports = bench.run(num_runs=3, base_seed=0)
        assert bench._setup_called is True
        assert bench._teardown_called is True

    def test_run_produces_correct_number_of_reports(self) -> None:
        bench = _SimpleBenchmark()
        reports = bench.run(num_runs=5, base_seed=10)
        assert len(reports) == 5

    def test_run_passes_sequential_seeds(self) -> None:
        bench = _SimpleBenchmark()
        bench.run(num_runs=3, base_seed=100)
        assert bench._scenarios_run == [100, 101, 102]

    def test_reports_contain_metrics(self) -> None:
        bench = _SimpleBenchmark()
        reports = bench.run(num_runs=1, base_seed=0)
        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, MetricsReport)
        assert len(report.results) > 0

    def test_failed_scenario_does_not_crash_suite(self) -> None:
        bench = _SimpleBenchmark(fail_on_run=True)
        reports = bench.run(num_runs=3, base_seed=0)
        # All scenarios fail, so no reports collected
        assert len(reports) == 0
        assert bench._teardown_called is True

    def test_repr(self) -> None:
        bench = _SimpleBenchmark()
        assert "_SimpleBenchmark" in repr(bench)


class TestBenchmarkSuite:
    """Test BenchmarkSuite composite runner."""

    def test_add_and_run_benchmarks(self) -> None:
        suite = BenchmarkSuite(name="test-suite")
        suite.add_benchmark("simple", _SimpleBenchmark())
        results = suite.run_all(num_runs=2, base_seed=0)
        assert "simple" in results
        assert len(results["simple"]) == 2

    def test_add_duplicate_raises(self) -> None:
        suite = BenchmarkSuite()
        suite.add_benchmark("a", _SimpleBenchmark())
        with pytest.raises(ValueError, match="already registered"):
            suite.add_benchmark("a", _SimpleBenchmark())

    def test_remove_benchmark(self) -> None:
        suite = BenchmarkSuite()
        suite.add_benchmark("a", _SimpleBenchmark())
        assert suite.remove_benchmark("a") is True
        assert suite.remove_benchmark("a") is False
        assert suite.num_benchmarks == 0

    def test_benchmark_names(self) -> None:
        suite = BenchmarkSuite()
        suite.add_benchmark("b1", _SimpleBenchmark())
        suite.add_benchmark("b2", _SimpleBenchmark())
        assert suite.benchmark_names == ["b1", "b2"]

    def test_get_benchmark(self) -> None:
        suite = BenchmarkSuite()
        bench = _SimpleBenchmark()
        suite.add_benchmark("test", bench)
        assert suite.get_benchmark("test") is bench
        assert suite.get_benchmark("missing") is None

    def test_run_benchmark_by_name(self) -> None:
        suite = BenchmarkSuite()
        suite.add_benchmark("test", _SimpleBenchmark())
        reports = suite.run_benchmark("test", num_runs=2)
        assert len(reports) == 2

    def test_run_benchmark_not_found_raises(self) -> None:
        suite = BenchmarkSuite()
        with pytest.raises(KeyError, match="not found"):
            suite.run_benchmark("missing")

    def test_summary_no_results(self) -> None:
        suite = BenchmarkSuite(name="empty")
        summary = suite.summary()
        assert "No results" in summary

    def test_summary_with_results(self) -> None:
        suite = BenchmarkSuite(name="test-suite")
        suite.add_benchmark("simple", _SimpleBenchmark())
        suite.run_all(num_runs=2)
        summary = suite.summary()
        assert "test-suite" in summary
        assert "simple" in summary

    def test_contains(self) -> None:
        suite = BenchmarkSuite()
        suite.add_benchmark("a", _SimpleBenchmark())
        assert "a" in suite
        assert "b" not in suite

    def test_len(self) -> None:
        suite = BenchmarkSuite()
        assert len(suite) == 0
        suite.add_benchmark("a", _SimpleBenchmark())
        assert len(suite) == 1

    def test_repr(self) -> None:
        suite = BenchmarkSuite(name="my-suite")
        r = repr(suite)
        assert "BenchmarkSuite" in r
        assert "my-suite" in r
