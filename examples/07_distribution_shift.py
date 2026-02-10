#!/usr/bin/env python3
"""Example 07: Distribution shift benchmark.

Demonstrates:
- Configuring and running the DistributionShiftBenchmark
- Executing multiple runs with different seeds
- Collecting and comparing adaptivity metrics (GP, TC, AD, LS)
- Inspecting per-run MetricsReport results

The benchmark creates a numeric environment, runs an agent for N steps,
injects a perturbation (state shift + noise increase), then runs N more
steps.  Metrics are collected to assess how well the agent adapts.

Run:
    PYTHONPATH=src python examples/07_distribution_shift.py
"""

from __future__ import annotations

import numpy as np

from synthetic_teleology.measurement.benchmarks.distribution_shift import (
    DistributionShiftBenchmark,
)


def main() -> None:
    print("=== Distribution Shift Benchmark ===\n")

    # -- Configure benchmark --------------------------------------------------
    benchmark = DistributionShiftBenchmark(
        dimensions=3,
        steps_per_phase=15,
        target_values=(5.0, 5.0, 5.0),
        perturbation_magnitude=4.0,
        noise_std=0.05,
        post_shift_noise_std=0.2,
        step_size=0.5,
    )

    print(f"Benchmark: {benchmark}")
    print(f"  Dimensions:          3")
    print(f"  Steps per phase:     15 (30 total)")
    print(f"  Perturbation:        magnitude=4.0")
    print(f"  Noise (pre/post):    0.05 / 0.20")
    print()

    # -- Run benchmark --------------------------------------------------------
    num_runs = 3
    base_seed = 42
    print(f"Running {num_runs} scenarios (base_seed={base_seed})...\n")

    reports = benchmark.run(num_runs=num_runs, base_seed=base_seed)

    print(f"Completed {len(reports)}/{num_runs} runs.\n")

    # -- Display per-run results ----------------------------------------------
    for i, report in enumerate(reports):
        seed = base_seed + i
        print(f"--- Run {i + 1} (seed={seed}) ---")
        print(f"  Agent: {report.agent_id}")
        print(f"  Metrics computed: {len(report.results)}")

        for result in report.results:
            print(f"    {result.name:<30s}  value={result.value:.4f}")
        print()

    # -- Aggregate statistics across runs -------------------------------------
    if len(reports) >= 2:
        print("=== Aggregate Statistics ===\n")

        # Collect metric names from the first report
        metric_names = [r.name for r in reports[0].results]

        for metric_name in metric_names:
            values = []
            for report in reports:
                result = report.get_metric(metric_name)
                if result is not None:
                    values.append(result.value)

            if values:
                arr = np.array(values)
                print(
                    f"  {metric_name:<30s}  "
                    f"mean={arr.mean():.4f}  "
                    f"std={arr.std():.4f}  "
                    f"min={arr.min():.4f}  "
                    f"max={arr.max():.4f}"
                )

    print()

    # -- Single run with detailed inspection ----------------------------------
    print("=== Detailed Single Run ===\n")

    benchmark_detail = DistributionShiftBenchmark(
        dimensions=2,
        steps_per_phase=10,
        target_values=(3.0, 7.0),
        perturbation_magnitude=3.0,
        noise_std=0.0,
        step_size=0.5,
    )
    benchmark_detail.setup()

    log = benchmark_detail.run_scenario(seed=99)
    print(f"Agent log: {log.agent_id}")
    print(f"  Total entries: {log.num_steps}")
    print(f"  Goal revisions: {log.revision_count}")

    report = benchmark_detail.collect_metrics(log)
    print(f"\nMetrics report:")
    print(report.summary())

    benchmark_detail.teardown()

    print("\nDone.")


if __name__ == "__main__":
    main()
