"""Benchmark infrastructure for the Synthetic Teleology measurement layer.

This package provides scenario-based benchmarks for evaluating agent
performance under controlled conditions:

- :class:`BaseBenchmark` -- Template Method ABC for all benchmarks.
- :class:`DistributionShiftBenchmark` -- Tests adaptivity under distribution shift.
- :class:`ConflictingObjectivesBenchmark` -- Tests behaviour with conflicting objectives.
- :class:`NegotiationBenchmark` -- Tests multi-agent negotiation convergence.
- :class:`KnowledgeSynthesisBenchmark` -- Tests knowledge accumulation and synthesis.
- :class:`BenchmarkSuite` -- Composite runner for multiple benchmarks.
- :class:`CoordinationMediator` -- Consensus mediator for multi-agent scenarios.
"""

from synthetic_teleology.measurement.benchmarks.base import BaseBenchmark
from synthetic_teleology.measurement.benchmarks.conflicting_obj import (
    ConflictingObjectivesBenchmark,
)
from synthetic_teleology.measurement.benchmarks.distribution_shift import (
    DistributionShiftBenchmark,
)
from synthetic_teleology.measurement.benchmarks.knowledge_synthesis import (
    KnowledgeSynthesisBenchmark,
    ResearchPlanner,
)
from synthetic_teleology.measurement.benchmarks.negotiation import (
    CoordinationMediator,
    NegotiationBenchmark,
    NegotiationRound,
)
from synthetic_teleology.measurement.benchmarks.suite import BenchmarkSuite

__all__ = [
    "BaseBenchmark",
    "BenchmarkSuite",
    "ConflictingObjectivesBenchmark",
    "CoordinationMediator",
    "DistributionShiftBenchmark",
    "KnowledgeSynthesisBenchmark",
    "NegotiationBenchmark",
    "NegotiationRound",
    "ResearchPlanner",
]
