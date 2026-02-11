"""Service layer for the Synthetic Teleology framework.

Re-exports public service types for convenient top-level access::

    from synthetic_teleology.services import (
        BaseEvaluator, NumericEvaluator, CompositeEvaluator,
        ReflectiveEvaluator, LLMCriticEvaluator,
        BaseGoalUpdater, ThresholdUpdater, GradientUpdater,
        GoalUpdaterChain, HierarchicalUpdater,
        UncertaintyAwareUpdater, ConstrainedUpdater, LLMGoalEditor,
        BasePlanner, GreedyPlanner, StochasticPlanner,
        HierarchicalPlanner, LLMPlanner,
        BaseConstraintChecker, SafetyChecker, BudgetChecker,
        EthicalChecker, ConstraintPipeline, PolicyFilter,
        BaseAgenticLoop, SyncAgenticLoop, AsyncAgenticLoop, RunResult,
        BaseNegotiator, ConsensusNegotiator, VotingNegotiator,
        AuctionNegotiator, CoordinationMediator,
    )
"""

from synthetic_teleology.services.constraint_engine import (
    BaseConstraintChecker,
    BudgetChecker,
    ConstraintPipeline,
    EthicalChecker,
    PolicyFilter,
    SafetyChecker,
)
from synthetic_teleology.services.coordination import (
    AuctionNegotiator,
    BaseNegotiator,
    ConsensusNegotiator,
    CoordinationMediator,
    VotingNegotiator,
)
from synthetic_teleology.services.evaluation import (
    BaseEvaluator,
    CompositeEvaluator,
    LLMCriticEvaluator,
    NumericEvaluator,
    ReflectiveEvaluator,
)
from synthetic_teleology.services.goal_revision import (
    BaseGoalUpdater,
    ConstrainedUpdater,
    GoalUpdaterChain,
    GradientUpdater,
    HierarchicalUpdater,
    LLMGoalEditor,
    ThresholdUpdater,
    UncertaintyAwareUpdater,
)
from synthetic_teleology.services.llm_constraints import LLMConstraintChecker
from synthetic_teleology.services.llm_evaluation import LLMEvaluator
from synthetic_teleology.services.llm_planning import LLMPlanner as LLMHypothesisPlanner
from synthetic_teleology.services.llm_revision import LLMReviser
from synthetic_teleology.services.loop import (
    AsyncAgenticLoop,
    BaseAgenticLoop,
    RunResult,
    SyncAgenticLoop,
)
from synthetic_teleology.services.planning import (
    BasePlanner,
    GreedyPlanner,
    HierarchicalPlanner,
    LLMPlanner,
    StochasticPlanner,
)

__all__ = [
    # evaluation
    "BaseEvaluator",
    "NumericEvaluator",
    "CompositeEvaluator",
    "ReflectiveEvaluator",
    "LLMCriticEvaluator",
    # goal revision
    "BaseGoalUpdater",
    "ThresholdUpdater",
    "GradientUpdater",
    "GoalUpdaterChain",
    "HierarchicalUpdater",
    "UncertaintyAwareUpdater",
    "ConstrainedUpdater",
    "LLMGoalEditor",
    # planning
    "BasePlanner",
    "GreedyPlanner",
    "StochasticPlanner",
    "HierarchicalPlanner",
    "LLMPlanner",
    # constraint engine
    "BaseConstraintChecker",
    "SafetyChecker",
    "BudgetChecker",
    "EthicalChecker",
    "ConstraintPipeline",
    "PolicyFilter",
    # coordination
    "BaseNegotiator",
    "ConsensusNegotiator",
    "VotingNegotiator",
    "AuctionNegotiator",
    "CoordinationMediator",
    # agentic loop
    "BaseAgenticLoop",
    "SyncAgenticLoop",
    "AsyncAgenticLoop",
    "RunResult",
    # LLM-backed services (v1.0)
    "LLMEvaluator",
    "LLMHypothesisPlanner",
    "LLMReviser",
    "LLMConstraintChecker",
]
