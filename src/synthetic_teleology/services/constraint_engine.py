"""Constraint checking engine for the Synthetic Teleology framework.

Implements Chain of Responsibility for constraint validation: each checker
inspects a specific constraint type and reports pass/fail with a message.

Classes
-------
BaseConstraintChecker
    Abstract base class for constraint checkers.
SafetyChecker
    Checks safety constraints (state values within safe bounds).
BudgetChecker
    Checks cost budget constraints.
EthicalChecker
    Checks ethical constraint specifications.
ConstraintPipeline
    Chain of Responsibility aggregating multiple checkers.
PolicyFilter
    Filters a PolicySpec's actions against the constraint pipeline.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import ConstraintType
from synthetic_teleology.domain.values import (
    ActionSpec,
    ConstraintResult,
    PolicySpec,
    StateSnapshot,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Constraint Check Result                                               #
# ===================================================================== #


@dataclass(frozen=True)
class ConstraintViolation:
    """Record of a single constraint violation."""

    checker_name: str
    constraint_name: str
    constraint_type: ConstraintType
    message: str
    severity: float = 1.0  # 0=negligible, 1=critical
    metadata: dict[str, Any] = field(default_factory=dict)


# ===================================================================== #
#  Base Constraint Checker (ABC)                                         #
# ===================================================================== #


class BaseConstraintChecker(ABC):
    """Abstract base class for constraint checkers.

    Each checker inspects the goal, state, and optional action to determine
    whether a specific type of constraint is satisfied.
    """

    @abstractmethod
    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        """Check a constraint condition.

        Parameters
        ----------
        goal:
            The current goal entity.
        state:
            The current state snapshot.
        action:
            Optional action being considered (for pre-action checks).

        Returns
        -------
        tuple[bool, str]
            ``(passed, violation_message)``. If ``passed`` is ``True``,
            the message is the empty string.
        """

    def check_detailed(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> ConstraintResult:
        """Check a constraint and return a detailed ``ConstraintResult``.

        Default implementation wraps ``check()`` for backward compatibility.
        Subclasses may override for richer results (e.g. severity, mitigation).
        """
        passed, message = self.check(goal, state, action)
        return ConstraintResult(
            passed=passed,
            message=message,
            checker_name=getattr(self, "_name", type(self).__name__),
        )


# ===================================================================== #
#  Safety Checker                                                        #
# ===================================================================== #


class SafetyChecker(BaseConstraintChecker):
    """Check that state values fall within specified safe bounds.

    Parameters
    ----------
    lower_bounds:
        Per-dimension lower bounds. ``None`` for any dimension means no
        lower bound.
    upper_bounds:
        Per-dimension upper bounds. ``None`` for any dimension means no
        upper bound.
    name:
        Human-readable name for this checker. Defaults to ``"SafetyChecker"``.
    """

    def __init__(
        self,
        lower_bounds: Sequence[float | None] | None = None,
        upper_bounds: Sequence[float | None] | None = None,
        name: str = "SafetyChecker",
    ) -> None:
        self._lower = list(lower_bounds) if lower_bounds else []
        self._upper = list(upper_bounds) if upper_bounds else []
        self._name = name

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        """Check state values against safe bounds."""
        violations: list[str] = []

        for i, val in enumerate(state.values):
            if i < len(self._lower) and self._lower[i] is not None:
                lb = self._lower[i]
                assert lb is not None  # for type checker
                if val < lb:
                    violations.append(
                        f"dim[{i}]={val:.4f} < lower_bound={lb:.4f}"
                    )
            if i < len(self._upper) and self._upper[i] is not None:
                ub = self._upper[i]
                assert ub is not None
                if val > ub:
                    violations.append(
                        f"dim[{i}]={val:.4f} > upper_bound={ub:.4f}"
                    )

        # Also check action effects if action is provided
        if action is not None:
            effect = action.parameters.get("effect")
            if effect is not None and isinstance(effect, (list, tuple)):
                projected = [
                    s + e
                    for s, e in zip(state.values, effect)
                    if isinstance(e, (int, float))
                ]
                for i, val in enumerate(projected):
                    if i < len(self._lower) and self._lower[i] is not None:
                        lb = self._lower[i]
                        assert lb is not None
                        if val < lb:
                            violations.append(
                                f"projected_dim[{i}]={val:.4f} < lower_bound={lb:.4f}"
                            )
                    if i < len(self._upper) and self._upper[i] is not None:
                        ub = self._upper[i]
                        assert ub is not None
                        if val > ub:
                            violations.append(
                                f"projected_dim[{i}]={val:.4f} > upper_bound={ub:.4f}"
                            )

        if violations:
            msg = f"{self._name}: " + "; ".join(violations)
            return False, msg
        return True, ""


# ===================================================================== #
#  Budget Checker                                                        #
# ===================================================================== #


class BudgetChecker(BaseConstraintChecker):
    """Check that cumulative action cost stays within budget.

    Tracks accumulated cost across calls and rejects actions that would
    exceed the budget.

    Parameters
    ----------
    total_budget:
        Maximum cumulative cost allowed.
    name:
        Human-readable name. Defaults to ``"BudgetChecker"``.
    """

    def __init__(
        self,
        total_budget: float,
        name: str = "BudgetChecker",
    ) -> None:
        if total_budget < 0:
            raise ValueError(f"total_budget must be non-negative, got {total_budget}")
        self._total_budget = total_budget
        self._spent: float = 0.0
        self._name = name
        self._lock = threading.Lock()

    @property
    def budget_remaining(self) -> float:
        """Return the remaining budget."""
        return max(0.0, self._total_budget - self._spent)

    @property
    def total_spent(self) -> float:
        """Return the total spent so far."""
        return self._spent

    def record_cost(self, cost: float) -> None:
        """Record a cost expenditure (called after action execution)."""
        with self._lock:
            self._spent += cost

    def reset(self) -> None:
        """Reset the budget tracker.

        IMPORTANT: Call ``reset()`` between runs when reusing a BudgetChecker
        instance across multiple graph invocations, otherwise the accumulated
        cost from previous runs carries over.
        """
        with self._lock:
            self._spent = 0.0

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        """Check whether the action cost would exceed the remaining budget."""
        if action is None:
            # No action to check, just verify budget is not exhausted
            if self._spent >= self._total_budget:
                return False, (
                    f"{self._name}: budget exhausted "
                    f"(spent={self._spent:.4f}, budget={self._total_budget:.4f})"
                )
            return True, ""

        projected_cost = self._spent + action.cost
        if projected_cost > self._total_budget:
            return False, (
                f"{self._name}: action '{action.name}' cost={action.cost:.4f} "
                f"would exceed budget (spent={self._spent:.4f}, "
                f"remaining={self.budget_remaining:.4f}, "
                f"budget={self._total_budget:.4f})"
            )
        return True, ""


# ===================================================================== #
#  Ethical Checker                                                       #
# ===================================================================== #


class EthicalChecker(BaseConstraintChecker):
    """Check ethical constraint specifications.

    Validates actions and states against a set of ethical rules, each
    expressed as a callable predicate or a declarative rule specification.

    Parameters
    ----------
    rules:
        Sequence of ``(rule_name, predicate)`` pairs. Each predicate takes
        ``(goal, state, action)`` and returns ``True`` if the rule is
        satisfied.
    name:
        Human-readable name. Defaults to ``"EthicalChecker"``.
    """

    RulePredicate = type(lambda g, s, a: True)  # for documentation only

    def __init__(
        self,
        rules: Sequence[
            tuple[str, Any]
        ],
        name: str = "EthicalChecker",
    ) -> None:
        self._rules = list(rules)
        self._name = name

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        """Evaluate all ethical rules and report violations."""
        violations: list[str] = []

        for rule_name, predicate in self._rules:
            try:
                satisfied = predicate(goal, state, action)
                if not satisfied:
                    violations.append(f"rule '{rule_name}' violated")
            except Exception as exc:
                violations.append(
                    f"rule '{rule_name}' evaluation error: {exc}"
                )
                logger.warning(
                    "%s: rule '%s' raised %s", self._name, rule_name, exc
                )

        if violations:
            msg = f"{self._name}: " + "; ".join(violations)
            return False, msg
        return True, ""


# ===================================================================== #
#  Constraint Pipeline (Chain of Responsibility)                         #
# ===================================================================== #


class ConstraintPipeline:
    """Chain of Responsibility -- run checkers in order and collect violations.

    Parameters
    ----------
    checkers:
        Ordered sequence of constraint checkers.
    fail_fast:
        If ``True``, stop on the first hard constraint violation.
        Defaults to ``False`` (run all checkers and collect all violations).
    hard_checkers:
        Set of checker indices (0-based) that are considered hard constraints.
        A hard constraint violation causes immediate rejection when
        ``fail_fast`` is ``True``.  If not provided, all checkers are
        treated as hard constraints.
    """

    def __init__(
        self,
        checkers: Sequence[BaseConstraintChecker],
        fail_fast: bool = False,
        hard_checker_indices: Sequence[int] | None = None,
    ) -> None:
        self._checkers = list(checkers)
        self._fail_fast = fail_fast
        self._hard_indices: set[int] = (
            set(hard_checker_indices)
            if hard_checker_indices is not None
            else set(range(len(self._checkers)))
        )

    @property
    def checkers(self) -> list[BaseConstraintChecker]:
        """Return the list of checkers."""
        return list(self._checkers)

    def check_all(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, list[str]]:
        """Run all checkers and collect violation messages.

        Parameters
        ----------
        goal:
            The current goal.
        state:
            The current state.
        action:
            Optional action to check.

        Returns
        -------
        tuple[bool, list[str]]
            ``(all_passed, violation_messages)``.
        """
        violations: list[str] = []
        all_passed = True

        for idx, checker in enumerate(self._checkers):
            try:
                passed, message = checker.check(goal, state, action)
            except Exception as exc:
                passed = False
                message = f"{type(checker).__name__}: exception: {exc}"
                logger.exception("ConstraintPipeline: checker %d failed", idx)

            if not passed:
                all_passed = False
                violations.append(message)

                if self._fail_fast and idx in self._hard_indices:
                    logger.info(
                        "ConstraintPipeline: fail-fast on hard checker %d: %s",
                        idx,
                        message,
                    )
                    break

        return all_passed, violations

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        """Convenience method matching the BaseConstraintChecker interface.

        Returns a single combined violation message.
        """
        passed, violations = self.check_all(goal, state, action)
        return passed, " | ".join(violations) if violations else ""

    def check_all_detailed(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> list[ConstraintResult]:
        """Run all checkers and return detailed ``ConstraintResult`` objects.

        Unlike ``check_all()`` which returns ``(bool, list[str])``, this
        method provides per-checker ``ConstraintResult`` with severity,
        checker name, and suggested mitigations.
        """
        results: list[ConstraintResult] = []
        for checker in self._checkers:
            try:
                results.append(checker.check_detailed(goal, state, action))
            except Exception as exc:
                results.append(ConstraintResult(
                    passed=False,
                    message=f"{type(checker).__name__}: exception: {exc}",
                    checker_name=getattr(checker, "_name", type(checker).__name__),
                    severity=1.0,
                ))
                logger.exception(
                    "ConstraintPipeline: check_detailed failed for %s",
                    type(checker).__name__,
                )
        return results


# ===================================================================== #
#  Policy Filter                                                         #
# ===================================================================== #


class PolicyFilter:
    """Filter a PolicySpec's actions against a constraint pipeline.

    Returns a new PolicySpec containing only the actions that pass all
    constraint checks.

    Parameters
    ----------
    pipeline:
        The constraint pipeline to validate actions against.
    """

    def __init__(self, pipeline: ConstraintPipeline) -> None:
        self._pipeline = pipeline

    def filter(
        self,
        policy: PolicySpec,
        goal: Goal,
        state: StateSnapshot,
    ) -> PolicySpec:
        """Return a new PolicySpec with only compliant actions.

        For stochastic policies, the probabilities are renormalized over
        the remaining actions.  If all actions are filtered out, an empty
        policy is returned.
        """
        compliant_actions: list[ActionSpec] = []
        compliant_indices: list[int] = []
        filtered_reasons: list[str] = []

        for idx, action in enumerate(policy.actions):
            passed, violations = self._pipeline.check_all(goal, state, action)
            if passed:
                compliant_actions.append(action)
                compliant_indices.append(idx)
            else:
                filtered_reasons.append(
                    f"action '{action.name}': {'; '.join(violations)}"
                )

        if filtered_reasons:
            logger.info(
                "PolicyFilter: filtered %d/%d actions: %s",
                len(filtered_reasons),
                len(policy.actions),
                " | ".join(filtered_reasons),
            )

        if not compliant_actions:
            return PolicySpec(
                metadata={
                    "policy_filter": "all_actions_filtered",
                    "original_size": policy.size,
                    "filtered_reasons": filtered_reasons,
                },
            )

        # Handle probability renormalization for stochastic policies
        new_probs: tuple[float, ...] | None = None
        if policy.is_stochastic and policy.probabilities is not None:
            raw_probs = [policy.probabilities[i] for i in compliant_indices]
            total = sum(raw_probs)
            if total > 0:
                new_probs = tuple(p / total for p in raw_probs)
            else:
                # Uniform distribution over compliant actions
                n = len(compliant_actions)
                new_probs = tuple(1.0 / n for _ in range(n))

        return PolicySpec(
            actions=tuple(compliant_actions),
            probabilities=new_probs,
            metadata={
                **dict(policy.metadata),
                "policy_filter": "applied",
                "original_size": policy.size,
                "filtered_count": len(filtered_reasons),
            },
        )
