"""Planning strategies for the Synthetic Teleology framework.

Implements the Strategy pattern for generating action policies pi_t
from the current goal and state.

Classes
-------
BasePlanner
    Abstract base class for all planners.
GreedyPlanner
    Selects single best action from a predefined action space.
StochasticPlanner
    Samples actions from a probability distribution.
HierarchicalPlanner
    Decomposes goal into sub-goals and plans for each.
LLMPlanner
    LLM-based planning via a provider interface.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
    PolicySpec,
    StateSnapshot,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Base Planner (ABC)                                                    #
# ===================================================================== #


class BasePlanner(ABC):
    """Abstract base class for planning strategies.

    Subclasses must implement :meth:`plan` which produces a ``PolicySpec``
    from a goal and state.
    """

    @abstractmethod
    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Generate an action policy for *goal* given *state*.

        Parameters
        ----------
        goal:
            The target goal to plan toward.
        state:
            The current perceived state.

        Returns
        -------
        PolicySpec
            A deterministic or stochastic policy.
        """


# ===================================================================== #
#  Greedy Planner                                                        #
# ===================================================================== #


class GreedyPlanner(BasePlanner):
    """Select the single best action from a predefined action space.

    For each candidate action, the planner estimates the resulting state
    after applying the action (using a simple additive model) and picks
    the action that minimizes the distance to the goal objective.

    Parameters
    ----------
    action_space:
        The set of candidate actions available to the agent.
    state_effect_key:
        The key in ``ActionSpec.parameters`` that holds the per-dimension
        state effect vector (as a list of floats). Defaults to ``"effect"``.
    """

    def __init__(
        self,
        action_space: Sequence[ActionSpec],
        state_effect_key: str = "effect",
    ) -> None:
        if not action_space:
            raise ValueError("GreedyPlanner requires a non-empty action space")
        self._action_space = list(action_space)
        self._effect_key = state_effect_key

    @property
    def action_space(self) -> list[ActionSpec]:
        """Return the action space."""
        return list(self._action_space)

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Pick the single action that minimizes distance to goal."""
        if goal.objective is None:
            logger.warning("GreedyPlanner: goal has no objective, returning empty policy")
            return PolicySpec()

        best_action: ActionSpec | None = None
        best_distance = math.inf

        goal_arr = np.array(goal.objective.values, dtype=np.float64)
        state_arr = np.array(state.values, dtype=np.float64)
        weights = goal.objective.weights or tuple(
            1.0 for _ in goal.objective.values
        )
        weight_arr = np.array(weights, dtype=np.float64)

        for action in self._action_space:
            # Get the action's effect on state
            effect = action.parameters.get(self._effect_key)
            if effect is None:
                # No effect specified: assume no state change
                projected = state_arr.copy()
            else:
                effect_arr = np.array(effect, dtype=np.float64)
                if effect_arr.shape != state_arr.shape:
                    logger.debug(
                        "GreedyPlanner: action %s effect shape mismatch, skipping",
                        action.name,
                    )
                    continue
                projected = state_arr + effect_arr

            # Compute weighted distance from projected state to goal
            diff = projected - goal_arr
            distance = float(np.sqrt(np.sum(weight_arr * diff ** 2)))

            if distance < best_distance:
                best_distance = distance
                best_action = action

        if best_action is None:
            logger.warning("GreedyPlanner: no valid action found, returning empty policy")
            return PolicySpec()

        return PolicySpec(
            actions=(best_action,),
            metadata={
                "planner": "GreedyPlanner",
                "best_distance": best_distance,
            },
        )


# ===================================================================== #
#  Stochastic Planner                                                    #
# ===================================================================== #


class StochasticPlanner(BasePlanner):
    """Sample actions from a distribution weighted by goal proximity.

    Uses a softmax over negative distances to assign probabilities to
    each action in the action space.

    Parameters
    ----------
    action_space:
        The set of candidate actions available.
    state_effect_key:
        Key in ``ActionSpec.parameters`` for the state effect vector.
    temperature:
        Softmax temperature. Lower = more greedy, higher = more uniform.
        Defaults to 1.0.
    sample_size:
        Number of actions to include in the output policy.
        Defaults to 0 (include all with their probabilities).
    seed:
        Random seed for reproducibility. Defaults to None.
    """

    def __init__(
        self,
        action_space: Sequence[ActionSpec],
        state_effect_key: str = "effect",
        temperature: float = 1.0,
        sample_size: int = 0,
        seed: int | None = None,
    ) -> None:
        if not action_space:
            raise ValueError("StochasticPlanner requires a non-empty action space")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self._action_space = list(action_space)
        self._effect_key = state_effect_key
        self._temperature = temperature
        self._sample_size = sample_size
        self._rng = np.random.default_rng(seed)

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Generate a stochastic policy by sampling from a softmax distribution."""
        if goal.objective is None:
            return PolicySpec()

        goal_arr = np.array(goal.objective.values, dtype=np.float64)
        state_arr = np.array(state.values, dtype=np.float64)
        weights = goal.objective.weights or tuple(
            1.0 for _ in goal.objective.values
        )
        weight_arr = np.array(weights, dtype=np.float64)

        # Compute distances for all actions
        valid_actions: list[ActionSpec] = []
        distances: list[float] = []

        for action in self._action_space:
            effect = action.parameters.get(self._effect_key)
            if effect is None:
                projected = state_arr.copy()
            else:
                effect_arr = np.array(effect, dtype=np.float64)
                if effect_arr.shape != state_arr.shape:
                    continue
                projected = state_arr + effect_arr

            diff = projected - goal_arr
            dist = float(np.sqrt(np.sum(weight_arr * diff ** 2)))
            valid_actions.append(action)
            distances.append(dist)

        if not valid_actions:
            return PolicySpec()

        # Softmax over negative distances (lower distance = higher probability)
        neg_dists = np.array([-d / self._temperature for d in distances])
        # Numerical stability: subtract max
        neg_dists -= neg_dists.max()
        exp_vals = np.exp(neg_dists)
        probabilities = exp_vals / exp_vals.sum()

        if self._sample_size > 0 and self._sample_size < len(valid_actions):
            # Sample a subset
            indices = self._rng.choice(
                len(valid_actions),
                size=self._sample_size,
                replace=False,
                p=probabilities,
            )
            sampled_actions = tuple(valid_actions[i] for i in indices)
            sampled_probs = np.array([probabilities[i] for i in indices])
            sampled_probs = sampled_probs / sampled_probs.sum()  # renormalize
            return PolicySpec(
                actions=sampled_actions,
                probabilities=tuple(float(p) for p in sampled_probs),
                metadata={
                    "planner": "StochasticPlanner",
                    "temperature": self._temperature,
                    "sample_size": self._sample_size,
                },
            )

        return PolicySpec(
            actions=tuple(valid_actions),
            probabilities=tuple(float(p) for p in probabilities),
            metadata={
                "planner": "StochasticPlanner",
                "temperature": self._temperature,
            },
        )


# ===================================================================== #
#  Hierarchical Planner                                                  #
# ===================================================================== #


class HierarchicalPlanner(BasePlanner):
    """Decompose a goal into sub-goals and plan for each using a sub-planner.

    The planner splits the goal objective into sub-objectives by grouping
    dimensions, plans for each sub-goal independently, and then merges
    the resulting policies.

    Parameters
    ----------
    sub_planner:
        The planner used for leaf-level sub-goal planning.
    sub_goal_groups:
        Optional explicit grouping of dimension indices into sub-goals.
        Each group is a list of dimension indices. If not provided,
        each dimension becomes its own sub-goal.
    """

    def __init__(
        self,
        sub_planner: BasePlanner,
        sub_goal_groups: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self._sub_planner = sub_planner
        self._sub_goal_groups = (
            [list(g) for g in sub_goal_groups] if sub_goal_groups else None
        )

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Decompose goal, plan sub-goals, and merge policies."""
        if goal.objective is None:
            return PolicySpec()

        groups = self._get_groups(goal.objective.dimension)
        all_actions: list[ActionSpec] = []
        sub_metadata: list[dict[str, Any]] = []

        for group_idx, dim_indices in enumerate(groups):
            sub_goal = self._extract_sub_goal(goal, dim_indices, group_idx)
            sub_state = self._extract_sub_state(state, dim_indices)

            try:
                sub_policy = self._sub_planner.plan(sub_goal, sub_state)
                all_actions.extend(sub_policy.actions)
                sub_metadata.append({
                    "group": group_idx,
                    "dimensions": dim_indices,
                    "num_actions": sub_policy.size,
                })
            except Exception:
                logger.exception(
                    "HierarchicalPlanner: sub-planner failed for group %d",
                    group_idx,
                )
                continue

        if not all_actions:
            return PolicySpec()

        # Deduplicate actions by action_id
        seen: set[str] = set()
        unique_actions: list[ActionSpec] = []
        for action in all_actions:
            if action.action_id not in seen:
                seen.add(action.action_id)
                unique_actions.append(action)

        return PolicySpec(
            actions=tuple(unique_actions),
            metadata={
                "planner": "HierarchicalPlanner",
                "num_groups": len(groups),
                "sub_plans": sub_metadata,
            },
        )

    def _get_groups(self, dimension: int) -> list[list[int]]:
        """Return dimension index groups."""
        if self._sub_goal_groups is not None:
            return self._sub_goal_groups
        # Default: each dimension is its own group
        return [[i] for i in range(dimension)]

    def _extract_sub_goal(
        self,
        goal: Goal,
        dim_indices: list[int],
        group_idx: int,
    ) -> Goal:
        """Extract a sub-goal for the specified dimensions."""
        assert goal.objective is not None
        sub_values = tuple(goal.objective.values[i] for i in dim_indices)
        sub_directions = tuple(goal.objective.directions[i] for i in dim_indices)
        sub_weights = (
            tuple(goal.objective.weights[i] for i in dim_indices)
            if goal.objective.weights
            else None
        )

        sub_objective = ObjectiveVector(
            values=sub_values,
            directions=sub_directions,
            weights=sub_weights,
        )

        return Goal(
            name=f"{goal.name}_sub_{group_idx}",
            description=f"Sub-goal {group_idx} of {goal.name}",
            objective=sub_objective,
            parent_id=goal.goal_id,
            version=goal.version,
        )

    def _extract_sub_state(
        self,
        state: StateSnapshot,
        dim_indices: list[int],
    ) -> StateSnapshot:
        """Extract a sub-state for the specified dimensions."""
        sub_values = tuple(state.values[i] for i in dim_indices)
        return StateSnapshot(
            timestamp=state.timestamp,
            values=sub_values,
            source=state.source,
            metadata=dict(state.metadata),
        )


# ===================================================================== #
#  LLM Planner                                                           #
# ===================================================================== #


class LLMPlanner(BasePlanner):
    """LLM-based planner that generates action policies via a provider.

    The planner formats a structured prompt describing the goal and state,
    sends it to the LLM, and parses the response into a ``PolicySpec``.

    Parameters
    ----------
    provider:
        An LLM provider with a ``generate(prompt: str, **kwargs) -> str``
        method.
    available_actions:
        The set of action names/specs the LLM can choose from.
        Provided as context in the prompt.
    system_prompt:
        Optional system-level instructions.
    temperature:
        Sampling temperature. Defaults to 0.5.
    """

    def __init__(
        self,
        provider: Any,
        available_actions: Sequence[ActionSpec] | None = None,
        system_prompt: str = "",
        temperature: float = 0.5,
    ) -> None:
        self._provider = provider
        self._available_actions = list(available_actions or [])
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._temperature = temperature

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are an action planner for a goal-directed AI agent. "
            "Given a goal and current state, propose an ordered list of "
            "actions. Respond with a JSON object: "
            '{"actions": [{"name": "<action_name>", "parameters": {<params>}, '
            '"cost": <float>}], "reasoning": "<brief explanation>"}.'
        )

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Generate a policy via the LLM provider."""
        prompt = self._build_prompt(goal, state)

        try:
            response = self._provider.generate(
                prompt,
                system_prompt=self._system_prompt,
                temperature=self._temperature,
            )
            return self._parse_response(response)
        except Exception as exc:
            logger.warning("LLMPlanner: provider call failed: %s", exc)
            return PolicySpec()

    def _build_prompt(self, goal: Goal, state: StateSnapshot) -> str:
        """Build the planning prompt."""
        obj_repr = "None"
        if goal.objective is not None:
            obj_repr = (
                f"values={list(goal.objective.values)}, "
                f"directions=[{', '.join(d.value for d in goal.objective.directions)}]"
            )

        action_list = ""
        if self._available_actions:
            action_lines = []
            for a in self._available_actions:
                action_lines.append(
                    f"  - {a.name} (cost={a.cost}, params={dict(a.parameters)})"
                )
            action_list = "**Available Actions**:\n" + "\n".join(action_lines) + "\n\n"

        return (
            f"## Planning Request\n\n"
            f"**Goal**: {goal.name} (v{goal.version})\n"
            f"  Description: {goal.description}\n"
            f"  Objective: {obj_repr}\n\n"
            f"**Current State**: values={list(state.values)}\n\n"
            f"{action_list}"
            f"Propose an ordered list of actions to move from the current "
            f"state toward the goal objective."
        )

    def _parse_response(self, response: str) -> PolicySpec:
        """Parse the LLM response into a PolicySpec."""
        import json

        cleaned = response.strip()
        if "```json" in cleaned:
            start = cleaned.index("```json") + 7
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.index("```") + 3
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()

        try:
            data = json.loads(cleaned)
            raw_actions = data.get("actions", [])
            reasoning = data.get("reasoning", "")

            actions: list[ActionSpec] = []
            for raw in raw_actions:
                if not isinstance(raw, dict):
                    continue
                actions.append(
                    ActionSpec(
                        name=str(raw.get("name", "")),
                        parameters=dict(raw.get("parameters", {})),
                        cost=float(raw.get("cost", 0.0)),
                    )
                )

            return PolicySpec(
                actions=tuple(actions),
                metadata={
                    "planner": "LLMPlanner",
                    "reasoning": reasoning,
                },
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("LLMPlanner: failed to parse response: %s", exc)
            return PolicySpec()
