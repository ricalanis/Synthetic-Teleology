"""Custom evaluator, planner, and constraint checkers for trading."""

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.services.constraint_engine import BaseConstraintChecker
from synthetic_teleology.services.evaluation import BaseEvaluator
from synthetic_teleology.services.planning import BasePlanner

from .models import Conviction, Market, Portfolio


class PortfolioEvaluator(BaseEvaluator):
    """Score = how well portfolio positions align with convictions.

    High score: positions sized proportional to edge * confidence.
    Low score: holding positions against conviction or missing opportunities.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        convictions: list[Conviction],
        markets: dict[str, Market],
    ) -> None:
        self._portfolio = portfolio
        self._convictions = convictions
        self._markets = markets

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        if not self._convictions:
            return EvalSignal(score=0.0, explanation="No convictions to evaluate")

        alignment_scores: list[float] = []

        for conv in self._convictions:
            market = self._markets.get(conv.market_id)
            if market is None:
                continue

            # Compute edge: how far is market price from our fair value?
            edge = conv.fair_value - market.yes_price

            # Check if we have a position aligned with our conviction
            pos = self._portfolio.positions.get(conv.market_id)
            if pos is None:
                # No position — penalize proportional to missed opportunity
                alignment = -abs(edge) * conv.confidence
            else:
                # Position exists — score by alignment direction + sizing
                if (edge > 0 and pos.side == "YES") or (
                    edge < 0 and pos.side == "NO"
                ):
                    # Correct direction — score by position sizing vs target
                    target_size = (
                        abs(edge) * conv.confidence * self._portfolio.initial_capital
                    )
                    actual_size = pos.shares * pos.avg_cost
                    sizing_ratio = min(actual_size, target_size) / max(
                        target_size, 0.01
                    )
                    alignment = sizing_ratio * conv.confidence
                else:
                    # Wrong direction — strong penalty
                    alignment = -conv.confidence

            alignment_scores.append(alignment)

        avg_alignment = sum(alignment_scores) / len(alignment_scores)
        # Normalize to [-1, 1]
        score = max(-1.0, min(1.0, avg_alignment))

        # PnL bonus: small reward for positive P&L
        pnl_ratio = self._portfolio.pnl / max(self._portfolio.initial_capital, 1.0)
        pnl_bonus = max(-0.2, min(0.2, pnl_ratio))
        score = max(-1.0, min(1.0, score + pnl_bonus))

        return EvalSignal(
            score=score,
            confidence=min(c.confidence for c in self._convictions),
            explanation=(
                f"Portfolio alignment: {avg_alignment:.3f}, "
                f"PnL: {self._portfolio.pnl:.2f}"
            ),
        )


class TradingPlanner(BasePlanner):
    """Generate trade orders to move portfolio toward conviction targets.

    For each conviction:
    - Compute target_position = edge * confidence * kelly_fraction * capital
    - Compare to current position
    - If gap > threshold: generate buy/sell ActionSpec
    """

    def __init__(
        self,
        portfolio: Portfolio,
        convictions: list[Conviction],
        markets: dict[str, Market],
        kelly_fraction: float = 0.25,
        min_edge: float = 0.05,
    ) -> None:
        self._portfolio = portfolio
        self._convictions = convictions
        self._markets = markets
        self._kelly_fraction = kelly_fraction
        self._min_edge = min_edge

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        actions: list[ActionSpec] = []

        for conv in self._convictions:
            market = self._markets.get(conv.market_id)
            if market is None:
                continue

            edge = conv.fair_value - market.yes_price

            # Skip small edges
            if abs(edge) < self._min_edge:
                continue

            # Kelly criterion: fraction = edge / odds
            price = market.yes_price if edge > 0 else market.no_price
            kelly = abs(edge) / max(price, 0.01)
            fraction = min(kelly, 1.0) * self._kelly_fraction * conv.confidence

            target_notional = fraction * self._portfolio.initial_capital
            side = "YES" if edge > 0 else "NO"

            # Current position in this market
            pos = self._portfolio.positions.get(conv.market_id)
            current_notional = 0.0
            if pos and pos.side == side:
                current_notional = pos.shares * pos.avg_cost

            gap = target_notional - current_notional

            if abs(gap) < 10.0:
                continue  # Not worth trading

            if gap > 0:
                # Buy
                buy_price = market.yes_price if side == "YES" else market.no_price
                shares = gap / max(buy_price, 0.01)
                actions.append(
                    ActionSpec(
                        name=f"buy_{side}_{conv.market_id}",
                        parameters={
                            "market_id": conv.market_id,
                            "side": side,
                            "shares": round(shares, 2),
                            "price": round(buy_price, 4),
                            "action": "buy",
                        },
                        cost=round(gap, 2),
                    )
                )
            else:
                # Sell (reduce position)
                sell_shares = min(
                    abs(gap) / max(price, 0.01),
                    pos.shares if pos else 0,
                )
                if sell_shares > 0:
                    actions.append(
                        ActionSpec(
                            name=f"sell_{side}_{conv.market_id}",
                            parameters={
                                "market_id": conv.market_id,
                                "side": side,
                                "shares": round(sell_shares, 2),
                                "price": round(price, 4),
                                "action": "sell",
                            },
                            cost=0.0,
                        )
                    )

        if not actions:
            actions.append(
                ActionSpec(name="hold", parameters={"action": "hold"}, cost=0.0)
            )

        return PolicySpec(actions=tuple(actions))


class RiskChecker(BaseConstraintChecker):
    """Max position size per market, max total exposure, drawdown limit."""

    def __init__(
        self,
        portfolio: Portfolio,
        max_position_pct: float = 0.3,
        max_exposure_pct: float = 0.8,
        max_drawdown_pct: float = 0.2,
    ) -> None:
        self._portfolio = portfolio
        self._max_position_pct = max_position_pct
        self._max_exposure_pct = max_exposure_pct
        self._max_drawdown_pct = max_drawdown_pct

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        capital = self._portfolio.initial_capital

        # Check drawdown
        drawdown = -self._portfolio.pnl / max(capital, 1.0)
        if drawdown > self._max_drawdown_pct:
            return (
                False,
                f"Drawdown {drawdown:.1%} exceeds limit {self._max_drawdown_pct:.0%}",
            )

        # Check total exposure
        total_exposure = sum(
            pos.shares * pos.avg_cost for pos in self._portfolio.positions.values()
        )
        exposure_pct = total_exposure / max(capital, 1.0)
        if exposure_pct > self._max_exposure_pct:
            return (
                False,
                f"Exposure {exposure_pct:.1%} exceeds limit {self._max_exposure_pct:.0%}",
            )

        # Check single position limit (for proposed action)
        if action and action.parameters.get("action") == "buy":
            mid = action.parameters.get("market_id", "")
            pos = self._portfolio.positions.get(mid)
            current = (pos.shares * pos.avg_cost) if pos else 0.0
            proposed = current + action.cost
            if proposed / max(capital, 1.0) > self._max_position_pct:
                return (
                    False,
                    f"Position in {mid} would be {proposed / capital:.1%}, "
                    f"exceeds {self._max_position_pct:.0%}",
                )

        return (True, "")


class CapitalChecker(BaseConstraintChecker):
    """Ensure trades don't exceed available cash."""

    def __init__(self, portfolio: Portfolio, min_cash_reserve: float = 100.0) -> None:
        self._portfolio = portfolio
        self._min_cash_reserve = min_cash_reserve

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        if action and action.parameters.get("action") == "buy":
            cost = action.cost
            available = self._portfolio.cash - self._min_cash_reserve
            if cost > available:
                return (
                    False,
                    f"Trade cost {cost:.2f} exceeds available cash {available:.2f}",
                )
        return (True, "")
