"""Graph wiring for the Polymarket trading agent."""

import random
import time

from synthetic_teleology.domain.values import ActionSpec, PolicySpec, StateSnapshot
from synthetic_teleology.graph import GraphBuilder

from .market_data import MarketDataProvider
from .models import Conviction, Market, Portfolio, Position
from .strategies import (
    CapitalChecker,
    PortfolioEvaluator,
    RiskChecker,
    TradingPlanner,
)


def _generate_convictions(
    markets: list[Market], rng: random.Random
) -> list[Conviction]:
    """Generate random conviction estimates for each market."""
    convictions = []
    for m in markets:
        # Our "fair value" deviates from market price
        deviation = rng.gauss(0, 0.15)
        fair_value = max(0.05, min(0.95, m.yes_price + deviation))
        confidence = rng.uniform(0.3, 0.9)
        edge = fair_value - m.yes_price
        convictions.append(
            Conviction(
                market_id=m.id,
                fair_value=round(fair_value, 4),
                confidence=round(confidence, 3),
                edge=round(edge, 4),
            )
        )
    return convictions


def _execute_trade(portfolio: Portfolio, action: ActionSpec) -> None:
    """Execute a trade on the portfolio (simulated)."""
    params = action.parameters
    act = params.get("action", "hold")

    if act == "buy":
        mid = params["market_id"]
        side = params["side"]
        shares = params["shares"]
        price = params["price"]
        cost = shares * price

        if cost > portfolio.cash:
            return  # Skip if insufficient cash

        portfolio.cash -= cost

        if mid in portfolio.positions:
            pos = portfolio.positions[mid]
            if pos.side == side:
                # Average into existing position
                total_shares = pos.shares + shares
                avg_cost = (
                    (pos.shares * pos.avg_cost + shares * price) / total_shares
                )
                portfolio.positions[mid] = Position(
                    market_id=mid,
                    side=side,
                    shares=round(total_shares, 2),
                    avg_cost=round(avg_cost, 4),
                )
            else:
                # Opposite side — close existing, open new if remainder
                if shares >= pos.shares:
                    remainder = shares - pos.shares
                    portfolio.cash += pos.shares * price  # close old
                    if remainder > 0:
                        portfolio.positions[mid] = Position(
                            market_id=mid,
                            side=side,
                            shares=round(remainder, 2),
                            avg_cost=round(price, 4),
                        )
                    else:
                        del portfolio.positions[mid]
                else:
                    portfolio.positions[mid] = Position(
                        market_id=mid,
                        side=pos.side,
                        shares=round(pos.shares - shares, 2),
                        avg_cost=pos.avg_cost,
                    )
        else:
            portfolio.positions[mid] = Position(
                market_id=mid,
                side=side,
                shares=round(shares, 2),
                avg_cost=round(price, 4),
            )

    elif act == "sell":
        mid = params["market_id"]
        shares = params["shares"]
        price = params["price"]

        if mid in portfolio.positions:
            pos = portfolio.positions[mid]
            sell_qty = min(shares, pos.shares)
            portfolio.cash += sell_qty * price
            remaining = pos.shares - sell_qty
            if remaining > 0.01:
                portfolio.positions[mid] = Position(
                    market_id=mid,
                    side=pos.side,
                    shares=round(remaining, 2),
                    avg_cost=pos.avg_cost,
                )
            else:
                del portfolio.positions[mid]


def build_trading_agent(
    provider: MarketDataProvider,
    capital: float = 10_000.0,
    max_steps: int = 50,
    seed: int = 42,
):
    """Build a LangGraph trading agent.

    Returns ``(app, initial_state, portfolio, trade_log)`` tuple.
    """
    rng = random.Random(seed)

    # Initialize portfolio and market state
    portfolio = Portfolio(cash=capital, initial_capital=capital)
    markets = provider.get_markets()
    market_map: dict[str, Market] = {m.id: m for m in markets}
    convictions = _generate_convictions(markets, rng)

    # Trade log for reporting
    trade_log: list[dict] = []

    # --- Environment callbacks ---

    def perceive_fn() -> StateSnapshot:
        """Fetch latest market data and encode as StateSnapshot."""
        fresh_markets = provider.get_markets()
        market_map.clear()
        for m in fresh_markets:
            market_map[m.id] = m

        # Encode portfolio + market state as values tuple
        # (cash, total_value, pnl, num_positions, *[yes_price for each market])
        prices = tuple(m.yes_price for m in fresh_markets)
        values = (
            portfolio.cash,
            portfolio.total_value,
            portfolio.pnl,
            float(len(portfolio.positions)),
            *prices,
        )
        return StateSnapshot(
            timestamp=time.time(),
            values=values,
            metadata={
                "markets": {m.id: m.yes_price for m in fresh_markets},
                "positions": len(portfolio.positions),
                "cash": portfolio.cash,
            },
        )

    def act_fn(policy: PolicySpec, state: StateSnapshot) -> ActionSpec | None:
        """Select the first action from the policy."""
        if policy.size > 0:
            return policy.actions[0]
        return None

    def transition_fn(action: ActionSpec) -> None:
        """Execute the trade and log it."""
        _execute_trade(portfolio, action)
        trade_log.append({
            "step": len(trade_log) + 1,
            "action": action.name,
            "params": dict(action.parameters),
            "cash_after": round(portfolio.cash, 2),
            "portfolio_value": round(portfolio.total_value, 2),
            "pnl": round(portfolio.pnl, 2),
        })

    # --- Build strategies ---
    evaluator = PortfolioEvaluator(portfolio, convictions, market_map)
    planner = TradingPlanner(portfolio, convictions, market_map)
    risk_checker = RiskChecker(portfolio)
    capital_checker = CapitalChecker(portfolio)

    # --- Build objective vector ---
    # Target: high alignment score (we aim for eval_score → 1.0)
    # Dimensions: (target_pnl_ratio, target_alignment)
    target_values = (0.1, 1.0)  # 10% return, perfect alignment

    app, initial_state = (
        GraphBuilder("polymarket-trader")
        .with_objective(target_values)
        .with_evaluator(evaluator)
        .with_planner(planner)
        .with_constraint_checkers(risk_checker, capital_checker)
        .with_max_steps(max_steps)
        .with_goal_achieved_threshold(0.85)
        .with_environment(
            perceive_fn=perceive_fn,
            act_fn=act_fn,
            transition_fn=transition_fn,
        )
        .with_metadata(
            portfolio=portfolio,
            convictions=convictions,
            provider=provider,
        )
        .build()
    )

    return app, initial_state, portfolio, trade_log, convictions
