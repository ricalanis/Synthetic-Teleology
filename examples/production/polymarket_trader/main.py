"""Entry point for the Polymarket Trader agent.

Run:
    PYTHONPATH=src python -m examples.production.polymarket_trader.main

Options:
    --capital 10000     Initial capital
    --markets 5         Number of markets to track
    --steps 50          Max trading rounds
    --live              Use real Polymarket API (needs POLYMARKET_API_KEY)
    --seed 42           Random seed for reproducibility
"""

import argparse

from .agent import build_trading_agent
from .market_data import get_provider


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket Trader — Goal-directed prediction market agent"
    )
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital")
    parser.add_argument("--markets", type=int, default=5, help="Number of markets to track")
    parser.add_argument("--steps", type=int, default=50, help="Max trading rounds")
    parser.add_argument("--live", action="store_true", help="Use real Polymarket API")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- Setup ---
    provider = get_provider(num_markets=args.markets, seed=args.seed)
    mode = "LIVE" if args.live else "SIMULATED"

    print("=" * 60)
    print(f"  Polymarket Trader Agent ({mode} mode)")
    print("=" * 60)
    print(f"  Capital: ${args.capital:,.2f}")
    print(f"  Markets: {args.markets}")
    print(f"  Max steps: {args.steps}")
    print()

    # Show initial markets
    markets = provider.get_markets()
    print("Markets:")
    for m in markets:
        print(f"  [{m.id}] {m.question}")
        print(f"          YES: {m.yes_price:.4f}  NO: {m.no_price:.4f}  Vol: ${m.volume_24h:,.0f}")
    print()

    # --- Build and run agent ---
    app, initial_state, portfolio, trade_log, convictions = build_trading_agent(
        provider=provider,
        capital=args.capital,
        max_steps=args.steps,
        seed=args.seed,
    )

    print("Convictions:")
    for c in convictions:
        direction = "LONG" if c.edge > 0 else "SHORT"
        print(
            f"  {c.market_id}: fair={c.fair_value:.4f} "
            f"edge={c.edge:+.4f} conf={c.confidence:.3f} → {direction}"
        )
    print()

    print("Running teleological trading loop...")
    print("-" * 60)

    result = app.invoke(initial_state)

    # --- Results ---
    print("-" * 60)
    print()

    # Trade log
    trades_executed = [t for t in trade_log if "hold" not in t["action"]]
    print(f"Trades executed: {len(trades_executed)}")
    for t in trades_executed[:20]:  # Show first 20
        print(
            f"  Step {t['step']:3d}: {t['action']:30s} "
            f"| Cash: ${t['cash_after']:>10,.2f} "
            f"| Value: ${t['portfolio_value']:>10,.2f} "
            f"| PnL: ${t['pnl']:>+8,.2f}"
        )
    if len(trades_executed) > 20:
        print(f"  ... and {len(trades_executed) - 20} more trades")
    print()

    # Final positions
    print("Final Positions:")
    if portfolio.positions:
        for mid, pos in portfolio.positions.items():
            print(
                f"  {mid}: {pos.side} {pos.shares:.2f} shares "
                f"@ avg {pos.avg_cost:.4f}"
            )
    else:
        print("  (no open positions)")
    print()

    # Summary
    print("=" * 60)
    print(f"  Stop reason:      {result.get('stop_reason', 'none')}")
    print(f"  Steps completed:  {result['step']}")
    print(f"  Final eval score: {result['eval_signal'].score:.4f}")
    print(f"  Final cash:       ${portfolio.cash:>10,.2f}")
    print(f"  Portfolio value:  ${portfolio.total_value:>10,.2f}")
    pnl_pct = portfolio.pnl / portfolio.initial_capital
    print(f"  P&L:              ${portfolio.pnl:>+10,.2f} ({pnl_pct:+.1%})")
    print(f"  Events emitted:   {len(result.get('events', []))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
