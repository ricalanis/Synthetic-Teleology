"""Domain model for the Polymarket trader agent."""

from dataclasses import dataclass, field


@dataclass
class Market:
    """A prediction market with current prices."""

    id: str
    question: str
    yes_price: float  # 0.0-1.0 (implied probability)
    no_price: float
    volume_24h: float


@dataclass
class Position:
    """A position held in a specific market."""

    market_id: str
    side: str  # "YES" or "NO"
    shares: float
    avg_cost: float

    @property
    def notional_value(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class Portfolio:
    """Collection of positions plus cash."""

    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 10000.0
    initial_capital: float = 10000.0

    @property
    def total_value(self) -> float:
        position_value = sum(
            pos.shares * (1.0 if pos.side == "YES" else 1.0)  # mark at par
            for pos in self.positions.values()
        )
        return self.cash + position_value

    @property
    def pnl(self) -> float:
        return self.total_value - self.initial_capital


@dataclass(frozen=True)
class Conviction:
    """Our estimate of a market's true probability."""

    market_id: str
    fair_value: float  # Our estimate of true probability (0-1)
    confidence: float  # How sure we are (0-1)
    edge: float = 0.0  # fair_value - current_price (computed)
