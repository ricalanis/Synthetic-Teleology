"""Data layer: real Polymarket API client + simulated fallback."""

import os
import random
from abc import ABC, abstractmethod

from .models import Market


class MarketDataProvider(ABC):
    """Abstract interface for fetching market data."""

    @abstractmethod
    def get_markets(self) -> list[Market]:
        """Fetch all tracked markets with current prices."""

    @abstractmethod
    def get_market(self, market_id: str) -> Market:
        """Fetch a single market by ID."""


class PolymarketClient(MarketDataProvider):
    """Real Polymarket CLOB API client.

    Requires ``POLYMARKET_API_KEY`` environment variable.
    Uses httpx to fetch from the Polymarket REST API.
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self) -> None:
        import httpx

        self._api_key = os.environ["POLYMARKET_API_KEY"]
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=15.0,
        )
        self._tracked_ids: list[str] = []

    def get_markets(self) -> list[Market]:
        resp = self._client.get("/markets", params={"limit": 50, "active": True})
        resp.raise_for_status()
        markets = []
        for item in resp.json().get("data", []):
            m = self._parse_market(item)
            if m:
                markets.append(m)
        return markets

    def get_market(self, market_id: str) -> Market:
        resp = self._client.get(f"/markets/{market_id}")
        resp.raise_for_status()
        m = self._parse_market(resp.json())
        if m is None:
            raise ValueError(f"Could not parse market {market_id}")
        return m

    @staticmethod
    def _parse_market(data: dict) -> Market | None:
        try:
            tokens = data.get("tokens", [])
            yes_price = float(tokens[0].get("price", 0.5)) if tokens else 0.5
            return Market(
                id=str(data["condition_id"]),
                question=data.get("question", "Unknown"),
                yes_price=yes_price,
                no_price=round(1.0 - yes_price, 4),
                volume_24h=float(data.get("volume_num_24hr", 0)),
            )
        except (KeyError, IndexError, TypeError):
            return None


class SimulatedMarket(MarketDataProvider):
    """Random-walk simulation for testing without API access.

    Initializes N markets with random prices. Each call to
    ``get_markets()`` applies small random price movements.
    """

    QUESTIONS = [
        "Will BTC exceed $100k by end of Q2 2026?",
        "Will the Fed cut rates in March 2026?",
        "Will GPT-5 be released before July 2026?",
        "Will US GDP growth exceed 3% in 2026?",
        "Will there be a ceasefire in Ukraine by June 2026?",
        "Will Apple release AR glasses in 2026?",
        "Will SpaceX land Starship on Mars by 2027?",
        "Will the S&P 500 reach 7000 in 2026?",
        "Will a new pandemic be declared in 2026?",
        "Will the EU pass comprehensive AI regulation in 2026?",
    ]

    def __init__(self, num_markets: int = 5, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._markets: dict[str, Market] = {}
        for i in range(min(num_markets, len(self.QUESTIONS))):
            mid = f"sim-{i:03d}"
            yes_price = round(self._rng.uniform(0.15, 0.85), 4)
            self._markets[mid] = Market(
                id=mid,
                question=self.QUESTIONS[i],
                yes_price=yes_price,
                no_price=round(1.0 - yes_price, 4),
                volume_24h=round(self._rng.uniform(10_000, 500_000), 2),
            )

    def get_markets(self) -> list[Market]:
        # Apply random walk to each market
        for mid, market in self._markets.items():
            drift = self._rng.gauss(0, 0.02)
            new_yes = max(0.01, min(0.99, market.yes_price + drift))
            new_yes = round(new_yes, 4)
            self._markets[mid] = Market(
                id=mid,
                question=market.question,
                yes_price=new_yes,
                no_price=round(1.0 - new_yes, 4),
                volume_24h=round(
                    market.volume_24h * self._rng.uniform(0.9, 1.1), 2
                ),
            )
        return list(self._markets.values())

    def get_market(self, market_id: str) -> Market:
        if market_id not in self._markets:
            raise KeyError(f"Market {market_id} not found")
        return self._markets[market_id]


def get_provider(
    num_markets: int = 5, seed: int | None = 42
) -> MarketDataProvider:
    """Return real client if ``POLYMARKET_API_KEY`` is set, else simulated."""
    if os.environ.get("POLYMARKET_API_KEY"):
        return PolymarketClient()
    return SimulatedMarket(num_markets=num_markets, seed=seed)
