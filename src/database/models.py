"""Database models — dataclasses for all SQLite table rows.

Each class maps 1-to-1 to a database table and provides:
  - Proper type hints and default values
  - to_dict()     — serialise to a plain dict (for DB insertion / JSON)
  - from_dict()   — deserialise from a plain dict (from DB row / JSON)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Helper shared by all models                                                  #
# --------------------------------------------------------------------------- #

def _from_dict_helper(cls: type, data: dict[str, Any]) -> Any:
    """Return a new *cls* instance, ignoring keys not in the dataclass."""
    valid = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in valid})


# --------------------------------------------------------------------------- #
# trades table                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class Trade:
    """Represents one row in the ``trades`` table.

    A trade spans from the entry signal to the exit (or is still open when
    ``exit_date`` is None).
    """

    symbol: str                           # e.g. "TCS"
    trade_type: str                       # "BUY" or "SELL"
    quantity: int
    price: float                          # entry price

    id: Optional[int] = None
    order_id: Optional[str] = None        # Angel One order ID
    status: str = "PENDING"               # PENDING | EXECUTED | FAILED | CANCELLED
    strategy_signal: Optional[str] = None # JSON blob — signal that triggered trade
    research_summary: Optional[str] = None
    risk_check_result: Optional[str] = None  # JSON blob — risk manager assessment
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    entry_date: Optional[str] = None      # ISO datetime string (IST)
    exit_date: Optional[str] = None       # None while still holding
    exit_price: Optional[float] = None
    pnl: Optional[float] = None           # None while still holding
    pnl_percentage: Optional[float] = None
    holding_days: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trade":
        return _from_dict_helper(cls, data)


# --------------------------------------------------------------------------- #
# watchlist table                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class WatchlistItem:
    """Represents one row in the ``watchlist`` table."""

    symbol: str
    date: str = ""                        # "YYYY-MM-DD"
    price: Optional[float] = None
    avg_volume_cr: Optional[float] = None # average daily volume in crores INR
    atr_pct: Optional[float] = None       # ATR as % of price
    ema_50: Optional[float] = None
    sector: Optional[str] = None
    in_index: Optional[str] = None        # "NIFTY_50", "NIFTY_NEXT_50", etc.
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchlistItem":
        return _from_dict_helper(cls, data)


# --------------------------------------------------------------------------- #
# signals table                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Signal:
    """Represents one row in the ``signals`` table."""

    symbol: str
    date: str = ""                          # "YYYY-MM-DD"
    signal_type: Optional[str] = None       # "BUY" | "SELL" | "HOLD"
    signal_source: Optional[str] = None     # "QUANT" | "RESEARCH" | "ORCHESTRATOR"
    strength: Optional[float] = None        # 0.0 – 1.0
    indicators: Optional[str] = None        # JSON blob (RSI, MACD values, etc.)
    was_acted_on: int = 0                   # 1 if a trade was placed, else 0
    reason_if_skipped: Optional[str] = None # why risk manager rejected the signal
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Signal":
        return _from_dict_helper(cls, data)


# --------------------------------------------------------------------------- #
# portfolio_snapshots table                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class PortfolioSnapshot:
    """Represents one row in the ``portfolio_snapshots`` table."""

    date: str                               # "YYYY-MM-DD"
    time: str                               # "HH:MM:SS"
    total_value: Optional[float] = None     # current portfolio value (INR)
    invested_amount: Optional[float] = None # total money currently in positions
    available_cash: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl_today: Optional[float] = None
    open_positions: Optional[int] = None
    snapshot_data: Optional[str] = None     # JSON blob with per-stock breakdown
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortfolioSnapshot":
        return _from_dict_helper(cls, data)


# --------------------------------------------------------------------------- #
# agent_logs table                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class AgentLog:
    """Represents one row in the ``agent_logs`` table."""

    agent_name: str                         # "orchestrator", "research", etc.
    session_type: str                       # "morning_briefing", "trade_decision", …
    date: str = ""                          # "YYYY-MM-DD"
    input_data: Optional[str] = None        # JSON — what was given to the agent
    output_data: Optional[str] = None       # JSON — what the agent produced
    llm_provider_used: Optional[str] = None # "gemini_flash", "groq", etc.
    llm_calls_count: int = 0
    search_calls_count: int = 0
    duration_seconds: float = 0.0
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentLog":
        return _from_dict_helper(cls, data)


# --------------------------------------------------------------------------- #
# daily_summary table                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class DailySummary:
    """Represents one row in the ``daily_summary`` table."""

    date: str                               # "YYYY-MM-DD" (UNIQUE in DB)
    market_outlook: Optional[str] = None    # morning briefing narrative
    trades_executed: int = 0
    trades_profitable: int = 0
    total_pnl: float = 0.0
    portfolio_value_eod: Optional[float] = None
    nifty_change_pct: Optional[float] = None  # benchmark comparison
    agent_summary: Optional[str] = None       # orchestrator learnings
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DailySummary":
        return _from_dict_helper(cls, data)


# --------------------------------------------------------------------------- #
# LLMUsageRecord — for budget_manager tracking                                #
# --------------------------------------------------------------------------- #

@dataclass
class LLMUsageRecord:
    """Tracks one LLM API call for budget management (used by budget_manager)."""

    provider: str
    model: str
    tokens_used: int
    timestamp: str                # ISO datetime string (IST)
    purpose: str                  # "sentiment", "research", "signal_review", …
    id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMUsageRecord":
        return _from_dict_helper(cls, data)
