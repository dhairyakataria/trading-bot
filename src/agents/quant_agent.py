"""Quant Agent — technical analysis signal generator for swing trading.

Scans the daily watchlist (~30-50 stocks) every 60 minutes during market
hours (9:30 AM – 3:15 PM IST) and generates BUY/SELL signals based on
five swing-trading strategies.  Pure Python/maths — zero LLM calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from src.database.models import Signal
from src.utils.logger import get_logger

IST = ZoneInfo("Asia/Kolkata")
_log = get_logger("agents.quant_agent")


class QuantAgent:
    """Generates BUY/SELL signals via technical analysis for swing trading.

    Args:
        config:               Project-wide config dict.
        broker_client:        AngelOne (or compatible) broker wrapper.
        technical_indicators: TechnicalIndicators instance.
        db_manager:           DatabaseManager instance for signal persistence.
    """

    # Base signal strength per strategy (0.0 – 1.0)
    _BASE_STRENGTH: Dict[str, float] = {
        "RSI_OVERSOLD_BOUNCE": 0.70,
        "EMA_PULLBACK":        0.65,
        "VOLUME_BREAKOUT":     0.75,
        "TREND_FOLLOWING":     0.60,
        "EXIT_SIGNAL":         0.80,
    }

    # Strength boost added per additional confirming strategy
    _STRENGTH_BOOST = 0.30

    def __init__(
        self,
        config: Dict[str, Any],
        broker_client: Any,
        technical_indicators: Any,
        db_manager: Any,
    ) -> None:
        self.config = config
        self.broker = broker_client
        self.ti = technical_indicators
        self.db = db_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_watchlist(
        self,
        watchlist: List[Dict[str, Any]],
        current_holdings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Scan every stock in *watchlist* and return actionable BUY/SELL signals.

        HOLD stocks are omitted from the output.

        Args:
            watchlist:         List of stock dicts — each must have a ``"symbol"`` key.
            current_holdings:  Symbols currently held in the portfolio.  BUY
                               signals are suppressed for these stocks.

        Returns:
            List of signal dicts (BUY or SELL only).  Each dict matches the
            schema documented in the module docstring.
        """
        holdings: Set[str] = set(current_holdings or [])
        results: List[Dict[str, Any]] = []
        seen_buy: Set[str] = set()  # dedup — one active BUY per symbol

        for stock in watchlist:
            symbol = stock.get("symbol", "").strip()
            if not symbol:
                continue
            try:
                signal = self._analyse_stock(symbol, holdings, seen_buy)
                if signal is not None:
                    results.append(signal)
                    self._store_signal(signal)
                    if signal["signal"] == "BUY":
                        seen_buy.add(symbol)
            except Exception as exc:
                _log.error("Unexpected error analysing %s: %s", symbol, exc)

        _log.info(
            "Scan complete — %d actionable signal(s) from %d stock(s)",
            len(results),
            len(watchlist),
        )
        return results

    # ------------------------------------------------------------------
    # Strategy checkers — public so they can be unit-tested in isolation
    # ------------------------------------------------------------------

    def check_rsi_oversold_bounce(
        self,
        df: Any,
        symbol: str,
        inds: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """BUY — RSI oversold dip inside an established uptrend.

        Conditions (ALL required):
        - RSI(14) < 35
        - Price above 50-day EMA  (uptrend intact)
        - MACD histogram > 0 OR MACD bullish crossover
        - Volume > 1.2× 20-day average
        """
        if inds is None:
            inds = self.ti.generate_full_analysis(df, symbol).get("indicators", {})

        rsi    = inds.get("rsi", {})
        macd   = inds.get("macd", {})
        ema    = inds.get("ema", {})
        volume = inds.get("volume", {})
        atr    = inds.get("atr", {})

        if "error" in rsi or "error" in macd or "error" in ema:
            _log.debug("%s RSI_OVERSOLD_BOUNCE: indicator error — skipping", symbol)
            return None

        rsi_val       = rsi.get("value", 50.0)
        price         = float(df["close"].iloc[-1])
        price_vs_50   = ema.get("price_vs_ema_50", "BELOW")
        macd_sig      = macd.get("signal", "")
        hist          = macd.get("histogram", 0.0)
        vol_ratio     = volume.get("volume_ratio", 0.0)
        atr_val       = atr.get("atr", 0.0)
        ema_50        = ema.get("ema_50")

        if rsi_val >= 35:
            _log.debug("%s RSI_OVERSOLD_BOUNCE: RSI %.1f ≥ 35", symbol, rsi_val)
            return None
        if price_vs_50 != "ABOVE":
            _log.debug("%s RSI_OVERSOLD_BOUNCE: price below EMA50", symbol)
            return None
        macd_bullish = macd_sig in ("BULLISH_CROSSOVER", "BULLISH") or hist > 0
        if not macd_bullish:
            _log.debug(
                "%s RSI_OVERSOLD_BOUNCE: MACD not bullish (sig=%s, hist=%.4f)",
                symbol, macd_sig, hist,
            )
            return None
        if vol_ratio < 1.2:
            _log.debug(
                "%s RSI_OVERSOLD_BOUNCE: volume ratio %.2f < 1.2", symbol, vol_ratio
            )
            return None

        stop_loss        = self.calculate_stop_loss("RSI_OVERSOLD_BOUNCE", price, atr_val)
        target_1, target_2 = self.calculate_targets(
            "RSI_OVERSOLD_BOUNCE", price, atr_val, inds.get("support_resistance", {})
        )
        rr = self.calculate_risk_reward(price, stop_loss, target_1)

        ema_50_str = f"{ema_50:.2f}" if ema_50 is not None else "N/A"
        reasons = [
            f"RSI oversold at {rsi_val:.1f} — historically bounces from this level",
            f"MACD {macd_sig.lower().replace('_', ' ')} confirmed",
            f"Price above 50-day EMA ({ema_50_str})",
            f"Volume {int((vol_ratio - 1) * 100)}% above 20-day average",
        ]
        return self._build_signal(
            symbol=symbol,
            signal="BUY",
            strength=self._BASE_STRENGTH["RSI_OVERSOLD_BOUNCE"],
            entry_price=price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            rr=rr,
            reasons=reasons,
            inds=inds,
            strategy="RSI_OVERSOLD_BOUNCE",
        )

    def check_ema_pullback(
        self,
        df: Any,
        symbol: str,
        inds: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """BUY — healthy pullback to 20-day EMA in a confirmed uptrend.

        Conditions (ALL required):
        - Price within 1% of EMA_20  (touching the EMA)
        - Price above 50-day EMA     (uptrend intact)
        - RSI between 40–55          (not overbought, not panic-selling)
        - Volume < 1.0× average      (quiet pullback, not distribution)
        """
        if inds is None:
            inds = self.ti.generate_full_analysis(df, symbol).get("indicators", {})

        rsi    = inds.get("rsi", {})
        ema    = inds.get("ema", {})
        volume = inds.get("volume", {})
        atr    = inds.get("atr", {})

        if "error" in rsi or "error" in ema:
            _log.debug("%s EMA_PULLBACK: indicator error — skipping", symbol)
            return None

        price        = float(df["close"].iloc[-1])
        rsi_val      = rsi.get("value", 50.0)
        ema_20       = ema.get("ema_20")
        ema_50       = ema.get("ema_50")
        price_vs_50  = ema.get("price_vs_ema_50", "BELOW")
        vol_ratio    = volume.get("volume_ratio", 1.0)
        atr_val      = atr.get("atr", 0.0)

        if ema_20 is None or ema_50 is None:
            _log.debug("%s EMA_PULLBACK: EMA values unavailable", symbol)
            return None

        pct_from_ema20 = abs(price - ema_20) / ema_20 * 100 if ema_20 > 0 else 99.0
        if pct_from_ema20 > 1.0:
            _log.debug(
                "%s EMA_PULLBACK: %.2f%% from EMA20 (need ≤1%%)", symbol, pct_from_ema20
            )
            return None
        if price_vs_50 != "ABOVE":
            _log.debug("%s EMA_PULLBACK: price below EMA50", symbol)
            return None
        if not (40.0 <= rsi_val <= 55.0):
            _log.debug(
                "%s EMA_PULLBACK: RSI %.1f not in [40, 55]", symbol, rsi_val
            )
            return None
        if vol_ratio >= 1.0:
            _log.debug(
                "%s EMA_PULLBACK: vol ratio %.2f ≥ 1.0 (not a quiet pullback)",
                symbol, vol_ratio,
            )
            return None

        stop_loss = self.calculate_stop_loss(
            "EMA_PULLBACK", price, atr_val, ema_50=ema_50
        )
        target_1, target_2 = self.calculate_targets(
            "EMA_PULLBACK", price, atr_val, inds.get("support_resistance", {})
        )
        rr = self.calculate_risk_reward(price, stop_loss, target_1)

        reasons = [
            f"Price pulling back to 20-day EMA ({ema_20:.2f}) — {pct_from_ema20:.1f}% away",
            f"Uptrend intact: price above 50-day EMA ({ema_50:.2f})",
            f"RSI at {rsi_val:.1f} — not overbought, not panic selling",
            f"Volume ratio {vol_ratio:.2f} below average — healthy pullback",
        ]
        return self._build_signal(
            symbol=symbol,
            signal="BUY",
            strength=self._BASE_STRENGTH["EMA_PULLBACK"],
            entry_price=price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            rr=rr,
            reasons=reasons,
            inds=inds,
            strategy="EMA_PULLBACK",
        )

    def check_volume_breakout(
        self,
        df: Any,
        symbol: str,
        inds: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """BUY — price breaks above resistance_1 with strong volume conviction.

        Conditions (ALL required):
        - Price > resistance_1  (confirmed breakout)
        - Volume ≥ 2× average   (institutional conviction)
        - RSI < 65               (not already overbought)
        - MACD above signal line (positive momentum)
        """
        if inds is None:
            inds = self.ti.generate_full_analysis(df, symbol).get("indicators", {})

        rsi    = inds.get("rsi", {})
        macd   = inds.get("macd", {})
        volume = inds.get("volume", {})
        sr     = inds.get("support_resistance", {})
        atr    = inds.get("atr", {})

        if "error" in rsi or "error" in macd or "error" in sr:
            _log.debug("%s VOLUME_BREAKOUT: indicator error — skipping", symbol)
            return None

        price        = float(df["close"].iloc[-1])
        rsi_val      = rsi.get("value", 50.0)
        macd_sig     = macd.get("signal", "")
        vol_ratio    = volume.get("volume_ratio", 0.0)
        resistance_1 = sr.get("resistance_1")
        resistance_2 = sr.get("resistance_2")
        atr_val      = atr.get("atr", 0.0)

        if resistance_1 is None:
            _log.debug("%s VOLUME_BREAKOUT: no resistance level available", symbol)
            return None
        if price <= resistance_1:
            _log.debug(
                "%s VOLUME_BREAKOUT: price %.2f not above resistance %.2f",
                symbol, price, resistance_1,
            )
            return None
        if vol_ratio < 2.0:
            _log.debug(
                "%s VOLUME_BREAKOUT: vol ratio %.2f < 2.0", symbol, vol_ratio
            )
            return None
        if rsi_val >= 65.0:
            _log.debug(
                "%s VOLUME_BREAKOUT: RSI %.1f ≥ 65 (already overbought)", symbol, rsi_val
            )
            return None
        if macd_sig not in ("BULLISH_CROSSOVER", "BULLISH"):
            _log.debug(
                "%s VOLUME_BREAKOUT: MACD not bullish (%s)", symbol, macd_sig
            )
            return None

        # Stop-loss: just below broken resistance (now support) — resistance_1 − 0.5%
        stop_loss = round(resistance_1 * 0.995, 2)
        # Target: next resistance level, or ATR-based if unavailable
        target_1 = (
            resistance_2
            if resistance_2 and resistance_2 > price
            else round(price + 3 * atr_val, 2)
        )
        target_2 = round(price + 5 * atr_val, 2)
        if target_2 <= target_1:
            target_2 = round(target_1 * 1.02, 2)
        rr = self.calculate_risk_reward(price, stop_loss, target_1)

        reasons = [
            f"Price {price:.2f} broke above resistance {resistance_1:.2f}",
            f"Volume {int((vol_ratio - 1) * 100)}% above average — strong conviction",
            f"RSI at {rsi_val:.1f} — not overbought at breakout",
            f"MACD {macd_sig.lower().replace('_', ' ')} — positive momentum",
        ]
        return self._build_signal(
            symbol=symbol,
            signal="BUY",
            strength=self._BASE_STRENGTH["VOLUME_BREAKOUT"],
            entry_price=price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            rr=rr,
            reasons=reasons,
            inds=inds,
            strategy="VOLUME_BREAKOUT",
        )

    def check_trend_following(
        self,
        df: Any,
        symbol: str,
        inds: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """BUY — ride an established trend when momentum is healthy.

        Conditions (ALL required):
        - Price > EMA_20 > EMA_50  (EMAs properly stacked)
        - MACD histogram > 0       (positive momentum)
        - RSI between 50–65        (momentum without being overbought)
        - ATR% > 2%                (enough volatility for a swing trade)
        """
        if inds is None:
            inds = self.ti.generate_full_analysis(df, symbol).get("indicators", {})

        rsi  = inds.get("rsi", {})
        macd = inds.get("macd", {})
        ema  = inds.get("ema", {})
        atr  = inds.get("atr", {})

        if "error" in rsi or "error" in macd or "error" in ema or "error" in atr:
            _log.debug("%s TREND_FOLLOWING: indicator error — skipping", symbol)
            return None

        price        = float(df["close"].iloc[-1])
        rsi_val      = rsi.get("value", 50.0)
        ema_20       = ema.get("ema_20")
        ema_50       = ema.get("ema_50")
        price_vs_20  = ema.get("price_vs_ema_20", "BELOW")
        price_vs_50  = ema.get("price_vs_ema_50", "BELOW")
        macd_hist    = macd.get("histogram", 0.0)
        atr_val      = atr.get("atr", 0.0)
        atr_pct      = atr.get("atr_pct", 0.0)

        if ema_20 is None or ema_50 is None:
            _log.debug("%s TREND_FOLLOWING: EMA values unavailable", symbol)
            return None
        if price_vs_20 != "ABOVE" or price_vs_50 != "ABOVE":
            _log.debug("%s TREND_FOLLOWING: price not above both EMAs", symbol)
            return None
        if ema_20 <= ema_50:
            _log.debug(
                "%s TREND_FOLLOWING: EMA20 %.2f ≤ EMA50 %.2f — not stacked",
                symbol, ema_20, ema_50,
            )
            return None
        if macd_hist <= 0:
            _log.debug(
                "%s TREND_FOLLOWING: MACD histogram %.4f ≤ 0", symbol, macd_hist
            )
            return None
        if not (50.0 <= rsi_val <= 65.0):
            _log.debug(
                "%s TREND_FOLLOWING: RSI %.1f not in [50, 65]", symbol, rsi_val
            )
            return None
        if atr_pct < 2.0:
            _log.debug(
                "%s TREND_FOLLOWING: ATR%% %.2f < 2%% — insufficient volatility",
                symbol, atr_pct,
            )
            return None

        # Stop-loss: below 20-day EMA
        stop_loss = self.calculate_stop_loss(
            "TREND_FOLLOWING", price, atr_val, ema_20=ema_20
        )
        target_1, target_2 = self.calculate_targets(
            "TREND_FOLLOWING", price, atr_val, inds.get("support_resistance", {})
        )
        rr = self.calculate_risk_reward(price, stop_loss, target_1)

        reasons = [
            (
                f"EMAs properly stacked: price ({price:.2f}) "
                f"> EMA20 ({ema_20:.2f}) > EMA50 ({ema_50:.2f})"
            ),
            f"MACD histogram positive at {macd_hist:.4f} — growing momentum",
            f"RSI at {rsi_val:.1f} — momentum zone without being overbought",
            f"ATR {atr_pct:.1f}% — sufficient volatility for a swing trade",
        ]
        return self._build_signal(
            symbol=symbol,
            signal="BUY",
            strength=self._BASE_STRENGTH["TREND_FOLLOWING"],
            entry_price=price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            rr=rr,
            reasons=reasons,
            inds=inds,
            strategy="TREND_FOLLOWING",
        )

    def check_exit_signals(
        self,
        df: Any,
        symbol: str,
        current_holdings: List[str],
        inds: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """SELL — exit a held position when momentum deteriorates.

        Triggers (any combination):
        - MACD bearish crossover           (+0.40 strength)
        - RSI > 70  (overbought)           (+0.35 strength)
        - Price below EMA_20 + high volume (+0.35 strength)
        """
        if inds is None:
            inds = self.ti.generate_full_analysis(df, symbol).get("indicators", {})

        rsi    = inds.get("rsi", {})
        macd   = inds.get("macd", {})
        ema    = inds.get("ema", {})
        volume = inds.get("volume", {})
        atr    = inds.get("atr", {})

        if "error" in rsi or "error" in macd:
            _log.debug("%s EXIT: indicator error — skipping", symbol)
            return None

        is_held      = symbol in current_holdings
        price        = float(df["close"].iloc[-1])
        rsi_val      = rsi.get("value", 50.0)
        macd_sig     = macd.get("signal", "")
        price_vs_20  = ema.get("price_vs_ema_20", "ABOVE")
        vol_ratio    = volume.get("volume_ratio", 0.0)
        atr_val      = atr.get("atr", 0.0)

        exit_reasons: List[str] = []
        exit_strength = 0.0

        if macd_sig == "BEARISH_CROSSOVER":
            exit_reasons.append("MACD bearish crossover — momentum turning down")
            exit_strength += 0.40

        if rsi_val > 70.0:
            exit_reasons.append(
                f"RSI overbought at {rsi_val:.1f} — time to take profits"
            )
            exit_strength += 0.35

        if price_vs_20 == "BELOW" and vol_ratio >= 1.5:
            exit_reasons.append(
                f"Price broke below 20-day EMA with volume "
                f"{int((vol_ratio - 1) * 100)}% above average"
            )
            exit_strength += 0.35

        if not exit_reasons:
            _log.debug("%s EXIT: no exit conditions met", symbol)
            return None

        # For un-held stocks only return SELL if conditions are overwhelmingly strong
        if not is_held and exit_strength < 0.70:
            _log.debug(
                "%s EXIT: not held and exit_strength %.2f < 0.70 — suppressed",
                symbol, exit_strength,
            )
            return None

        return self._build_signal(
            symbol=symbol,
            signal="SELL",
            strength=min(1.0, exit_strength),
            entry_price=price,
            stop_loss=round(price + 2 * atr_val, 2),  # informational for SELL
            target_1=round(price - 2 * atr_val, 2),
            target_2=round(price - 4 * atr_val, 2),
            rr=1.0,  # not applicable to exits
            reasons=exit_reasons,
            inds=inds,
            strategy="EXIT_SIGNAL",
        )

    # ------------------------------------------------------------------
    # Risk management helpers
    # ------------------------------------------------------------------

    def calculate_stop_loss(
        self,
        strategy: str,
        price: float,
        atr: float,
        ema_50: Optional[float] = None,
        ema_20: Optional[float] = None,
        resistance_1: Optional[float] = None,
    ) -> float:
        """Return the stop-loss price for *strategy*.

        Args:
            strategy:     Strategy name (e.g. ``"RSI_OVERSOLD_BOUNCE"``).
            price:        Current entry price.
            atr:          14-day ATR value.
            ema_50:       50-day EMA — used by EMA_PULLBACK.
            ema_20:       20-day EMA — used by TREND_FOLLOWING.
            resistance_1: Nearest resistance — used by VOLUME_BREAKOUT.
        """
        if strategy == "RSI_OVERSOLD_BOUNCE":
            return round(price - 2 * atr, 2)
        if strategy == "EMA_PULLBACK":
            sl_atr  = price - 2 * atr
            sl_ema50 = ema_50 * 0.995 if ema_50 is not None else sl_atr
            return round(min(sl_atr, sl_ema50), 2)
        if strategy == "VOLUME_BREAKOUT":
            return round(resistance_1 * 0.995, 2) if resistance_1 is not None else round(price - 2 * atr, 2)
        if strategy == "TREND_FOLLOWING":
            return round(ema_20, 2) if ema_20 is not None else round(price - 2 * atr, 2)
        return round(price - 2 * atr, 2)

    def calculate_targets(
        self,
        strategy: str,
        price: float,
        atr: float,
        resistance_levels: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Return ``(target_1, target_2)`` for *strategy*.

        Falls back to ATR multiples when resistance levels are unavailable.
        """
        r1 = resistance_levels.get("resistance_1")
        r2 = resistance_levels.get("resistance_2")

        if strategy == "RSI_OVERSOLD_BOUNCE":
            t1 = round(price + 3 * atr, 2)
            t2 = round(price + 5 * atr, 2)
        elif strategy == "EMA_PULLBACK":
            t1 = r1 if (r1 and r1 > price) else round(price + 2.5 * atr, 2)
            t2 = r2 if (r2 and r2 > price) else round(price + 4.0 * atr, 2)
        elif strategy == "VOLUME_BREAKOUT":
            t1 = r2 if (r2 and r2 > price) else round(price + 3 * atr, 2)
            t2 = round(price + 5 * atr, 2)
        elif strategy == "TREND_FOLLOWING":
            t1 = round(price + 3 * atr, 2)
            t2 = round(price + 5 * atr, 2)
        else:
            t1 = round(price + 3 * atr, 2)
            t2 = round(price + 5 * atr, 2)

        if t2 <= t1:
            t2 = round(t1 * 1.02, 2)
        return t1, t2

    def calculate_risk_reward(
        self, entry: float, stop_loss: float, target: float
    ) -> float:
        """Return ``reward / risk`` ratio (≥ 1.0 required for any BUY trade).

        Returns 0.0 when risk is zero or negative to avoid division errors.
        """
        risk   = entry - stop_loss
        reward = target - entry
        if risk <= 0:
            return 0.0
        return round(reward / risk, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse_stock(
        self,
        symbol: str,
        holdings: Set[str],
        seen_buy: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Fetch data → run all strategies → return best signal or None."""
        if symbol in seen_buy:
            _log.debug("%s: already has an active BUY signal this scan — skipping", symbol)
            return None

        df = self._fetch_daily_data(symbol)
        if df is None or len(df) < 30:
            rows = len(df) if df is not None else 0
            _log.debug("%s: insufficient data (%d rows) — skipping", symbol, rows)
            return None

        analysis = self.ti.generate_full_analysis(df, symbol)
        if "error" in analysis:
            _log.debug("%s: analysis error (%s) — skipping", symbol, analysis["error"])
            return None

        inds = analysis.get("indicators", {})

        # --- Collect BUY signals (only for stocks not currently held) ---
        buy_signals: List[Dict[str, Any]] = []
        if symbol not in holdings:
            for checker in (
                self.check_rsi_oversold_bounce,
                self.check_ema_pullback,
                self.check_volume_breakout,
                self.check_trend_following,
            ):
                result = checker(df, symbol, inds)
                if result is not None:
                    buy_signals.append(result)

        # --- Check SELL signals (for held stocks) ---
        sell_signal: Optional[Dict[str, Any]] = None
        if symbol in holdings:
            sell_signal = self.check_exit_signals(df, symbol, list(holdings), inds)

        # --- Return the best signal ---
        if buy_signals:
            combined = self._combine_buy_signals(buy_signals)
            rr = combined.get("risk_reward_ratio", 0.0)
            if rr > 1.0:
                _log.info(
                    "BUY  | %-10s | strategy=%-30s | strength=%.2f | rr=%.2f",
                    symbol,
                    combined.get("strategy_name", ""),
                    combined.get("strength", 0.0),
                    rr,
                )
                return combined
            _log.debug(
                "%s: BUY rejected — risk:reward %.2f ≤ 1.0", symbol, rr
            )

        if sell_signal is not None:
            _log.info(
                "SELL | %-10s | strategy=%-30s | strength=%.2f",
                symbol,
                sell_signal.get("strategy_name", ""),
                sell_signal.get("strength", 0.0),
            )
            return sell_signal

        _log.debug("HOLD | %s — no actionable signal this scan", symbol)
        return None

    def _fetch_daily_data(self, symbol: str, candles: int = 55) -> Optional[Any]:
        """Fetch the last *candles* daily OHLCV bars for *symbol*."""
        try:
            to_dt   = datetime.now(IST)
            from_dt = to_dt - timedelta(days=candles + 25)  # buffer for holidays
            df = self.broker.get_historical_data(
                symbol=symbol,
                interval="ONE_DAY",
                from_date=from_dt.strftime("%Y-%m-%d %H:%M"),
                to_date=to_dt.strftime("%Y-%m-%d %H:%M"),
            )
            if df is not None and len(df) > 0:
                return df.tail(candles).reset_index(drop=True)
            return None
        except Exception as exc:
            _log.warning("Failed to fetch data for %s: %s", symbol, exc)
            return None

    def _combine_buy_signals(
        self, signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge multiple BUY signals into one, boosting strength.

        The strongest signal is primary; each additional confirming signal
        adds ``_STRENGTH_BOOST`` (capped at 1.0).
        """
        signals.sort(key=lambda s: s.get("strength", 0.0), reverse=True)
        primary = dict(signals[0])

        combined_strength = primary["strength"]
        for _ in signals[1:]:
            combined_strength = min(1.0, combined_strength + self._STRENGTH_BOOST)

        combined_reasons: List[str] = []
        for sig in signals:
            combined_reasons.extend(sig.get("reasons", []))

        strategy_name = " + ".join(s["strategy_name"] for s in signals)

        primary["strength"]      = round(combined_strength, 2)
        primary["strategy_name"] = strategy_name
        primary["reasons"]       = combined_reasons
        return primary

    def _build_signal(
        self,
        symbol: str,
        signal: str,
        strength: float,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        rr: float,
        reasons: List[str],
        inds: Dict[str, Any],
        strategy: str,
    ) -> Dict[str, Any]:
        """Build and return a standardised signal dict."""
        rsi_val    = inds.get("rsi",    {}).get("value", 0.0)
        macd_hist  = inds.get("macd",   {}).get("histogram", 0.0)
        atr_val    = inds.get("atr",    {}).get("atr", 0.0)
        vol_ratio  = inds.get("volume", {}).get("volume_ratio", 0.0)
        p_ema50    = inds.get("ema",    {}).get("price_vs_ema_50", "UNKNOWN")
        p_ema200   = inds.get("ema",    {}).get("price_vs_ema_200", "UNKNOWN")

        return {
            "symbol":           symbol,
            "signal":           signal,
            "strength":         round(strength, 2),
            "entry_price":      round(entry_price, 2),
            "stop_loss":        round(stop_loss, 2),
            "target_1":         round(target_1, 2),
            "target_2":         round(target_2, 2),
            "risk_reward_ratio": rr,
            "reasons":          reasons,
            "indicators": {
                "rsi":             rsi_val,
                "macd_histogram":  macd_hist,
                "atr":             atr_val,
                "volume_ratio":    vol_ratio,
                "price_vs_ema_50":  p_ema50,
                "price_vs_ema_200": p_ema200,
            },
            "strategy_name":    strategy,
            "timestamp":        datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _store_signal(self, signal: Dict[str, Any]) -> None:
        """Persist *signal* to the database (best-effort — never raises)."""
        try:
            db_signal = Signal(
                symbol=signal["symbol"],
                signal_type=signal["signal"],
                signal_source="QUANT",
                strength=signal.get("strength"),
                indicators=json.dumps(signal.get("indicators", {})),
            )
            self.db.record_signal(db_signal)
        except Exception as exc:
            _log.warning(
                "Failed to persist signal for %s: %s", signal["symbol"], exc
            )
