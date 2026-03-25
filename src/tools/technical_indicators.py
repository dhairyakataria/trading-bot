"""Technical Indicators — mathematical calculations on OHLCV DataFrames.

Pure Python + pandas + pandas_ta. No LLM calls.
All methods accept a DataFrame with columns [datetime, open, high, low, close, volume]
and return structured dicts that an LLM agent can interpret.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

_log = get_logger("tools.technical_indicators")


class TechnicalIndicators:
    """Computes technical indicators on OHLCV DataFrames.

    All methods:
    - Accept a pandas DataFrame with columns [datetime, open, high, low, close, volume]
    - Return a dict with indicator values, signals, and interpretations
    - Never raise — return a dict with an 'error' key on failure
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        try:
            import pandas_ta  # noqa: F401
            self._has_pandas_ta = True
        except ImportError:
            _log.warning("pandas_ta not installed — using fallback calculations")
            self._has_pandas_ta = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with lowercase columns, sorted by datetime."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        if "datetime" in df.columns:
            df = df.sort_values("datetime")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> dict:
        """Return RSI value, signal (OVERSOLD/OVERBOUGHT/NEUTRAL), and interpretation."""
        try:
            df = self._validate_df(df)
            if len(df) < period + 1:
                return {"error": f"Insufficient data: need {period + 1} rows, got {len(df)}"}

            if self._has_pandas_ta:
                import pandas_ta as ta
                rsi_series = ta.rsi(df["close"], length=period)
            else:
                delta = df["close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
                avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
                # avg_loss = 0 and avg_gain > 0 → RSI = 100 (all gains, no losses)
                # avg_gain = avg_loss = 0 → no movement → neutral RSI = 50
                rsi_values = np.where(
                    avg_loss == 0,
                    np.where(avg_gain == 0, 50.0, 100.0),
                    100.0 - 100.0 / (1.0 + avg_gain / avg_loss),
                )
                rsi_series = pd.Series(rsi_values, index=df.index)

            value = float(rsi_series.iloc[-1])
            if np.isnan(value):
                return {"error": "RSI calculation returned NaN — insufficient data"}

            if value < 30:
                signal = "OVERSOLD"
                interpretation = (
                    f"RSI at {value:.1f} indicates oversold condition — potential bounce"
                )
            elif value > 70:
                signal = "OVERBOUGHT"
                interpretation = (
                    f"RSI at {value:.1f} indicates overbought condition — potential pullback"
                )
            else:
                signal = "NEUTRAL"
                interpretation = f"RSI at {value:.1f} is in neutral territory"

            return {"value": round(value, 2), "signal": signal, "interpretation": interpretation}

        except Exception as exc:
            _log.error("RSI calculation failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    def calculate_macd(
        self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> dict:
        """Return MACD line, signal line, histogram, crossover signal, and interpretation."""
        try:
            df = self._validate_df(df)
            if len(df) < slow + signal:
                return {
                    "error": (
                        f"Insufficient data: need {slow + signal} rows, got {len(df)}"
                    )
                }

            if self._has_pandas_ta:
                import pandas_ta as ta
                macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
                if macd_df is None or macd_df.empty:
                    return {"error": "MACD returned empty result"}

                macd_col = next(c for c in macd_df.columns if c.startswith("MACD_"))
                signal_col = next(c for c in macd_df.columns if c.startswith("MACDs_"))
                hist_col = next(c for c in macd_df.columns if c.startswith("MACDh_"))

                macd_line = float(macd_df[macd_col].iloc[-1])
                signal_line = float(macd_df[signal_col].iloc[-1])
                histogram = float(macd_df[hist_col].iloc[-1])

                prev_macd_val = macd_df[macd_col].iloc[-2] if len(macd_df) >= 2 else np.nan
                prev_sig_val = macd_df[signal_col].iloc[-2] if len(macd_df) >= 2 else np.nan
                prev_macd = float(prev_macd_val) if not np.isnan(prev_macd_val) else macd_line
                prev_signal_line = (
                    float(prev_sig_val) if not np.isnan(prev_sig_val) else signal_line
                )
            else:
                ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
                ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
                macd_series = ema_fast - ema_slow
                signal_series = macd_series.ewm(span=signal, adjust=False).mean()
                hist_series = macd_series - signal_series

                macd_line = float(macd_series.iloc[-1])
                signal_line = float(signal_series.iloc[-1])
                histogram = float(hist_series.iloc[-1])
                prev_macd = float(macd_series.iloc[-2]) if len(macd_series) >= 2 else macd_line
                prev_signal_line = (
                    float(signal_series.iloc[-2]) if len(signal_series) >= 2 else signal_line
                )

            if any(np.isnan(v) for v in [macd_line, signal_line, histogram]):
                return {"error": "MACD calculation returned NaN"}

            crossed_above = prev_macd < prev_signal_line and macd_line > signal_line
            crossed_below = prev_macd > prev_signal_line and macd_line < signal_line

            if crossed_above:
                sig = "BULLISH_CROSSOVER"
                interp = "MACD crossed above signal line — bullish momentum"
            elif crossed_below:
                sig = "BEARISH_CROSSOVER"
                interp = "MACD crossed below signal line — bearish momentum"
            elif macd_line > signal_line:
                sig = "BULLISH"
                interp = "MACD above signal line — bullish trend"
            else:
                sig = "BEARISH"
                interp = "MACD below signal line — bearish trend"

            return {
                "macd_line": round(macd_line, 4),
                "signal_line": round(signal_line, 4),
                "histogram": round(histogram, 4),
                "signal": sig,
                "interpretation": interp,
            }

        except Exception as exc:
            _log.error("MACD calculation failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    def calculate_bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> dict:
        """Return Bollinger Bands upper/middle/lower, current price, signal, interpretation."""
        try:
            df = self._validate_df(df)
            if len(df) < period:
                return {
                    "error": f"Insufficient data: need {period} rows, got {len(df)}"
                }

            if self._has_pandas_ta:
                import pandas_ta as ta
                bb_df = ta.bbands(df["close"], length=period, std=std_dev)
                if bb_df is None or bb_df.empty:
                    return {"error": "Bollinger Bands returned empty result"}

                lower_col = next(c for c in bb_df.columns if c.startswith("BBL_"))
                middle_col = next(c for c in bb_df.columns if c.startswith("BBM_"))
                upper_col = next(c for c in bb_df.columns if c.startswith("BBU_"))

                lower = float(bb_df[lower_col].iloc[-1])
                middle = float(bb_df[middle_col].iloc[-1])
                upper = float(bb_df[upper_col].iloc[-1])
            else:
                rolling = df["close"].rolling(window=period)
                middle = float(rolling.mean().iloc[-1])
                std = float(rolling.std().iloc[-1])
                upper = middle + std_dev * std
                lower = middle - std_dev * std

            current_price = float(df["close"].iloc[-1])
            band_width = upper - lower
            pct_from_lower = (current_price - lower) / band_width if band_width > 0 else 0.5

            if pct_from_lower <= 0.1:
                sig = "NEAR_LOWER_BAND"
                interp = "Price near lower Bollinger Band — potential bounce"
            elif pct_from_lower >= 0.9:
                sig = "NEAR_UPPER_BAND"
                interp = "Price near upper Bollinger Band — potential reversal"
            elif pct_from_lower > 0.5:
                sig = "ABOVE_MIDDLE"
                interp = "Price above middle Bollinger Band — mild bullish"
            else:
                sig = "BELOW_MIDDLE"
                interp = "Price below middle Bollinger Band — mild bearish"

            return {
                "upper": round(upper, 2),
                "middle": round(middle, 2),
                "lower": round(lower, 2),
                "current_price": round(current_price, 2),
                "signal": sig,
                "interpretation": interp,
            }

        except Exception as exc:
            _log.error("Bollinger Bands calculation failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def calculate_ema(
        self, df: pd.DataFrame, periods: list[int] | None = None
    ) -> dict:
        """Return EMA values for multiple periods with price-vs-EMA signals."""
        if periods is None:
            periods = [20, 50, 200]
        try:
            df = self._validate_df(df)
            current_price = float(df["close"].iloc[-1])
            result: dict[str, Any] = {}

            for p in periods:
                if len(df) < p:
                    result[f"ema_{p}"] = None
                    result[f"price_vs_ema_{p}"] = "INSUFFICIENT_DATA"
                    continue

                if self._has_pandas_ta:
                    import pandas_ta as ta
                    ema_series = ta.ema(df["close"], length=p)
                else:
                    ema_series = df["close"].ewm(span=p, adjust=False).mean()

                ema_val = float(ema_series.iloc[-1])
                result[f"ema_{p}"] = round(ema_val, 2)
                result[f"price_vs_ema_{p}"] = "ABOVE" if current_price > ema_val else "BELOW"

            valid_positions = [
                result[f"price_vs_ema_{p}"]
                for p in periods
                if result.get(f"price_vs_ema_{p}") in ("ABOVE", "BELOW")
            ]
            above_count = sum(1 for v in valid_positions if v == "ABOVE")
            total = len(valid_positions)

            if total == 0:
                sig = "UNKNOWN"
                interp = "Insufficient data to determine EMA trend"
            elif above_count == total:
                sig = "STRONG_UPTREND"
                interp = "Price above all major EMAs — strong uptrend"
            elif above_count >= total * 0.67:
                sig = "UPTREND"
                interp = "Price above most EMAs — uptrend"
            elif above_count == 0:
                sig = "STRONG_DOWNTREND"
                interp = "Price below all major EMAs — strong downtrend"
            elif above_count <= total * 0.33:
                sig = "DOWNTREND"
                interp = "Price below most EMAs — downtrend"
            else:
                sig = "MIXED"
                interp = "Price mixed relative to EMAs — no clear trend"

            result["signal"] = sig
            result["interpretation"] = interp
            return result

        except Exception as exc:
            _log.error("EMA calculation failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------

    def calculate_vwap(self, df: pd.DataFrame) -> dict:
        """Return VWAP value and ABOVE/BELOW signal (for intraday data)."""
        try:
            df = self._validate_df(df)
            if len(df) < 2:
                return {"error": "Insufficient data for VWAP calculation"}

            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            total_volume = df["volume"].sum()
            if total_volume == 0:
                return {"error": "Total volume is zero — cannot compute VWAP"}

            vwap_val = float((typical_price * df["volume"]).sum() / total_volume)
            current_price = float(df["close"].iloc[-1])

            if current_price > vwap_val:
                sig = "ABOVE_VWAP"
                interp = "Price above VWAP — bullish intraday sentiment"
            else:
                sig = "BELOW_VWAP"
                interp = "Price below VWAP — bearish intraday sentiment"

            return {
                "vwap": round(vwap_val, 2),
                "current_price": round(current_price, 2),
                "signal": sig,
                "interpretation": interp,
            }

        except Exception as exc:
            _log.error("VWAP calculation failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> dict:
        """Return ATR value, ATR as % of price, and volatility interpretation."""
        try:
            df = self._validate_df(df)
            if len(df) < period + 1:
                return {
                    "error": f"Insufficient data: need {period + 1} rows, got {len(df)}"
                }

            if self._has_pandas_ta:
                import pandas_ta as ta
                atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
            else:
                high_low = df["high"] - df["low"]
                high_prev_close = (df["high"] - df["close"].shift(1)).abs()
                low_prev_close = (df["low"] - df["close"].shift(1)).abs()
                tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
                atr_series = tr.ewm(com=period - 1, min_periods=period).mean()

            atr_val = float(atr_series.iloc[-1])
            current_price = float(df["close"].iloc[-1])
            atr_pct = round(atr_val / current_price * 100, 2) if current_price > 0 else 0.0

            if atr_pct < 1.0:
                interp = "Low volatility — suitable for tight stop-losses"
            elif atr_pct < 2.5:
                interp = "Moderate volatility — suitable for swing trading"
            else:
                interp = "High volatility — wider stops required, reduce position size"

            return {"atr": round(atr_val, 2), "atr_pct": atr_pct, "interpretation": interp}

        except Exception as exc:
            _log.error("ATR calculation failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Volume Analysis
    # ------------------------------------------------------------------

    def calculate_volume_analysis(self, df: pd.DataFrame, period: int = 20) -> dict:
        """Return current volume vs. average and a HIGH/NORMAL/LOW signal."""
        try:
            df = self._validate_df(df)
            if len(df) < period:
                return {
                    "error": f"Insufficient data: need {period} rows, got {len(df)}"
                }

            current_volume = float(df["volume"].iloc[-1])
            avg_volume = float(df["volume"].tail(period).mean())

            if avg_volume == 0:
                return {"error": "Average volume is zero"}

            volume_ratio = round(current_volume / avg_volume, 2)

            if volume_ratio >= 1.5:
                sig = "HIGH_VOLUME"
                interp = (
                    f"Volume {int((volume_ratio - 1) * 100)}% above average "
                    "— confirms price move"
                )
            elif volume_ratio <= 0.5:
                sig = "LOW_VOLUME"
                interp = (
                    f"Volume {int((1 - volume_ratio) * 100)}% below average "
                    "— weak conviction"
                )
            else:
                sig = "NORMAL_VOLUME"
                interp = "Volume near average — no strong conviction signal"

            return {
                "current_volume": int(current_volume),
                "avg_volume": int(avg_volume),
                "volume_ratio": volume_ratio,
                "signal": sig,
                "interpretation": interp,
            }

        except Exception as exc:
            _log.error("Volume analysis failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Support / Resistance
    # ------------------------------------------------------------------

    def calculate_support_resistance(
        self, df: pd.DataFrame, lookback: int = 60
    ) -> dict:
        """Return pivot-based support and resistance levels."""
        try:
            df = self._validate_df(df)
            data = df.tail(lookback).reset_index(drop=True)
            if len(data) < 10:
                return {
                    "error": f"Insufficient data: need 10 rows, got {len(data)}"
                }

            current_price = float(df["close"].iloc[-1])
            highs = data["high"].values
            lows = data["low"].values

            pivot_highs: list[float] = []
            pivot_lows: list[float] = []
            for i in range(2, len(data) - 2):
                if (
                    highs[i] > highs[i - 1]
                    and highs[i] > highs[i - 2]
                    and highs[i] > highs[i + 1]
                    and highs[i] > highs[i + 2]
                ):
                    pivot_highs.append(float(highs[i]))
                if (
                    lows[i] < lows[i - 1]
                    and lows[i] < lows[i - 2]
                    and lows[i] < lows[i + 1]
                    and lows[i] < lows[i + 2]
                ):
                    pivot_lows.append(float(lows[i]))

            # Fall back to top/bottom values if no pivots found
            if not pivot_highs:
                pivot_highs = sorted(highs.tolist(), reverse=True)[:3]
            if not pivot_lows:
                pivot_lows = sorted(lows.tolist())[:3]

            # Resistances: pivot highs above current price (nearest first)
            resistances = sorted(
                [h for h in pivot_highs if h > current_price]
            ) or sorted(pivot_highs, reverse=True)[:2]

            # Supports: pivot lows below current price (nearest first, i.e. highest first)
            supports = sorted(
                [l for l in pivot_lows if l < current_price], reverse=True
            ) or sorted(pivot_lows)[:2]

            r1 = round(resistances[0], 2) if resistances else round(float(data["high"].max()), 2)
            r2 = round(resistances[1], 2) if len(resistances) >= 2 else round(r1 * 1.02, 2)
            s1 = round(supports[0], 2) if supports else round(float(data["low"].min()), 2)
            s2 = round(supports[1], 2) if len(supports) >= 2 else round(s1 * 0.98, 2)

            return {
                "support_1": s1,
                "support_2": s2,
                "resistance_1": r1,
                "resistance_2": r2,
                "current_price": round(current_price, 2),
            }

        except Exception as exc:
            _log.error("Support/resistance calculation failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Full Analysis
    # ------------------------------------------------------------------

    def generate_full_analysis(self, df: pd.DataFrame, symbol: str) -> dict:
        """Run all indicators and return a comprehensive analysis report.

        Scoring (Score > 3 → BUY, < -3 → SELL, else HOLD):
          RSI oversold = +2, overbought = -2
          MACD bullish crossover = +2, bearish crossover = -2
          Price above EMA = +1 per EMA (20, 50, 200)
          High volume confirming direction = +1 (bullish) or -1 (bearish)
          Near support_1 (within 2%) = +1
          Near resistance_1 (within 2%) = -1
        """
        try:
            df_val = self._validate_df(df)
            current_price = float(df_val["close"].iloc[-1])

            ts = (
                str(df_val["datetime"].iloc[-1])
                if "datetime" in df_val.columns
                else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            # --- Run all indicators ---
            rsi = self.calculate_rsi(df)
            macd = self.calculate_macd(df)
            bollinger = self.calculate_bollinger_bands(df)
            ema = self.calculate_ema(df)
            vwap = self.calculate_vwap(df)
            atr = self.calculate_atr(df)
            volume = self.calculate_volume_analysis(df)
            sr = self.calculate_support_resistance(df)

            # --- Scoring (phase 1: everything except volume) ---
            score = 0
            reasons: list[str] = []

            # RSI
            rsi_sig = rsi.get("signal", "")
            if rsi_sig == "OVERSOLD":
                score += 2
                reasons.append(f"RSI oversold at {rsi.get('value', 'N/A')}")
            elif rsi_sig == "OVERBOUGHT":
                score -= 2
                reasons.append(f"RSI overbought at {rsi.get('value', 'N/A')}")

            # MACD
            macd_sig = macd.get("signal", "")
            if macd_sig == "BULLISH_CROSSOVER":
                score += 2
                reasons.append("MACD bullish crossover")
            elif macd_sig == "BEARISH_CROSSOVER":
                score -= 2
                reasons.append("MACD bearish crossover")

            # EMA — +1 per period where price is above
            for p in [20, 50, 200]:
                if ema.get(f"price_vs_ema_{p}") == "ABOVE":
                    score += 1
            ema_sig = ema.get("signal", "")
            if ema_sig in ("STRONG_UPTREND", "UPTREND"):
                reasons.append(f"Price {ema_sig.lower().replace('_', ' ')} (EMAs)")
            elif ema_sig in ("STRONG_DOWNTREND", "DOWNTREND"):
                reasons.append(f"Price {ema_sig.lower().replace('_', ' ')} (EMAs)")

            # Support / Resistance proximity
            if "error" not in sr:
                if abs(current_price - sr["support_1"]) / current_price * 100 <= 2:
                    score += 1
                    reasons.append(f"Price near support at {sr['support_1']}")
                if abs(sr["resistance_1"] - current_price) / current_price * 100 <= 2:
                    score -= 1
                    reasons.append(f"Price near resistance at {sr['resistance_1']}")

            # Bollinger (informational in reasons only — not scored)
            bb_sig = bollinger.get("signal", "")
            if bb_sig == "NEAR_LOWER_BAND":
                reasons.append("Price near Bollinger lower band")
            elif bb_sig == "NEAR_UPPER_BAND":
                reasons.append("Price near Bollinger upper band")

            # --- Volume (phase 2: direction-aware) ---
            if volume.get("signal") == "HIGH_VOLUME":
                vol_ratio = volume.get("volume_ratio", 1.0)
                pct_above = int((vol_ratio - 1) * 100)
                if score > 0:
                    score += 1
                    reasons.append(
                        f"Volume {pct_above}% above average confirming move"
                    )
                elif score < 0:
                    score -= 1
                    reasons.append(
                        f"High volume {pct_above}% confirming bearish move"
                    )

            # --- Overall signal ---
            # Max achievable positive score:
            #   RSI(2) + MACD_crossover(2) + 3 EMAs(3) + volume(1) + support(1) = 9
            _MAX_SCORE = 9

            if score > 3:
                overall_signal = "BUY"
            elif score < -3:
                overall_signal = "SELL"
            else:
                overall_signal = "HOLD"

            signal_strength = round(min(abs(score) / _MAX_SCORE, 1.0), 2)

            return {
                "symbol": symbol,
                "timestamp": ts,
                "price": round(current_price, 2),
                "indicators": {
                    "rsi": rsi,
                    "macd": macd,
                    "bollinger": bollinger,
                    "ema": ema,
                    "vwap": vwap,
                    "atr": atr,
                    "volume": volume,
                    "support_resistance": sr,
                },
                "overall_signal": overall_signal,
                "signal_strength": signal_strength,
                "score": score,
                "reasons": reasons,
            }

        except Exception as exc:
            _log.error("Full analysis failed for %s: %s", symbol, exc)
            return {
                "symbol": symbol,
                "error": str(exc),
                "overall_signal": "HOLD",
                "signal_strength": 0.0,
                "score": 0,
                "reasons": [],
            }
