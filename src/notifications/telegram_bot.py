"""Telegram Bot — sends trade alerts, daily reports, and interactive approvals.

Supports three interaction patterns:

1. **Send-only** (paper / auto modes): Fire-and-forget alerts, no reply expected.
2. **Interactive approval** (approval mode): Sends a message with inline
   Yes / No buttons and blocks until the user taps one or the timeout expires.
3. **Reports**: Formatted morning briefings, EOD summaries, weekly reviews.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

_log = logging.getLogger("notifications.telegram")


def _cfg(config: Any, *keys: str, default: Any = None) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(*keys, default=default)
        obj: Any = config
        for k in keys:
            obj = obj[k]
        return obj
    except (KeyError, TypeError, AttributeError):
        return default


class TelegramNotifier:
    """Sends notifications to a Telegram chat using python-telegram-bot.

    Responsibilities:
    - Send plain-text and Markdown-formatted messages to a configured chat
    - Send trade entry / exit alerts with key details
    - Send morning briefings and daily / weekly performance summaries
    - Handle Telegram API rate-limits and errors gracefully (never crash)
    """

    _LEVEL_EMOJI: dict[str, str] = {
        "DEBUG":    "🔍",
        "INFO":     "ℹ️",
        "WARNING":  "⚠️",
        "ERROR":    "🚨",
        "CRITICAL": "🔴🔴🔴",
    }

    def __init__(self, config: Any) -> None:
        self.config = config
        self._bot_token: str  = _cfg(config, "notifications", "telegram", "bot_token", default="")
        self._chat_id: str    = str(_cfg(config, "notifications", "telegram", "chat_id", default=""))
        self._enabled: bool   = bool(_cfg(config, "notifications", "telegram", "enabled", default=True))

        if not self._bot_token or not self._chat_id:
            _log.warning("Telegram not configured (missing token/chat_id) — notifications disabled")
            self._enabled = False

    # ── Internal async helpers ─────────────────────────────────────────────────

    async def _async_send(self, text: str, parse_mode: str = "Markdown") -> None:
        """Async send — wraps python-telegram-bot Bot.send_message."""
        from telegram import Bot
        from telegram.error import RetryAfter, TelegramError

        async with Bot(token=self._bot_token) as bot:
            try:
                await bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
            except RetryAfter as exc:
                # Rate-limited: wait the required time and retry once
                wait = exc.retry_after + 1
                _log.warning("Telegram rate-limited — retrying in %ds", wait)
                await asyncio.sleep(wait)
                await bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
            except TelegramError as exc:
                raise  # Re-raise for the sync wrapper to catch and log

    def _send(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Synchronous wrapper: runs the async send and handles all errors."""
        if not self._enabled:
            return False
        try:
            asyncio.run(self._async_send(text, parse_mode))
            return True
        except Exception as exc:
            _log.error("Telegram send failed: %s", exc)
            return False

    # ── Public API ─────────────────────────────────────────────────────────────

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a plain text or Markdown-formatted message.

        Returns True if sent successfully, False otherwise.
        """
        return self._send(text, parse_mode)

    def send_trade_alert(
        self,
        trade_type: str,
        symbol: str,
        quantity: int,
        price: float,
        details: str = "",
    ) -> bool:
        """Send a formatted trade entry notification.

        Example output::

            🟢 BUY EXECUTED
            ----------------------------
            Stock: TCS
            Qty: 8 shares
            Price: ₹3,852.00
            Investment: ₹30,816.00
            Strategy: RSI_OVERSOLD_BOUNCE
        """
        is_buy     = trade_type.upper() == "BUY"
        emoji      = "🟢" if is_buy else "🔴"
        action     = "BUY EXECUTED" if is_buy else "SELL EXECUTED"
        investment = quantity * price

        lines = [
            f"{emoji} *{action}*",
            "----------------------------",
            f"Stock: {symbol}",
            f"Qty: {quantity} shares",
            f"Price: ₹{price:,.2f}",
            f"Investment: ₹{investment:,.2f}",
        ]
        if details:
            lines.append(details)

        return self._send("\n".join(lines))

    def send_exit_alert(
        self,
        symbol: str,
        exit_type: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
    ) -> bool:
        """Send a formatted trade exit notification.

        Example output::

            🔴 SELL EXECUTED
            ----------------------------
            Stock: TCS
            Exit Type: TARGET_HIT
            Entry: ₹3,852.00
            Exit: ₹3,965.00
            P&L: +₹904.00 (+2.93%)
        """
        sign = "+" if pnl >= 0 else ""
        lines = [
            "🔴 *SELL EXECUTED*",
            "----------------------------",
            f"Stock: {symbol}",
            f"Exit Type: {exit_type}",
            f"Entry: ₹{entry_price:,.2f}",
            f"Exit: ₹{exit_price:,.2f}",
            f"P&L: {sign}₹{pnl:,.2f} ({sign}{pnl_pct:.2f}%)",
        ]
        return self._send("\n".join(lines))

    def send_morning_briefing(self, briefing: dict) -> bool:
        """Send a formatted pre-market summary.

        Example output::

            🌅 MORNING BRIEFING — 16 Mar 2026
            -----------------------------------------
            Global Sentiment: NEUTRAL
            ...
        """
        today     = datetime.now().strftime("%d %b %Y")
        sentiment = briefing.get("global_sentiment", "NEUTRAL")
        outlook   = briefing.get("market_outlook", "")

        lines: list[str] = [
            f"🌅 *MORNING BRIEFING — {today}*",
            "-----------------------------------------",
            f"Global Sentiment: {sentiment}",
        ]

        # Market indices (S&P 500, Nasdaq, etc.)
        indices: dict = briefing.get("indices", {})
        if indices:
            lines.append(" | ".join(f"{k}: {v}" for k, v in list(indices.items())[:4]))

        # FII / DII flows
        fii = briefing.get("fii_activity", "")
        dii = briefing.get("dii_activity", "")
        if fii or dii:
            lines.append(f"FII: {fii} | DII: {dii}")

        # India VIX
        vix = briefing.get("india_vix", "")
        if vix:
            lines.append(f"India VIX: {vix}")

        # Crude / USD-INR
        crude  = briefing.get("crude_price", "")
        usdinr = briefing.get("usd_inr", "")
        if crude:
            lines.append(f"Crude: {crude}")
        if usdinr:
            lines.append(f"USD/INR: ₹{usdinr}")

        # Sector outlook
        sectors: dict = briefing.get("sector_outlook", {})
        if sectors:
            lines.append("")
            lines.append("📊 *Sectors to Watch:*")
            for sector, view in list(sectors.items())[:6]:
                lines.append(f"• {sector} — {view}")

        # Risky symbols flagged by research
        risky_syms: list = briefing.get("risky_symbols", [])
        if risky_syms:
            lines.append(f"⚠️ Caution: {', '.join(risky_syms[:5])}")

        if outlook:
            lines.append("")
            lines.append(f"📝 Outlook: {outlook[:200]}")

        lines.append("-----------------------------------------")
        return self._send("\n".join(lines))

    def send_daily_report(self, report: dict) -> bool:
        """Send the end-of-day performance summary.

        Example output::

            📊 DAILY REPORT — 16 Mar 2026
            -----------------------------------------
            Portfolio Value: ₹52,340
            Today's P&L: +₹340 (+0.65%)
            ...
        """
        today       = datetime.now().strftime("%d %b %Y")
        port_value  = report.get("portfolio_value", 0.0)
        pnl         = report.get("pnl", 0.0)
        pnl_pct     = report.get("pnl_pct", 0.0)
        buys        = report.get("buys", 0)
        sells       = report.get("sells", 0)
        wins        = report.get("wins", 0)
        losses      = report.get("losses", 0)
        holdings    = report.get("holdings", [])
        nifty_chg   = report.get("nifty_change_pct", 0.0)

        sign        = "+" if pnl >= 0 else ""
        nifty_sign  = "+" if nifty_chg >= 0 else ""
        beat_marker = "✅" if pnl_pct >= nifty_chg else "❌"

        lines: list[str] = [
            f"📊 *DAILY REPORT — {today}*",
            "-----------------------------------------",
            f"Portfolio Value: ₹{port_value:,.0f}",
            f"Today's P&L: {sign}₹{pnl:,.0f} ({sign}{pnl_pct:.2f}%)",
            "",
            f"Trades: {buys} buy, {sells} sell",
            f"Wins: {wins} | Losses: {losses}",
        ]

        if holdings:
            lines.append("")
            lines.append("*Holdings:*")
            for h in holdings[:8]:
                h_pnl  = h.get("pnl", 0.0)
                h_sign = "+" if h_pnl >= 0 else ""
                lines.append(
                    f"• {h['symbol']}: {h.get('quantity', 0)} shares"
                    f" @ ₹{h.get('avg_price', 0):,.0f}"
                    f" (P&L: {h_sign}₹{h_pnl:,.0f})"
                )

        lines += [
            "",
            f"Nifty 50: {nifty_sign}{nifty_chg:.2f}%",
            f"Your portfolio: {sign}{pnl_pct:.2f}% {beat_marker}",
            "-----------------------------------------",
        ]
        return self._send("\n".join(lines))

    def send_signal_alert(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
        strategy: str = "",
        reasoning: str = "",
        sector: str = "",
    ) -> bool:
        """Send a BUY signal advisory message (signal-only mode).

        Example output::

            📡 BUY SIGNAL — RELIANCE
            ----------------------------
            Qty:      12 shares
            Entry:    ₹2,845.00
            Stop-Loss:₹2,760.00  (−2.99%)
            Target:   ₹2,960.00  (+4.04%)
            R:R Ratio: 1 : 1.35
            Strategy: EMA_PULLBACK
            Sector:   Energy
            📝 RSI recovering from oversold; above 50-EMA
            ⚠️ This is an advisory signal — no order has been placed.
        """
        sl_pct = ((entry_price - stop_loss) / entry_price) * 100
        tgt_pct = ((target - entry_price) / entry_price) * 100
        risk    = entry_price - stop_loss
        reward  = target - entry_price
        rr_ratio = f"1 : {reward / risk:.2f}" if risk > 0 else "N/A"

        lines = [
            f"📡 *BUY SIGNAL — {symbol}*",
            "----------------------------",
            f"Qty:       {quantity} shares",
            f"Entry:     ₹{entry_price:,.2f}",
            f"Stop-Loss: ₹{stop_loss:,.2f}  (−{sl_pct:.2f}%)",
            f"Target:    ₹{target:,.2f}  (+{tgt_pct:.2f}%)",
            f"R:R Ratio: {rr_ratio}",
        ]
        if strategy:
            lines.append(f"Strategy:  {strategy}")
        if sector:
            lines.append(f"Sector:    {sector}")
        if reasoning:
            lines.append(f"📝 {reasoning[:200]}")
        lines.append("⚠️ _Advisory signal — no order has been placed._")

        return self._send("\n".join(lines))

    def send_exit_signal_alert(
        self,
        symbol: str,
        quantity: int,
        exit_type: str,
        current_price: float,
        entry_price: float = 0.0,
        pnl: float = 0.0,
        reasoning: str = "",
    ) -> bool:
        """Send a SELL/EXIT signal advisory message (signal-only mode).

        Example output::

            📡 EXIT SIGNAL — RELIANCE
            ----------------------------
            Exit Type: TARGET_HIT
            Qty:       12 shares
            Price:     ₹2,960.00
            Entry:     ₹2,845.00
            Est. P&L:  +₹1,380.00 (+4.04%)
            📝 Target price reached
            ⚠️ This is an advisory signal — no order has been placed.
        """
        lines = [
            f"📡 *EXIT SIGNAL — {symbol}*",
            "----------------------------",
            f"Exit Type: {exit_type}",
            f"Qty:       {quantity} shares",
            f"Price:     ₹{current_price:,.2f}",
        ]
        if entry_price > 0:
            lines.append(f"Entry:     ₹{entry_price:,.2f}")
        if pnl != 0.0:
            sign = "+" if pnl >= 0 else ""
            pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price > 0 and quantity > 0 else 0.0
            pnl_sign = "+" if pnl_pct >= 0 else ""
            lines.append(f"Est. P&L:  {sign}₹{pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)")
        if reasoning:
            lines.append(f"📝 {reasoning[:200]}")
        lines.append("⚠️ _Advisory signal — no order has been placed._")

        return self._send("\n".join(lines))

    def send_alert(self, message: str, level: str = "INFO") -> bool:
        """Send a generic alert with a severity emoji prefix.

        Level mapping: INFO → ℹ️ | WARNING → ⚠️ | ERROR → 🚨 | CRITICAL → 🔴🔴🔴
        """
        emoji = self._LEVEL_EMOJI.get(level.upper(), "ℹ️")
        return self._send(f"{emoji} {message}")

    def send_weekly_report(self, report: dict) -> bool:
        """Send the weekly performance summary.

        Example output::

            📅 WEEKLY REPORT — W12 2026
            -----------------------------------------
            Total P&L: +₹1,240 (+2.48%)
            Trades: 7 | Win Rate: 71.4%
            ...
        """
        week_label   = report.get("week_label", datetime.now().strftime("W%W %Y"))
        total_pnl    = report.get("total_pnl", 0.0)
        pnl_pct      = report.get("pnl_pct", 0.0)
        win_rate     = report.get("win_rate_pct", 0.0)
        total_trades = report.get("total_trades", 0)
        best_trade   = report.get("best_trade", {})
        worst_trade  = report.get("worst_trade", {})
        insights     = report.get("insights", "")

        sign = "+" if total_pnl >= 0 else ""

        lines: list[str] = [
            f"📅 *WEEKLY REPORT — {week_label}*",
            "-----------------------------------------",
            f"Total P&L: {sign}₹{total_pnl:,.0f} ({sign}{pnl_pct:.2f}%)",
            f"Trades: {total_trades} | Win Rate: {win_rate:.1f}%",
        ]

        if best_trade:
            bt_pnl = best_trade.get("pnl", 0.0)
            lines.append(f"Best:  {best_trade.get('symbol', 'N/A')} (+₹{bt_pnl:,.0f})")
        if worst_trade:
            wt_pnl = worst_trade.get("pnl", 0.0)
            wt_sign = "+" if wt_pnl >= 0 else ""
            lines.append(f"Worst: {worst_trade.get('symbol', 'N/A')} ({wt_sign}₹{wt_pnl:,.0f})")

        if insights:
            lines.append("")
            lines.append(f"📝 {insights[:300]}")

        lines.append("-----------------------------------------")
        return self._send("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════════
    # Interactive Approval Flow (Route 2: Human-in-the-Loop)
    # ══════════════════════════════════════════════════════════════════════

    def send_approval_request(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
        strategy: str = "",
        sector: str = "",
        timeout_seconds: int = 300,
    ) -> str:
        """Send a trade approval request with inline Yes/No buttons.

        Blocks the calling thread until the user taps a button or the
        timeout expires.

        Args:
            symbol:          Stock ticker (e.g. ``"RELIANCE"``).
            quantity:        Number of shares.
            entry_price:     Proposed limit price.
            stop_loss:       Stop-loss price.
            target:          First target price.
            strategy:        Strategy name (optional).
            sector:          Sector (optional).
            timeout_seconds: Max seconds to wait for a reply (default 300 = 5 min).

        Returns:
            ``"approved"``, ``"rejected"``, or ``"timeout"``.
        """
        if not self._enabled:
            _log.warning("Telegram not enabled — auto-rejecting approval for %s", symbol)
            return "timeout"

        sl_pct  = ((entry_price - stop_loss) / entry_price) * 100
        tgt_pct = ((target - entry_price) / entry_price) * 100
        cost    = quantity * entry_price
        risk    = entry_price - stop_loss
        reward  = target - entry_price
        rr      = f"1 : {reward / risk:.2f}" if risk > 0 else "N/A"

        lines = [
            f"🔔 *TRADE APPROVAL REQUIRED*",
            "----------------------------",
            f"Stock:      {symbol}",
            f"Action:     BUY",
            f"Qty:        {quantity} shares",
            f"Entry:      ₹{entry_price:,.2f}",
            f"Stop-Loss:  ₹{stop_loss:,.2f}  (−{sl_pct:.2f}%)",
            f"Target:     ₹{target:,.2f}  (+{tgt_pct:.2f}%)",
            f"R:R Ratio:  {rr}",
            f"Cost:       ₹{cost:,.2f}",
        ]
        if strategy:
            lines.append(f"Strategy:   {strategy}")
        if sector:
            lines.append(f"Sector:     {sector}")
        lines.append("")
        lines.append(f"⏳ Respond within {timeout_seconds // 60} min or trade is skipped.")

        text = "\n".join(lines)

        # Generate a unique callback ID so concurrent approvals don't collide
        request_id = uuid.uuid4().hex[:8]
        approve_data = f"approve_{request_id}"
        reject_data  = f"reject_{request_id}"

        try:
            result = asyncio.run(
                self._async_send_approval(
                    text, approve_data, reject_data, timeout_seconds
                )
            )
            return result
        except Exception as exc:
            _log.error("Approval request failed for %s: %s — defaulting to timeout", symbol, exc)
            return "timeout"

    async def _async_send_approval(
        self,
        text: str,
        approve_data: str,
        reject_data: str,
        timeout_seconds: int,
    ) -> str:
        """Send message with inline keyboard and poll for callback response."""
        from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
        from telegram.error import TelegramError

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data=approve_data),
                InlineKeyboardButton("❌ Reject",  callback_data=reject_data),
            ]
        ])

        async with Bot(token=self._bot_token) as bot:
            # Send the approval message
            sent_msg = await bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="Markdown",
                reply_markup=keyboard,
            )

            # Poll for callback query updates until timeout
            deadline    = time.monotonic() + timeout_seconds
            last_update = 0

            while time.monotonic() < deadline:
                try:
                    updates = await bot.get_updates(
                        offset=last_update + 1,
                        timeout=min(10, max(1, int(deadline - time.monotonic()))),
                        allowed_updates=["callback_query"],
                    )
                except TelegramError as exc:
                    _log.warning("Polling error during approval wait: %s", exc)
                    await asyncio.sleep(2)
                    continue

                for update in updates:
                    last_update = update.update_id
                    cb = update.callback_query
                    if cb is None:
                        continue
                    if cb.data == approve_data:
                        await cb.answer(text="✅ Trade approved!")
                        await bot.edit_message_text(
                            chat_id=self._chat_id,
                            message_id=sent_msg.message_id,
                            text=text + "\n\n✅ *APPROVED* by user",
                            parse_mode="Markdown",
                        )
                        return "approved"
                    elif cb.data == reject_data:
                        await cb.answer(text="❌ Trade rejected.")
                        await bot.edit_message_text(
                            chat_id=self._chat_id,
                            message_id=sent_msg.message_id,
                            text=text + "\n\n❌ *REJECTED* by user",
                            parse_mode="Markdown",
                        )
                        return "rejected"

            # Timeout — edit message to show expiry
            try:
                await bot.edit_message_text(
                    chat_id=self._chat_id,
                    message_id=sent_msg.message_id,
                    text=text + "\n\n⏰ *TIMED OUT* — trade skipped",
                    parse_mode="Markdown",
                )
            except TelegramError:
                pass

            return "timeout"

    def send_approval_exit_request(
        self,
        symbol: str,
        quantity: int,
        exit_type: str,
        current_price: float,
        entry_price: float = 0.0,
        pnl: float = 0.0,
        timeout_seconds: int = 300,
    ) -> str:
        """Send an EXIT approval request with inline Yes/No buttons.

        Same mechanics as :meth:`send_approval_request` but for exits.

        Returns:
            ``"approved"``, ``"rejected"``, or ``"timeout"``.
        """
        if not self._enabled:
            _log.warning("Telegram not enabled — auto-approving exit for %s", symbol)
            return "approved"  # Exits default to approved for safety

        sign = "+" if pnl >= 0 else ""
        pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price > 0 and quantity > 0 else 0.0

        lines = [
            f"🔔 *EXIT APPROVAL REQUIRED*",
            "----------------------------",
            f"Stock:      {symbol}",
            f"Action:     SELL ({exit_type})",
            f"Qty:        {quantity} shares",
            f"Price:      ₹{current_price:,.2f}",
        ]
        if entry_price > 0:
            lines.append(f"Entry:      ₹{entry_price:,.2f}")
        if pnl != 0.0:
            lines.append(f"Est. P&L:   {sign}₹{pnl:,.2f} ({sign}{pnl_pct:.2f}%)")
        lines.append("")
        lines.append(f"⏳ Respond within {timeout_seconds // 60} min or exit is *AUTO-EXECUTED*.")
        lines.append("(Exits auto-execute on timeout for safety)")

        text = "\n".join(lines)
        request_id   = uuid.uuid4().hex[:8]
        approve_data = f"approve_{request_id}"
        reject_data  = f"reject_{request_id}"

        try:
            result = asyncio.run(
                self._async_send_approval(text, approve_data, reject_data, timeout_seconds)
            )
            return result
        except Exception as exc:
            _log.error("Exit approval failed for %s: %s — auto-approving for safety", symbol, exc)
            return "approved"
