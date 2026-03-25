# Indian Stock Market Swing Trading Bot

A multi-agent swing trading bot for NSE/BSE equities powered by Angel One SmartAPI and LLMs. The bot runs on a schedule, scans NIFTY indices for swing trade setups, performs AI-driven research, sizes positions with risk controls, and places/manages orders — all with optional Telegram notifications.

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
  - [Environment Variables (.env)](#environment-variables-env)
  - [Application Settings (config.yaml)](#application-settings-configyaml)
- [Running the Bot](#running-the-bot)
- [Paper Trading vs Live Trading](#paper-trading-vs-live-trading)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Scheduler & Daily Flow](#scheduler--daily-flow)
- [LLM Provider Setup](#llm-provider-setup)
- [Telegram Notifications](#telegram-notifications)

---

## Architecture

```
Orchestrator
├── UniverseAgent    — filters stocks from NIFTY indices by price, volume & sector
├── QuantAgent       — technical analysis & signal generation (RSI, MACD, Bollinger, etc.)
├── ResearchAgent    — LLM-powered news & fundamental research via web search
├── RiskAgent        — position sizing, portfolio-level risk, weekly/daily loss checks
├── ExecutionAgent   — order placement via Angel One SmartAPI
├── ExitAgent        — stop-loss / target / trailing-stop management
└── JournalAgent     — trade logging & performance reporting
```

Supporting infrastructure:

| Module | Purpose |
|--------|---------|
| `src/broker/angel_one.py` | Angel One SmartAPI wrapper (login, orders, quotes) |
| `src/llm/router.py` | Routes prompts across Gemini / Groq / NVIDIA / Ollama |
| `src/llm/budget_manager.py` | Enforces daily API call limits per provider |
| `src/data/market_data.py` | OHLCV data (Angel One + yfinance fallback) |
| `src/data/news_data.py` | News aggregation (NewsAPI, Tavily, SerpAPI, RSS feeds) |
| `src/tools/technical_indicators.py` | pandas_ta wrappers |
| `src/database/db_manager.py` | SQLite persistence (trades, journal, signals) |
| `src/notifications/telegram_bot.py` | Telegram alerts |
| `src/circuit_breakers/safety.py` | Daily/weekly loss-limit circuit breaker |
| `src/scheduler.py` | APScheduler job management |

---

## Prerequisites

- Python **3.11+**
- An **Angel One** trading account with SmartAPI enabled
- At least one **LLM provider** API key (Gemini free tier is a good starting point)
- (Optional) Telegram bot token for notifications

---

## Setup

### 1. Clone and enter the project

```bash
cd trading-bot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure secrets

```bash
cp .env.example .env
```

Open `.env` and fill in all required values (see [Environment Variables](#environment-variables-env) below).

### 5. (Optional) Review config.yaml

Open [config.yaml](config.yaml) and adjust trading parameters, LLM priorities, schedule times, etc. to match your preferences.

### 6. Verify setup

```bash
pytest tests/ -v
```

All tests should pass before you start the bot.

---

## Configuration

There are two config files:

| File | Purpose |
|------|---------|
| `.env` | **Secrets** — API keys, passwords. Never commit this file. |
| `config.yaml` | **Application settings** — trading params, LLM models, schedule, etc. |

### Environment Variables (.env)

Copy `.env.example` to `.env` and fill in each value:

#### Angel One SmartAPI (required for live trading)

| Variable | How to get it |
|----------|--------------|
| `ANGEL_ONE_CLIENT_ID` | Your Angel One client ID (shown in the app under Profile) |
| `ANGEL_ONE_API_KEY` | Create an app at [smartapi.angelbroking.com](https://smartapi.angelbroking.com/) |
| `ANGEL_ONE_PASSWORD` | Your 4-digit Angel One MPIN |
| `ANGEL_ONE_TOTP_SECRET` | Base32 secret shown when you enable 2FA in the Angel One app |

#### LLM Providers (at least one required)

| Variable | Where to get it |
|----------|----------------|
| `GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — has a generous free tier |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) — free tier available |
| `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com/) |

Leave any unused LLM key blank — the router will skip providers without a key.

#### Search & News APIs (optional but recommended)

| Variable | Where to get it |
|----------|----------------|
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com/) |
| `NEWSAPI_API_KEY` | [newsapi.org/account](https://newsapi.org/account) |
| `SERPAPI_API_KEY` | [serpapi.com/manage-api-key](https://serpapi.com/manage-api-key) |

The bot uses DuckDuckGo as a free fallback if none of these are set.

#### Telegram (optional)

| Variable | How to get it |
|----------|--------------|
| `TELEGRAM_BOT_TOKEN` | Message `@BotFather` on Telegram and create a new bot |
| `TELEGRAM_CHAT_ID` | Message `@userinfobot` on Telegram to get your chat ID |

#### Optional overrides

```env
# Force paper-trading mode regardless of config.yaml
PAPER_TRADING=true

# Override log level (DEBUG | INFO | WARNING | ERROR)
LOG_LEVEL=DEBUG
```

---

### Application Settings (config.yaml)

#### Broker

```yaml
broker:
  angel_one:
    default_exchange: "NSE"   # Change to "BSE" if needed
```

#### Trading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trading.capital` | `50000` | Total capital allocated to the bot (INR) |
| `trading.max_position_pct` | `5` | Max size of a single position as % of capital |
| `trading.max_daily_loss_pct` | `2` | Daily loss limit — bot halts if breached |
| `trading.max_weekly_loss_pct` | `5` | Weekly loss limit — bot halts if breached |
| `trading.max_open_positions` | `5` | Maximum concurrent open positions |
| `trading.stop_loss_pct` | `3` | Default stop-loss distance below entry (%) |
| `trading.min_stock_price` | `50` | Minimum stock price filter (INR) |
| `trading.max_stock_price` | `5000` | Maximum stock price filter (INR) |
| `trading.min_volume_cr` | `10` | Minimum avg daily volume (crores INR) |
| `trading.paper_trading` | `true` | Set to `false` to enable live order placement |

> **Always start with `paper_trading: true`** until you are confident in the bot's behaviour.

#### LLM Configuration

```yaml
llm:
  provider_priority:
    - "gemini"    # Tried first
    - "groq"      # Tried second
    - "ollama"    # Local fallback (no cost)
    - "nvidia"    # Tried last
```

Remove providers you don't have keys for, or reorder to change priority. Each provider has its own rate-limit settings:

```yaml
  gemini:
    model: "gemini-1.5-flash"
    daily_limit: 1500     # Max requests/day
    rpm_limit: 15         # Max requests/minute
```

To use Ollama (local models), install [Ollama](https://ollama.ai/) and pull a model:

```bash
ollama pull llama3.1:8b
```

#### Stock Universe

```yaml
universe:
  indices:
    - "NIFTY_50"
    - "NIFTY_NEXT_50"
    - "NIFTY_MIDCAP_SELECT"
  blacklisted_stocks: []    # Add symbols to exclude, e.g. ["YESBANK", "ZOMATO"]
  sector_limits: {}         # Optional: {IT: 2, PHARMA: 1}
```

#### Schedule (IST, 24h format)

```yaml
schedule:
  pre_market: "08:00"           # Pre-market scan (news, universe refresh)
  scan_interval_minutes: 60     # Intra-day scan frequency during market hours
  post_market: "15:45"          # Post-market reconciliation & daily report
```

#### Notifications

```yaml
notifications:
  telegram:
    enabled: true
    min_level: "INFO"   # Only send INFO and above. Use "WARNING" for fewer alerts.
```

#### Database & Logging

```yaml
database:
  sqlite:
    path: "data/trading_bot.db"   # Relative to project root

logging:
  level: "INFO"                   # DEBUG | INFO | WARNING | ERROR
  file: "logs/trading_bot.log"
```

---

## Running the Bot

Make sure your virtual environment is activated and `.env` is filled in, then:

```bash
python main.py
```

The bot will:
1. Load `config.yaml` and `.env`
2. Set up logging (console + file)
3. Initialize all agents, broker client, and scheduler
4. Send a Telegram message: `Trading Bot is ONLINE`
5. Block and run scheduled jobs until you press `Ctrl+C`

To stop gracefully:

```
Ctrl+C
```

The bot will drain running jobs, send a `Trading Bot is OFFLINE` Telegram message, and exit.

### Viewing logs

```bash
# Follow live log output
tail -f logs/trading_bot.log

# Filter for errors only
grep "ERROR\|CRITICAL" logs/trading_bot.log
```

---

## Paper Trading vs Live Trading

| Setting | Behaviour |
|---------|-----------|
| `paper_trading: true` (default) | Bot runs full logic but **never places real orders**. Angel One login is skipped. Safe for testing. |
| `paper_trading: false` | **Real orders are placed** via Angel One SmartAPI. Requires valid broker credentials. |

To switch to live trading:

1. Set `trading.paper_trading: false` in `config.yaml`
2. Ensure all four `ANGEL_ONE_*` variables are set in `.env`
3. Start with a small `capital` value (e.g. `10000`) and monitor closely

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_quant_agent.py -v
```

Test files:

| Test file | What it covers |
|-----------|---------------|
| `test_config.py` | Config loading & validation |
| `test_logger.py` | Logging setup |
| `test_angel_one.py` | Broker wrapper (mocked) |
| `test_llm_router.py` | LLM routing & fallback |
| `test_universe_agent.py` | Stock universe filtering |
| `test_quant_agent.py` | Technical indicator signals |
| `test_research_agent.py` | News & LLM research |
| `test_risk_agent.py` | Position sizing & risk checks |
| `test_risk_scenarios.py` | Circuit breaker scenarios |
| `test_exit_agent.py` | Stop-loss & target logic |
| `test_journal_agent.py` | Trade logging |
| `test_database.py` | SQLite persistence |
| `test_tools.py` | TA indicators & web search |

---

## Project Structure

```
trading-bot/
├── main.py                      # Entry point — boot sequence
├── config.yaml                  # Application configuration (edit this)
├── .env.example                 # Environment variable template (copy to .env)
├── requirements.txt             # Python dependencies
│
├── src/
│   ├── agents/
│   │   ├── orchestrator.py      # Coordinates all agents
│   │   ├── universe_agent.py    # Stock screening
│   │   ├── quant_agent.py       # Technical analysis
│   │   ├── research_agent.py    # LLM news research
│   │   ├── risk_agent.py        # Position sizing & risk
│   │   ├── execution_agent.py   # Order placement
│   │   ├── exit_agent.py        # Stop-loss / target management
│   │   └── journal_agent.py     # Trade logging & reports
│   │
│   ├── broker/
│   │   └── angel_one.py         # Angel One SmartAPI wrapper
│   │
│   ├── llm/
│   │   ├── router.py            # Multi-provider LLM routing
│   │   └── budget_manager.py    # Daily request limit tracking
│   │
│   ├── data/
│   │   ├── market_data.py       # OHLCV fetcher (Angel One + yfinance)
│   │   └── news_data.py         # News aggregation
│   │
│   ├── tools/
│   │   └── technical_indicators.py  # RSI, MACD, Bollinger, ATR, etc.
│   │
│   ├── database/
│   │   └── db_manager.py        # SQLite — trades, signals, journal
│   │
│   ├── notifications/
│   │   └── telegram_bot.py      # Telegram alerts
│   │
│   ├── circuit_breakers/
│   │   └── safety.py            # Loss-limit circuit breaker
│   │
│   ├── scheduler.py             # APScheduler job definitions
│   │
│   └── utils/
│       ├── config.py            # Config loader (YAML + dotenv)
│       └── logger.py            # Logging setup
│
├── tests/                       # pytest test suite
├── data/                        # Runtime data — SQLite DB, caches (auto-created)
└── logs/                        # Log files (auto-created)
```

---

## Scheduler & Daily Flow

```
08:00 IST  — Pre-market
             • Pull latest news
             • Refresh stock universe (apply price, volume, sector filters)

09:30–15:15 — Market hours (every 60 min by default)
             • Fetch OHLCV data
             • Run QuantAgent — generate buy/sell signals
             • Run ResearchAgent — LLM sentiment on candidates
             • Run RiskAgent — approve or reject trades based on portfolio risk
             • Run ExecutionAgent — place approved orders
             • Run ExitAgent — check stops/targets on open positions

15:45 IST  — Post-market
             • Reconcile open positions with broker
             • Run JournalAgent — generate daily P&L report
             • Send Telegram daily summary
```

To change the intra-day scan frequency, edit `schedule.scan_interval_minutes` in `config.yaml`.

---

## LLM Provider Setup

The bot routes LLM calls across providers in priority order, falling back to the next one when rate limits are hit.

### Gemini (recommended for free tier)

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Set `GEMINI_API_KEY` in `.env`

Free tier: 1,500 requests/day, 15 RPM — sufficient for most users.

### Groq

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Set `GROQ_API_KEY` in `.env`

### Ollama (local, no cost)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama3.1:8b`
3. Ollama runs at `http://localhost:11434` by default — no API key needed
4. Add `"ollama"` to `llm.provider_priority` in `config.yaml`

---

## Telegram Notifications

1. Open Telegram and message `@BotFather`
2. Send `/newbot` and follow prompts to get a bot token
3. Set `TELEGRAM_BOT_TOKEN` in `.env`
4. Message `@userinfobot` to get your `TELEGRAM_CHAT_ID`
5. Set `TELEGRAM_CHAT_ID` in `.env`
6. Make sure `notifications.telegram.enabled: true` in `config.yaml`

The bot sends alerts for: startup/shutdown, trade entries/exits, circuit breaker triggers, and the daily P&L report. Adjust `min_level` in `config.yaml` to control verbosity (`DEBUG` → all messages, `WARNING` → only important alerts).

---

## Important Notes

- **Never commit `.env`** — it contains your broker credentials and API keys.
- The `data/` and `logs/` directories are created automatically on first run.
- The SQLite database at `data/trading_bot.db` persists trade history across restarts.
- The circuit breaker halts all trading for the day if daily loss exceeds `max_daily_loss_pct`, or for the week if weekly loss exceeds `max_weekly_loss_pct`.
