"""Configuration management for the trading bot.

Loads config.yaml, resolves ${ENV_VAR} references from the environment,
and exposes a singleton Config instance used throughout the application.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root is three levels up from this file:
#   src/utils/config.py  →  src/utils/  →  src/  →  trading-bot/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfigError(Exception):
    """Raised when configuration is invalid or a required field is missing."""


class Config:
    """Singleton configuration manager.

    Loads ``config.yaml`` once, resolves every ``${ENV_VAR}`` placeholder
    against the process environment (populated from ``.env`` if present),
    validates required sections, and provides convenient accessor methods.

    Usage::

        from src.utils.config import get_config
        cfg = get_config()
        capital = cfg.get("trading", "capital")   # 50000
        api_key = cfg["broker"]["angel_one"]["api_key"]
    """

    _instance: Config | None = None
    _config: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Singleton construction                                               #
    # ------------------------------------------------------------------ #

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Guard: only load once per singleton lifetime.
        if Config._config is None:
            self._load()

    # ------------------------------------------------------------------ #
    # Internal loading pipeline                                            #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load .env, parse config.yaml, resolve env vars, then validate."""
        self._load_dotenv()
        raw = self._read_yaml()
        Config._config = self._resolve_env_vars(raw)
        self._validate()

    def _load_dotenv(self) -> None:
        """Load .env from the project root if it exists; fall back to CWD."""
        env_path = _PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
        else:
            load_dotenv(override=False)

    def _read_yaml(self) -> dict[str, Any]:
        """Read and parse config.yaml, raising ConfigError on failure."""
        config_path = _PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            raise ConfigError(
                f"config.yaml not found at {config_path}. "
                "Copy config.yaml to the project root before starting the bot."
            )
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError(
                "config.yaml must be a YAML mapping at the root level."
            )
        return data

    # ------------------------------------------------------------------ #
    # Environment variable interpolation                                   #
    # ------------------------------------------------------------------ #

    _ENV_PATTERN: re.Pattern[str] = re.compile(r"\$\{([^}]+)\}")

    def _resolve_env_vars(self, obj: Any) -> Any:
        """Recursively replace ``${VAR_NAME}`` with environment values."""
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        if isinstance(obj, str):
            return self._interpolate(obj)
        return obj

    def _interpolate(self, value: str) -> str:
        """Replace all ``${VAR}`` occurrences in *value* with env values.

        Unresolved placeholders (variable not in environment) are left as-is
        so that the application can detect and report missing secrets clearly.
        """
        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.environ.get(var, match.group(0))

        return self._ENV_PATTERN.sub(_replace, value)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    _REQUIRED_SECTIONS: tuple[str, ...] = (
        "broker",
        "trading",
        "llm",
        "apis",
        "notifications",
        "database",
        "logging",
        "universe",
        "schedule",
    )

    _REQUIRED_TRADING_KEYS: tuple[str, ...] = (
        "capital",
        "max_position_pct",
        "max_daily_loss_pct",
        "max_weekly_loss_pct",
        "max_open_positions",
        "stop_loss_pct",
    )

    def _validate(self) -> None:
        """Raise ConfigError if any required section or key is absent."""
        assert Config._config is not None  # set by _load before _validate

        missing_sections = [
            s for s in self._REQUIRED_SECTIONS if s not in Config._config
        ]
        if missing_sections:
            raise ConfigError(
                f"Missing required config sections: {missing_sections}. "
                "Check your config.yaml against config.yaml.example."
            )

        trading = Config._config["trading"]
        missing_trading = [
            k for k in self._REQUIRED_TRADING_KEYS if k not in trading
        ]
        if missing_trading:
            raise ConfigError(
                f"Missing required keys under 'trading': {missing_trading}."
            )

        capital = trading.get("capital", 0)
        if not isinstance(capital, (int, float)) or capital <= 0:
            raise ConfigError(
                f"'trading.capital' must be a positive number, got {capital!r}."
            )

    # ------------------------------------------------------------------ #
    # Public accessors                                                     #
    # ------------------------------------------------------------------ #

    def get(self, *keys: str, default: Any = None) -> Any:
        """Return a nested config value using a sequence of string keys.

        Example::

            cfg.get("trading", "capital")         # 50000
            cfg.get("llm", "gemini", "model")     # "gemini-1.5-flash"
            cfg.get("nonexistent", default=42)    # 42
        """
        obj: Any = Config._config
        for key in keys:
            if not isinstance(obj, dict) or key not in obj:
                return default
            obj = obj[key]
        return obj

    def __getitem__(self, key: str) -> Any:
        assert Config._config is not None
        return Config._config[key]

    def __contains__(self, key: str) -> bool:
        assert Config._config is not None
        return key in Config._config

    # ------------------------------------------------------------------ #
    # Test helpers                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton so the next call creates a fresh instance.

        Only for use in unit tests — do not call in production code.
        """
        cls._instance = None
        cls._config = None


# ------------------------------------------------------------------ #
# Module-level convenience function                                    #
# ------------------------------------------------------------------ #

def get_config() -> Config:
    """Return the application-wide singleton :class:`Config` instance."""
    return Config()
