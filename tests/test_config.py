"""Tests for src/utils/config.py."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from src.utils.config import Config, ConfigError, get_config

# ---------------------------------------------------------------------------
# Minimal valid config fixture data
# ---------------------------------------------------------------------------
_VALID_CONFIG: dict = {
    "broker": {
        "angel_one": {
            "client_id": "${TEST_CLIENT_ID}",
            "api_key": "${TEST_API_KEY}",
            "totp_secret": "test_totp_secret",
            "default_exchange": "NSE",
        }
    },
    "trading": {
        "capital": 50000,
        "max_position_pct": 5,
        "max_daily_loss_pct": 2,
        "max_weekly_loss_pct": 5,
        "max_open_positions": 5,
        "stop_loss_pct": 3,
        "min_stock_price": 50,
        "max_stock_price": 5000,
        "min_volume_cr": 10,
        "trading_start_time": "09:30",
        "trading_end_time": "15:15",
        "paper_trading": True,
    },
    "llm": {
        "provider_priority": ["gemini"],
        "gemini": {"model": "gemini-1.5-flash", "daily_limit": 1500, "rpm_limit": 15},
    },
    "apis": {"tavily": {"max_results": 5}},
    "notifications": {"telegram": {"enabled": False}},
    "database": {"sqlite": {"path": "data/test.db"}},
    "logging": {"level": "DEBUG", "file": "logs/test.log"},
    "universe": {
        "indices": ["NIFTY_50"],
        "blacklisted_stocks": [],
        "sector_limits": {},
    },
    "schedule": {
        "pre_market": "08:00",
        "scan_interval_minutes": 60,
        "post_market": "15:45",
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the Config singleton before and after every test."""
    Config.reset()
    yield
    Config.reset()


@pytest.fixture()
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a valid config.yaml to *tmp_path* and redirect _PROJECT_ROOT."""
    (tmp_path / "config.yaml").write_text(
        yaml.dump(_VALID_CONFIG), encoding="utf-8"
    )
    import src.utils.config as mod
    monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_loads_valid_config(self, config_dir: Path) -> None:
        cfg = get_config()
        assert cfg["trading"]["capital"] == 50000

    def test_get_nested_value(self, config_dir: Path) -> None:
        cfg = get_config()
        assert cfg.get("trading", "max_open_positions") == 5

    def test_get_deeply_nested(self, config_dir: Path) -> None:
        cfg = get_config()
        assert cfg.get("llm", "gemini", "model") == "gemini-1.5-flash"

    def test_get_missing_key_returns_default(self, config_dir: Path) -> None:
        cfg = get_config()
        assert cfg.get("nonexistent") is None
        assert cfg.get("nonexistent", default="fallback") == "fallback"

    def test_contains_existing_section(self, config_dir: Path) -> None:
        cfg = get_config()
        assert "trading" in cfg

    def test_contains_missing_section(self, config_dir: Path) -> None:
        cfg = get_config()
        assert "nonexistent" not in cfg


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_returns_same_instance(self, config_dir: Path) -> None:
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_allows_fresh_load(self, config_dir: Path) -> None:
        cfg1 = get_config()
        Config.reset()
        cfg2 = get_config()
        # Different Python objects after reset
        assert cfg1 is not cfg2

    def test_direct_instantiation_also_singleton(self, config_dir: Path) -> None:
        cfg1 = Config()
        cfg2 = Config()
        assert cfg1 is cfg2


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------

class TestEnvInterpolation:
    def test_resolves_env_var(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_CLIENT_ID", "client_abc_123")
        cfg = get_config()
        assert cfg["broker"]["angel_one"]["client_id"] == "client_abc_123"

    def test_unset_var_leaves_placeholder(self, config_dir: Path) -> None:
        os.environ.pop("TEST_API_KEY", None)
        cfg = get_config()
        assert "${TEST_API_KEY}" in cfg["broker"]["angel_one"]["api_key"]

    def test_multiple_vars_in_one_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A string like '${A}_${B}' should resolve both placeholders."""
        cfg_data = copy.deepcopy(_VALID_CONFIG)
        cfg_data["broker"]["angel_one"]["client_id"] = "${PART_A}_${PART_B}"
        (tmp_path / "config.yaml").write_text(
            yaml.dump(cfg_data), encoding="utf-8"
        )
        import src.utils.config as mod
        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("PART_A", "hello")
        monkeypatch.setenv("PART_B", "world")
        cfg = get_config()
        assert cfg["broker"]["angel_one"]["client_id"] == "hello_world"

    def test_loads_dotenv_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "config.yaml").write_text(
            yaml.dump(_VALID_CONFIG), encoding="utf-8"
        )
        (tmp_path / ".env").write_text(
            "TEST_CLIENT_ID=from_dotenv\n", encoding="utf-8"
        )
        import src.utils.config as mod
        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        cfg = get_config()
        assert cfg["broker"]["angel_one"]["client_id"] == "from_dotenv"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestConfigErrors:
    def test_missing_config_file_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.utils.config as mod
        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        with pytest.raises(ConfigError, match="config.yaml not found"):
            get_config()

    def test_missing_required_section_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.utils.config as mod
        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        incomplete = {k: v for k, v in _VALID_CONFIG.items() if k != "trading"}
        (tmp_path / "config.yaml").write_text(
            yaml.dump(incomplete), encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="Missing required config sections"):
            get_config()

    def test_missing_trading_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.utils.config as mod
        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        bad = dict(_VALID_CONFIG)
        bad["trading"] = {k: v for k, v in _VALID_CONFIG["trading"].items()
                          if k != "capital"}
        (tmp_path / "config.yaml").write_text(yaml.dump(bad), encoding="utf-8")
        with pytest.raises(ConfigError, match="capital"):
            get_config()

    def test_zero_capital_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.utils.config as mod
        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        bad = dict(_VALID_CONFIG)
        bad["trading"] = dict(_VALID_CONFIG["trading"])
        bad["trading"]["capital"] = 0
        (tmp_path / "config.yaml").write_text(yaml.dump(bad), encoding="utf-8")
        with pytest.raises(ConfigError, match="capital"):
            get_config()

    def test_non_dict_yaml_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.utils.config as mod
        monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "config.yaml").write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="YAML mapping"):
            get_config()
