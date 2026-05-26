from __future__ import annotations

import json
import time
from types import SimpleNamespace

from hermes_cli import codex_client_version as cv


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CODEX_CLIENT_VERSION", "0.200.0")
    monkeypatch.setattr(cv, "_fetch_npm_latest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("npm should not run")))

    assert cv.resolve_codex_client_version() == "0.200.0"


def test_config_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_CODEX_CLIENT_VERSION", raising=False)
    _write_config(
        tmp_path,
        """
providers:
  openai-codex:
    codex_client_version: "0.201.0"
""",
    )
    monkeypatch.setattr(cv, "_fetch_npm_latest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("npm should not run")))

    assert cv.resolve_codex_client_version() == "0.201.0"


def test_fresh_cache_used_before_npm(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_CODEX_CLIENT_VERSION", raising=False)
    _write_config(tmp_path, "providers:\n  openai-codex:\n    codex_client_version_ttl_hours: 24\n")
    cache = tmp_path / "cache" / "codex_client_version.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps({"version": "0.202.0", "fetched_at": time.time()}), encoding="utf-8")
    monkeypatch.setattr(cv, "_fetch_npm_latest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("npm should not run")))
    monkeypatch.setattr(cv, "_process_cache", None)

    assert cv.resolve_codex_client_version() == "0.202.0"


def test_npm_result_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_CODEX_CLIENT_VERSION", raising=False)
    _write_config(tmp_path, "{}\n")
    monkeypatch.setattr(cv, "_process_cache", None)
    monkeypatch.setattr(cv, "_fetch_npm_latest", lambda *a, **k: "0.203.0")

    assert cv.resolve_codex_client_version() == "0.203.0"
    payload = json.loads((tmp_path / "cache" / "codex_client_version.json").read_text())
    assert payload["version"] == "0.203.0"
    assert payload["source"] == "npm"


def test_stale_cache_used_when_npm_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_CODEX_CLIENT_VERSION", raising=False)
    _write_config(tmp_path, "providers:\n  openai-codex:\n    codex_client_version_ttl_hours: 0.001\n")
    cache = tmp_path / "cache" / "codex_client_version.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps({"version": "0.204.0", "fetched_at": 1}), encoding="utf-8")
    monkeypatch.setattr(cv, "_process_cache", None)
    monkeypatch.setattr(cv, "_fetch_npm_latest", lambda *a, **k: None)

    assert cv.resolve_codex_client_version() == "0.204.0"


def test_fallback_when_no_cache_and_npm_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_CODEX_CLIENT_VERSION", raising=False)
    _write_config(tmp_path, "{}\n")
    monkeypatch.setattr(cv, "_process_cache", None)
    monkeypatch.setattr(cv, "_fetch_npm_latest", lambda *a, **k: None)

    assert cv.resolve_codex_client_version() == cv.BUNDLED_CODEX_CLIENT_VERSION


def test_fetch_npm_latest_parses_json(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout='"0.205.0"\n', stderr="")

    monkeypatch.setattr(cv.subprocess, "run", fake_run)
    assert cv._fetch_npm_latest() == "0.205.0"
