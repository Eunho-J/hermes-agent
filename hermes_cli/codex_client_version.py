"""Resolve the Codex CLI compatibility version for ChatGPT Codex OAuth.

The chatgpt.com/backend-api/codex backend gates some models by Codex client
version. Hermes is not launching the official CLI, so it needs to advertise a
Codex-compatible version string in headers and model-discovery query params.

Resolution order:
1. HERMES_CODEX_CLIENT_VERSION if set to a concrete version.
2. providers.openai-codex.codex_client_version if concrete.
3. Fresh profile cache.
4. `npm view @openai/codex version --json`.
5. Stale profile cache.
6. Bundled known-good fallback.

The installed `codex` binary is deliberately ignored: gateway services often
run with a different PATH than interactive shells, and a local binary can be
older than the backend's currently accepted app version.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

NPM_CODEX_PACKAGE = "@openai/codex"
BUNDLED_CODEX_CLIENT_VERSION = "0.133.0"
DEFAULT_CODEX_CLIENT_VERSION_TTL_HOURS = 24.0
DEFAULT_NPM_TIMEOUT_SECONDS = 5.0
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_process_cache: tuple[str, float] | None = None


def _cache_path() -> Path:
    return get_hermes_home() / "cache" / "codex_client_version.json"


def _valid_version(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().strip('"')
    if not stripped or stripped.lower() == "auto":
        return None
    if not _VERSION_RE.match(stripped):
        return None
    return stripped


def _load_provider_config(provider_id: str) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:
        return {}
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    return provider_config if isinstance(provider_config, dict) else {}


def _ttl_hours(provider_config: dict[str, Any]) -> float:
    raw = os.getenv("HERMES_CODEX_CLIENT_VERSION_TTL_HOURS")
    if raw is None:
        raw = provider_config.get("codex_client_version_ttl_hours")
    try:
        ttl = float(raw) if raw is not None else DEFAULT_CODEX_CLIENT_VERSION_TTL_HOURS
    except (TypeError, ValueError):
        ttl = DEFAULT_CODEX_CLIENT_VERSION_TTL_HOURS
    return max(0.0, ttl)


def _read_cache() -> tuple[str | None, float]:
    path = _cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, 0.0
    version = _valid_version(data.get("version")) if isinstance(data, dict) else None
    try:
        fetched_at = float(data.get("fetched_at", 0.0)) if isinstance(data, dict) else 0.0
    except (TypeError, ValueError):
        fetched_at = 0.0
    return version, fetched_at


def _write_cache(version: str) -> None:
    path = _cache_path()
    payload = {
        "version": version,
        "fetched_at": time.time(),
        "source": "npm",
        "package": NPM_CODEX_PACKAGE,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        atomic_replace(tmp, path)
    except OSError as exc:
        logger.debug("failed to write Codex client version cache: %s", exc)


def _fetch_npm_latest(timeout: float = DEFAULT_NPM_TIMEOUT_SECONDS) -> str | None:
    try:
        proc = subprocess.run(
            ["npm", "view", NPM_CODEX_PACKAGE, "version", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("Codex client version npm lookup failed: %s", exc)
        return None
    if proc.returncode != 0:
        logger.debug(
            "Codex client version npm lookup exited %s: %s",
            proc.returncode,
            (proc.stderr or proc.stdout).strip()[:300],
        )
        return None
    raw = (proc.stdout or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    version = _valid_version(parsed)
    if version is None:
        logger.debug("Codex client version npm lookup returned invalid version: %r", raw)
    return version


def resolve_codex_client_version(provider_id: str = "openai-codex") -> str:
    """Return the Codex compatibility version Hermes should advertise."""
    env_version = _valid_version(os.getenv("HERMES_CODEX_CLIENT_VERSION"))
    if env_version:
        return env_version

    provider_config = _load_provider_config(provider_id)
    config_version = _valid_version(provider_config.get("codex_client_version"))
    if config_version:
        return config_version

    ttl_seconds = _ttl_hours(provider_config) * 3600.0
    now = time.time()

    global _process_cache
    if _process_cache is not None:
        version, fetched_at = _process_cache
        if ttl_seconds > 0 and (now - fetched_at) < ttl_seconds:
            return version

    cached_version, cached_at = _read_cache()
    if cached_version and ttl_seconds > 0 and (now - cached_at) < ttl_seconds:
        _process_cache = (cached_version, cached_at)
        return cached_version

    fetched = _fetch_npm_latest()
    if fetched:
        _write_cache(fetched)
        _process_cache = (fetched, now)
        return fetched

    if cached_version:
        _process_cache = (cached_version, cached_at or now)
        return cached_version

    return BUNDLED_CODEX_CLIENT_VERSION
