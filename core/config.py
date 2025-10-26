"""Configuration loader for the tracking_performance repository.

The loader consumes a YAML file (defaults to ``config/defaults.yaml``) and
applies overrides from environment variables or CLI flags.  It is intentionally
stdlib-only apart from PyYAML.

Usage
-----
>>> from core.config import get_config
>>> cfg = get_config()  # resolved defaults + overrides

Command line overrides are available everywhere because :func:`get_config`
looks at ``sys.argv`` on the first call.  The recognised options are::

    --config <path>            # alternative YAML file
    --set section.key=value    # dotted override (repeatable)

Environment overrides are supported in two forms:
* Prefixed variables: ``TP__SECTION__KEY=value`` (case-insensitive).
* Compatibility aliases for legacy variables (e.g. ``JERK_PSD``).

The loader records the resolved path and overrides so the orchestrator can
serialize the exact configuration that was used in a run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"
_ENV_PREFIX = "TP__"
_CONFIG_PATH_ENV_VARS: tuple[str, ...] = ("TP_CONFIG", "TP_CONFIG_FILE", "PIPELINE_CONFIG")

# Legacy environment aliases → dotted config paths
_ENV_ALIASES: dict[str, str] = {
    "GPM_SIGMA_URAD": "geometry.gpm.sigma_urad",
    "GPM_SEED": "geometry.gpm.seed",
    "JERK_PSD": "estimation.run_filter.jerk_psd",
    "P0_SCALE": "estimation.run_filter.p0_scale",
    "LOG_LEVEL": "estimation.run_filter.log_level",
    "CHI2_GATE_3DOF": "estimation.kf.chi2_gate_3dof",
    "FORCE_UPDATE": "estimation.kf.force_update",
}

_SENTINEL = object()

_last_config: dict[str, Any] | None = None
_last_signature: tuple[Any, ...] | None = None
_last_source: Path | None = None
_last_overrides: dict[str, Any] = {}


def _parse_cli(cli_args: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    return parser.parse_known_args(list(cli_args))


def _determine_config_path(env: Mapping[str, str], cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).expanduser()
    for key in _CONFIG_PATH_ENV_VARS:
        val = env.get(key)
        if val:
            return Path(val).expanduser()
    return _DEFAULT_CONFIG_PATH


def _collect_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for alias, path in _ENV_ALIASES.items():
        val = env.get(alias)
        if val is None:
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        overrides[path] = val
    for key, val in env.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        dotted = key[len(_ENV_PREFIX):].replace("__", ".")
        overrides[dotted] = val
    return overrides


def _collect_cli_overrides(values: Iterable[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if item is None:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --set override '{item}'; expected path=value")
        path, raw = item.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError(f"Invalid --set override '{item}'; empty path")
        overrides[path] = raw
    return overrides


def _parse_literal(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (bool, int, float)):
        return raw
    text = str(raw).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(text)
    except Exception:
        return text


def _match_key(container: MutableMapping[str, Any], token: str, create: bool = True) -> str:
    token_norm = token.lower()
    for existing in container.keys():
        if str(existing).lower() == token_norm:
            return existing  # type: ignore[return-value]
    if create:
        return token
    raise KeyError(token)


def _get_ref(container: MutableMapping[str, Any], tokens: Sequence[str]) -> Any:
    cur: Any = container
    for tok in tokens[:-1]:
        if not isinstance(cur, MutableMapping):
            return None
        key = _match_key(cur, tok, create=False)
        cur = cur.get(key)
    if isinstance(cur, MutableMapping):
        key = _match_key(cur, tokens[-1], create=False)
        return cur.get(key)
    return None


def _coerce_to_reference(value: Any, reference: Any) -> Any:
    if reference is None:
        return value
    try:
        if isinstance(reference, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(reference, int) and not isinstance(reference, bool):
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            return int(float(str(value)))
        if isinstance(reference, float):
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))
        if isinstance(reference, (list, tuple)):
            if isinstance(value, str):
                parsed = json.loads(value)
            else:
                parsed = value
            if isinstance(reference, tuple):
                return tuple(parsed)
            return list(parsed)
        if isinstance(reference, MutableMapping):
            if isinstance(value, str):
                parsed = json.loads(value)
            else:
                parsed = value
            if isinstance(parsed, Mapping):
                return dict(parsed)
            return reference
    except Exception:
        return value
    return value


def _set_path(container: MutableMapping[str, Any], path: str, raw_value: Any,
              record: dict[str, Any]) -> None:
    tokens = [tok for tok in path.split(".") if tok]
    if not tokens:
        raise ValueError("Empty config path in override")
    cur: MutableMapping[str, Any] = container
    for tok in tokens[:-1]:
        key = _match_key(cur, tok)
        nxt = cur.get(key)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[key] = nxt
        cur = nxt  # type: ignore[assignment]
    last_token = tokens[-1]
    key = _match_key(cur, last_token)
    ref = None
    try:
        ref = cur.get(key)
    except Exception:
        ref = None
    value = _parse_literal(raw_value)
    if ref is not None:
        coerced = _coerce_to_reference(value, ref)
    else:
        coerced = value
    cur[key] = coerced
    record[".".join(tokens)] = coerced


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Root of config must be a mapping, got {type(data)!r}")
        return data


def get_config(*, cli_args: Sequence[str] | object = _SENTINEL,
               env: Mapping[str, str] | None = None,
               reload: bool = False,
               with_cli: bool = False) -> Any:
    """Return the effective configuration dictionary.

    Args:
        cli_args: explicit CLI arguments.  By default ``sys.argv[1:]`` is used so
            overrides work even if modules access the configuration at import
            time.  Pass an empty list to ignore CLI overrides.
        env: mapping of environment variables.  Defaults to ``os.environ``.
        reload: force the YAML to be re-read even if the cached signature matches.
        with_cli: when ``True`` returns ``(config, remaining_args)``.
    """

    if env is None:
        env = os.environ
    if cli_args is _SENTINEL:
        cli_args = sys.argv[1:]
    known, remaining = _parse_cli(cli_args)  # type: ignore[arg-type]

    path = _determine_config_path(env, known.config_path)
    overrides_env = _collect_env_overrides(env)
    overrides_cli = _collect_cli_overrides(known.set_values)

    signature = (
        str(path.resolve()),
        tuple(sorted((k.lower(), str(v)) for k, v in overrides_env.items())),
        tuple(sorted((k.lower(), str(v)) for k, v in overrides_cli.items())),
    )

    global _last_config, _last_signature, _last_source, _last_overrides
    if not reload and _last_config is not None and signature == _last_signature:
        cfg = deepcopy(_last_config)
        if with_cli:
            return cfg, remaining
        return cfg

    cfg = _load_yaml(path)
    record: dict[str, Any] = {}
    for pth, raw in overrides_env.items():
        _set_path(cfg, pth, raw, record)
    for pth, raw in overrides_cli.items():
        _set_path(cfg, pth, raw, record)

    _last_config = deepcopy(cfg)
    _last_signature = signature
    _last_source = path.resolve()
    _last_overrides = record

    if with_cli:
        return deepcopy(cfg), remaining
    return deepcopy(cfg)


def get_config_source() -> Path | None:
    """Return the path of the last configuration file that was loaded."""

    return _last_source


def get_config_overrides() -> dict[str, Any]:
    """Return the overrides applied on top of the YAML defaults."""

    return deepcopy(_last_overrides)


# core/config.py - ADD THESE FUNCTIONS

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Return absolute project root."""
    cfg = get_config()
    root = Path(cfg.get("project", {}).get("root", "."))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    return root.resolve()


def get_exports_root() -> Path:
    """Return absolute exports root."""
    cfg = get_config()
    exports = cfg.get("project", {}).get("exports_root", "exports")
    root = get_project_root()
    return (root / exports).resolve()


def get_path(path_key: str, run_id: Optional[str] = None) -> Path:
    """
    Resolve a path from config by dotted key.

    Examples:
        get_path("ephemeris.raw")
        get_path("geometry.los", run_id="20251018T120000Z")

    If run_id is provided and path doesn't contain {run_id},
    it will be appended automatically.
    """
    cfg = get_config()
    paths = cfg.get("paths", {})

    # Navigate nested dict
    keys = path_key.split(".")
    current = paths
    for key in keys:
        if not isinstance(current, dict):
            raise KeyError(f"Path '{path_key}' not found in config")
        current = current.get(key)
        if current is None:
            raise KeyError(f"Path '{path_key}' not found in config")

    # Resolve to absolute
    path_str = str(current)
    if "{run_id}" in path_str:
        if run_id is None:
            raise ValueError(f"Path '{path_key}' requires run_id but none provided")
        path_str = path_str.format(run_id=run_id)

    path = Path(path_str)
    if not path.is_absolute():
        path = get_exports_root() / path

    # Auto-append run_id if requested but not in template
    if run_id and "{run_id}" not in str(current):
        path = path / run_id

    return path.resolve()


def get_pattern(pattern_key: str, **kwargs) -> str:
    """
    Get filename pattern from config with optional formatting.

    Example:
        get_pattern("los_csv", sat_id="SAT001", target_id="HGV_01")
        # Returns: "LOS_SAT001_HGV_01.csv"
    """
    cfg = get_config()
    patterns = cfg.get("patterns", {})
    pattern = patterns.get(pattern_key)
    if pattern is None:
        raise KeyError(f"Pattern '{pattern_key}' not found in config")
    return pattern.format(**kwargs) if kwargs else pattern

__all__ = ["get_config", "get_config_source", "get_config_overrides"]
