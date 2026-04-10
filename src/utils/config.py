import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_MARKERS = (
    ".git",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
)
_DEFAULT_CONFIG_RELATIVE = Path("config/default.yaml")
_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "data": (
        "train_path",
        "val_path",
        "test_path",
        "reference_path",
    ),
    "checkpoint": ("dir",),
    "inference": ("classifier_path", "encoder_path"),
    "sheet": ("save_path", "fonts_dir"),
}


def _find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve().parent).resolve()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in _MARKERS):
            return parent
    return Path.cwd().resolve()


def _resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _resolve_config_path(
    config_path: str | Path, project_root: Path
) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path

    project_candidate = project_root / path
    if project_candidate.exists():
        return project_candidate
    return path.resolve()


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config root must be a mapping in {path}, got {type(loaded).__name__}."
        )
    return loaded


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _apply_environment_overrides(config: dict[str, Any]) -> None:
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        return

    inference = config.setdefault("inference", {})
    if isinstance(inference, dict):
        inference["classifier_path"] = model_path


def _normalize_known_paths(
    config: dict[str, Any], project_root: Path
) -> None:
    for section_name, keys in _PATH_FIELDS.items():
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue

        for key in keys:
            value = section.get(key)
            if isinstance(value, str) and value.strip():
                section[key] = str(
                    _resolve_path(value.strip(), project_root)
                )


def _prepare_output_directories(config: dict[str, Any]) -> None:
    checkpoint_dir = config.get("checkpoint", {}).get("dir")
    if isinstance(checkpoint_dir, str) and checkpoint_dir.strip():
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    sheet_dir = config.get("sheet", {}).get("save_path")
    if isinstance(sheet_dir, str) and sheet_dir.strip():
        Path(sheet_dir).mkdir(parents=True, exist_ok=True)

    fonts_dir = config.get("sheet", {}).get("fonts_dir")
    if isinstance(fonts_dir, str) and fonts_dir.strip():
        Path(fonts_dir).mkdir(parents=True, exist_ok=True)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{
                key: _to_namespace(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def get_path(relative_path: str) -> str:
    project_root = _find_project_root()
    return str(_resolve_path(relative_path, project_root))


def load_config(config_path: str | Path | None = None) -> Any:
    """Load config using default.yaml as base and merge optional overrides."""
    project_root = _find_project_root()
    default_config_path = project_root / _DEFAULT_CONFIG_RELATIVE

    merged_config = copy.deepcopy(
        _load_yaml_dict(default_config_path)
    )

    if config_path is not None:
        override_path = _resolve_config_path(
            config_path, project_root
        )
        if override_path.resolve() != default_config_path.resolve():
            override_config = _load_yaml_dict(override_path)
            merged_config = _deep_merge(
                merged_config, override_config
            )

    _apply_environment_overrides(merged_config)
    _normalize_known_paths(merged_config, project_root)
    _prepare_output_directories(merged_config)
    return _to_namespace(merged_config)


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", "INFO")
