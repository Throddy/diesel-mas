"""Configuration loading. All thresholds live in config/*.yaml, never in code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    env = os.environ.get("DIESEL_MAS_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def _load_yaml(name: str) -> Dict[str, Any]:
    with open(project_root() / "config" / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Config:
    main: Dict[str, Any]
    constraints: Dict[str, Any]
    aliases: Dict[str, Any]
    controls: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.main["project"]["random_seed"])

    @property
    def model_version(self) -> str:
        return str(self.main["project"]["model_version"])

    @property
    def sulfur_limit(self) -> float:
        return float(self.main["quality"]["sulfur_limit_mgkg"])

    def path(self, key: str) -> Path:
        p = project_root() / self.main["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def raw_file(self, key: str) -> Path:
        """Locate one organiser file by exact name, then by its documented pattern.

        Sources are looked up in ``data/raw`` or in the directory given by
        ``--data-dir`` (exported as ``DIESEL_MAS_DATA_DIR``); later updates the
        organisers sent may sit in an ``organizer_updates`` subdirectory.  The
        directory is only ever read from - nothing is written beside the
        originals (K-44).
        """
        base = Path(
            os.environ.get("DIESEL_MAS_DATA_DIR", project_root() / self.main["paths"]["raw"])
        )
        roots = [base, base / "organizer_updates"]
        name = self.main["raw_files"][key]
        for root in roots:
            exact = root / name
            if exact.is_file():
                return exact
            pattern = self.main.get("raw_patterns", {}).get(key, name)
            matches = sorted(root.glob(pattern)) if root.exists() else []
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Неоднозначный источник {key}: {matches}")
        raise FileNotFoundError(
            f"Файл организаторов «{name}» (ключ {key}) не найден в {[str(r) for r in roots]}. "
            "Положите все файлы пакета в data/raw или укажите каталог через --data-dir."
        )

    def dq(self, key: str) -> Any:
        return self.main["data_quality"][key]


@lru_cache(maxsize=1)
def load_config() -> Config:
    controls_path = project_root() / "config" / "controls.yaml"
    controls = _load_yaml("controls.yaml") if controls_path.exists() else {"controls": []}
    return Config(
        main=_load_yaml("config.yaml"),
        constraints=_load_yaml("constraints.yaml"),
        aliases=_load_yaml("aliases.yaml"),
        controls=controls,
    )
