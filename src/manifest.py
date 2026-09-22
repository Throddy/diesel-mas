"""Run manifest: what was run, on which code, with which inputs.

Every number published in README or docs must be traceable to a file produced
by a specific run of a specific commit.  ``write_manifest`` records that link;
``scripts/render_results.py`` reads it when it renders figures into text, so a
number can never be typed in by hand.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, project_root

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "pyarrow",
    "streamlit",
    "PyYAML",
)


def _git(*args: str) -> str | None:
    """Run a git command in the project root; None when git or the repo is absent."""
    try:
        out = subprocess.run(
            ("git", *args), cwd=project_root(), capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def file_digest(path: Path) -> dict | None:
    """SHA-256 and size of one file, or None when it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _digests(root: Path, patterns: tuple[str, ...]) -> dict:
    found = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            digest = file_digest(path)
            if digest:
                found[str(path.relative_to(project_root()))] = digest
    return found


def build_manifest(cfg=None) -> dict:
    """Collect code, configuration, input and output fingerprints for this run."""
    cfg = cfg or load_config()
    root = project_root()
    dirty = _git("status", "--porcelain")
    raw_inputs = {}
    for key in cfg.main["raw_files"]:
        try:
            path = cfg.raw_file(key)
        except (FileNotFoundError, ValueError):
            raw_inputs[key] = None
            continue
        digest = file_digest(path)
        raw_inputs[key] = None if digest is None else {"name": path.name, **digest}
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": cfg.main["project"],
        "code": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_clean": (dirty == "") if dirty is not None else None,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "config": _digests(root / "config", ("*.yaml",)),
        "raw_inputs": raw_inputs,
        "models": _digests(root / cfg.main["paths"]["models"], ("*.joblib", "*.json")),
        "reports": _digests(
            root / cfg.main["paths"]["reports"], ("*.json", "*.csv", "periods/*/*.json")
        ),
        "split": cfg.main["split"],
        "feature_flags": cfg.main["features"],
    }


def write_manifest(cfg=None) -> Path:
    """Write ``manifest.json`` next to the reports and return its path."""
    cfg = cfg or load_config()
    manifest = build_manifest(cfg)
    out = project_root() / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(write_manifest())
