"""Test runner.

Delegates to pytest, which is pinned in requirements.txt.  Only when pytest is
genuinely absent does a tiny built-in shim run the same files with plain
Python; the shim cannot provide fixtures, so tests that take arguments are
reported as skipped-with-reason rather than failed - and the summary says so,
because a runner that hides tests is worse than no runner.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_shim() -> None:
    if importlib.util.find_spec("pytest") is not None:
        return
    mod = types.ModuleType("pytest")

    class _Skipped(Exception):
        pass

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, t, v, tb):
            if t is None:
                raise AssertionError(f"expected {self.exc.__name__} but nothing was raised")
            return issubclass(t, self.exc)

    class _Approx:
        def __init__(self, value, rel=1e-6, abs=1e-9):
            self.value, self.rel, self.abs = value, rel, abs

        def __eq__(self, other):
            return abs(float(other) - float(self.value)) <= max(
                self.abs, self.rel * abs(float(self.value))
            )

        def __repr__(self):
            return f"approx({self.value})"

    mod.raises = lambda exc: _Raises(exc)
    mod.approx = lambda v, rel=1e-6, abs=1e-9: _Approx(v, rel, abs)
    mod.skip = lambda reason="": (_ for _ in ()).throw(_Skipped(reason))
    mod.Skipped = _Skipped
    sys.modules["pytest"] = mod


def _run_with_pytest() -> int:
    """Preferred path: the real runner, with fixtures and parametrisation."""
    import pytest

    return int(pytest.main([str(ROOT / "tests"), "-q"]))


def main() -> int:
    if importlib.util.find_spec("pytest") is not None:
        return _run_with_pytest()
    print(
        "pytest не установлен: запуск встроенным упрощённым раннером; "
        "тесты, которым нужны фикстуры, будут пропущены"
    )
    _install_shim()
    import pytest

    skipped_exc = getattr(pytest, "Skipped", None) or getattr(pytest.skip, "Exception", Exception)
    passed = failed = skipped = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            fn = getattr(module, name)
            if inspect.signature(fn).parameters:
                skipped += 1
                print(f"  SKIP  {path.name}::{name}: нужен pytest (фикстуры)")
                continue
            try:
                fn()
                passed += 1
                print(f"  PASS  {path.name}::{name}")
            except Exception as exc:
                if skipped_exc and isinstance(exc, skipped_exc):
                    skipped += 1
                    print(f"  SKIP  {path.name}::{name}: {exc}")
                    continue
                failed += 1
                print(f"  FAIL  {path.name}::{name}: {exc}")
                traceback.print_exc(limit=2)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
