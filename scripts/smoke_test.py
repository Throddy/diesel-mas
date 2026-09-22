"""Fast end-to-end check: canonical layer -> agents -> recommendation."""

from __future__ import annotations

import time

import pandas as pd

from src.config import load_config
from src.pipeline import build_system


def main() -> int:
    cfg = load_config()
    t0 = time.time()
    system = build_system(cfg)
    print(
        f"[ok] system built in {time.time()-t0:.1f}s; telemetry {system.telemetry.shape}, "
        f"LIMS target {len(system.target)}, PAK {len(system.pak)}"
    )
    t = pd.Timestamp(cfg.main["demo"]["stable"])
    t0 = time.time()
    rec = system.decide(t)
    print(f"[ok] decision cycle in {time.time()-t0:.1f}s")
    print(f"     headline : {rec.headline}")
    print(f"     abstained: {rec.abstained}")
    print(
        f"     candidates: {rec.agent_trace['OptimizationAgent']['n_candidates']}, "
        f"feasible: {len(rec.agent_trace['SafetyAgent']['feasible_ids'])}"
    )
    assert rec.agent_trace["OptimizationAgent"]["n_candidates"] >= 2, "DO NOTHING must always exist"
    assert any(c["is_do_nothing"] for c in rec.agent_trace["OptimizationAgent"]["candidates"])
    print("[ok] smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
