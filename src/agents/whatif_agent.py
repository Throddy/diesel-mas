"""WhatIfAgent: owns the response model M and answers what-if questions.

The optimizer used to call the response model directly, so the journal showed
only the resulting number.  Routing the question through this agent records
what was asked - which controls moved, from which state - and what came back,
including the reason a move could not be evaluated at all.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.config import load_config


class WhatIfAgent:
    name = "WhatIfAgent"

    def __init__(self, response_model, cfg=None):
        self.cfg = cfg or load_config()
        self.model = response_model

    @property
    def diagnostics(self) -> dict:
        return dict(getattr(self.model, "diagnostics", {}))

    def handle(self, request: Mapping[str, Any]) -> dict:
        """Answer one what-if: the log change of sulfur for a proposed state.

        ``central`` is the model's estimate and ``pessimistic`` the block
        bootstrap's upper branch.  A move outside the fitted support is
        refused rather than extrapolated, and the refusal is part of the
        answer.
        """
        before = dict(request["before"])
        after = dict(request["after"])
        if request.get("is_do_nothing"):
            return {
                "action_id": request["action_id"],
                "supported": True,
                "central": 0.0,
                "pessimistic": 0.0,
                "note": "режим не меняется",
            }
        try:
            central, pessimistic = self.model.effect(before, after)
        except ValueError as exc:
            return {
                "action_id": request["action_id"],
                "supported": False,
                "central": 0.0,
                "pessimistic": 0.0,
                "note": str(exc),
            }
        return {
            "action_id": request["action_id"],
            "supported": True,
            "central": float(central),
            "pessimistic": float(pessimistic),
            "note": "наблюдательная модель со знаками; верхняя граница по блочному бутстрепу",
        }

    def sulfur_after(self, quality, central: float, pessimistic: float) -> dict:
        """Translate a log change into the forecast trio the card shows."""
        point = quality.point_forecast * float(np.exp(central))
        lower = quality.lower * float(np.exp(min(central, pessimistic)))
        upper = quality.upper * float(np.exp(max(central, pessimistic)))
        return {
            "predicted_sulfur": point,
            "predicted_sulfur_lower": lower,
            "predicted_sulfur_upper": upper,
        }
