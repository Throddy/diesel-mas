"""An independent kinetic model of the hydrotreater, used as "the plant".

The optimizer reasons with model M, a sign-constrained log-linear regression
fitted to the whole training block.  If the simulator computed the response
with M as well, a closed-loop run would only show that the system agrees with
itself.  This module is deliberately a different thing:

* its form is the analytic solution of an n-th order reaction in a plug-flow
  reactor, not a regression;
* its temperature term is Arrhenius, its space-velocity term comes from the
  residence time, and hydrogen enters through a partial-pressure order;
* only one constant is calibrated, and it is calibrated to a single operating
  point rather than fitted across the data.

So the two models are wrong in different ways, which is the point: the gap
between them is the model error a recommendation has to survive.

Nothing here is a validated plant model.  Section "Проверка на истории" in
``scripts/validate_plant_model.py`` reports how far it lands from the
laboratory, and the numbers are stated rather than tuned away.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

R_GAS = 8.314462618


@dataclass(frozen=True)
class PlantParams:
    """Kinetics of the reactor.  Every value carries its origin in the docs.

    ``activation_energy_kj_mol`` is the literature figure already recorded as
    assumption A-19 and reused here; ``order`` and ``hydrogen_order`` are
    declared kinetic assumptions (A-25); ``pre_exponential`` is the single
    constant calibrated, and it is calibrated to one reference point.
    """

    activation_energy_kj_mol: float = 116.91
    order: float = 1.5
    hydrogen_order: float = 0.7
    pre_exponential: float = 1.0
    reference_temperature_c: float = 368.15
    reference_feed_tph: float = 214.86
    reference_pressure_mpa: float = 3.675
    reference_feed_sulfur_mgkg: float = 9343.0
    reference_product_sulfur_mgkg: float = 8.5

    delay_minutes: float = 360.0
    time_constant_minutes: float = 180.0

    def rate_constant(self, temperature_c: float) -> float:
        """Arrhenius rate constant at the given reactor temperature."""
        kelvin = float(temperature_c) + 273.15
        energy = self.activation_energy_kj_mol * 1000.0
        return self.pre_exponential * float(np.exp(-energy / (R_GAS * kelvin)))


class PlantModel:
    """Steady-state sulfur out of the reactor, plus its dynamic response."""

    def __init__(self, params: Optional[PlantParams] = None):
        self.p = params or PlantParams()

    def steady_state(
        self, feed_sulfur_mgkg: float, temperature_c: float, feed_tph: float, pressure_mpa: float
    ) -> float:
        """Product sulfur for a held operating point.

        Plug flow with an n-th order reaction integrates to

            S_out^(1-n) = S_in^(1-n) + (n - 1) k tau,

        where the residence time ``tau`` is inversely proportional to the feed
        rate and hydrogen enters as ``P^m``.  The reactor volume is absorbed
        into the pre-exponential factor, so it never has to be guessed.
        """
        n = float(self.p.order)
        s_in = max(float(feed_sulfur_mgkg), 1e-6)
        feed = max(float(feed_tph), 1e-6)
        pressure = max(float(pressure_mpa), 1e-6)
        k = self.p.rate_constant(temperature_c)
        tau = 1.0 / feed
        driving = (n - 1.0) * k * tau * pressure**self.p.hydrogen_order
        value = s_in ** (1.0 - n) + driving
        if value <= 0:
            return 1e-6
        return float(value ** (1.0 / (1.0 - n)))

    def calibrate(
        self,
        feed_sulfur_mgkg: Optional[float] = None,
        temperature_c: Optional[float] = None,
        feed_tph: Optional[float] = None,
        pressure_mpa: Optional[float] = None,
        product_sulfur_mgkg: Optional[float] = None,
    ) -> "PlantModel":
        """Pin the pre-exponential factor to one reference operating point.

        One equation, one unknown: no data set is fitted, so the calibration
        cannot absorb the patterns model M learned.
        """
        p = self.p
        s_in = float(
            feed_sulfur_mgkg if feed_sulfur_mgkg is not None else p.reference_feed_sulfur_mgkg
        )
        s_out = float(
            product_sulfur_mgkg
            if product_sulfur_mgkg is not None
            else p.reference_product_sulfur_mgkg
        )
        temperature = float(
            temperature_c if temperature_c is not None else p.reference_temperature_c
        )
        feed = float(feed_tph if feed_tph is not None else p.reference_feed_tph)
        pressure = float(pressure_mpa if pressure_mpa is not None else p.reference_pressure_mpa)

        n = float(p.order)
        required = (s_out ** (1.0 - n) - s_in ** (1.0 - n)) / (n - 1.0)
        kelvin = temperature + 273.15
        exponent = float(np.exp(-p.activation_energy_kj_mol * 1000.0 / (R_GAS * kelvin)))
        tau = 1.0 / feed
        pre = required / (exponent * tau * pressure**p.hydrogen_order)
        return PlantModel(
            replace(
                p,
                pre_exponential=float(pre),
                reference_temperature_c=temperature,
                reference_feed_tph=feed,
                reference_pressure_mpa=pressure,
                reference_feed_sulfur_mgkg=s_in,
                reference_product_sulfur_mgkg=s_out,
            )
        )

    def step(
        self,
        current_sulfur: float,
        target_sulfur: float,
        elapsed_minutes: float,
        step_minutes: float,
    ) -> float:
        """Relax towards the steady state after the transport delay.

        ``elapsed_minutes`` counts from the moment the controls were changed;
        before the delay has passed nothing has reached the analyser yet.
        """
        if elapsed_minutes < self.p.delay_minutes:
            return float(current_sulfur)
        tau = max(float(self.p.time_constant_minutes), 1.0)
        alpha = 1.0 - float(np.exp(-float(step_minutes) / tau))
        return float(current_sulfur + alpha * (float(target_sulfur) - float(current_sulfur)))

    def temperature_sensitivity(
        self,
        feed_sulfur_mgkg: float,
        temperature_c: float,
        feed_tph: float,
        pressure_mpa: float,
        delta_c: float = 1.0,
    ) -> float:
        """d ln S / dT at an operating point, for comparison with model M."""
        base = self.steady_state(feed_sulfur_mgkg, temperature_c, feed_tph, pressure_mpa)
        hotter = self.steady_state(
            feed_sulfur_mgkg, temperature_c + delta_c, feed_tph, pressure_mpa
        )
        return float((np.log(hotter) - np.log(base)) / delta_c)

    def describe(self) -> dict:
        return {
            "form": "плуг-флоу, реакция порядка n, Аррениус, P^m по водороду",
            "order": self.p.order,
            "hydrogen_order": self.p.hydrogen_order,
            "activation_energy_kj_mol": self.p.activation_energy_kj_mol,
            "pre_exponential": self.p.pre_exponential,
            "delay_minutes": self.p.delay_minutes,
            "time_constant_minutes": self.p.time_constant_minutes,
            "calibration_point": {
                "temperature_c": self.p.reference_temperature_c,
                "feed_tph": self.p.reference_feed_tph,
                "pressure_mpa": self.p.reference_pressure_mpa,
                "feed_sulfur_mgkg": self.p.reference_feed_sulfur_mgkg,
                "product_sulfur_mgkg": self.p.reference_product_sulfur_mgkg,
            },
        }
