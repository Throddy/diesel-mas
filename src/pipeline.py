"""System assembly: one call gives a ready-to-use orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.agents.data_quality_agent import DataQualityAgent
from src.agents.optimization_agent import OptimizationAgent
from src.agents.orchestrator import Orchestrator
from src.agents.quality_agent import QualityAgent
from src.agents.reliability_agent import ReliabilityAgent
from src.agents.safety_agent import SafetyAgent
from src.agents.whatif_agent import WhatIfAgent
from src.config import load_config, project_root
from src.data.loaders import (
    clean_telemetry,
    feed_sulfur,
    pak_d15,
    pak_sulfur,
    product_t95,
    split_bounds,
    target_series,
)
from src.data.store import load_df
from src.models.quality import SulfurModel
from src.models.reliability import ReliabilityModel
from src.models.response_model import ResponseModel


@dataclass
class System:
    cfg: object
    telemetry: pd.DataFrame
    lims_long: pd.DataFrame
    pak_long: pd.DataFrame
    target: pd.Series
    pak: pd.Series
    pak_density: pd.Series
    model: SulfurModel
    reliability: ReliabilityModel
    orchestrator: Orchestrator

    def decide(self, t, log: bool = True, recent_moves=None):
        return self.orchestrator.decide(t, log=log, recent_moves=recent_moves)


def build_system(cfg=None) -> System:
    cfg = cfg or load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    models_dir = project_root() / cfg.main["paths"]["models"]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims = load_df(interim / "lims_long.pkl")
    pak = load_df(interim / "pak_long.pkl")
    y = target_series(lims, cfg)
    ps, pdn = pak_sulfur(pak), pak_d15(pak)
    model = SulfurModel.load(models_dir / "sulfur_model.joblib")
    rel = ReliabilityModel.load(models_dir / "reliability_model.joblib")
    importance = None
    imp_path = models_dir / "feature_importance.pkl"
    if imp_path.exists():
        importance = load_df(imp_path)

    train_end, _, _ = split_bounds(cfg)
    medians = tel.loc[:train_end].median(numeric_only=True).to_dict()

    feed = feed_sulfur(lims)
    dq = DataQualityAgent(tel, y, ps, pdn, lims_long=lims, cfg=cfg, feed_sulfur=feed)
    qa = QualityAgent(tel, y, ps, pdn, model, cfg=cfg, importance=importance)
    qa.lims_other = {"feed_sulfur": feed, "product_t95": product_t95(lims)}
    ra = ReliabilityAgent(rel, cfg=cfg)
    whatif = WhatIfAgent(ResponseModel.load(models_dir / "response_model.joblib"), cfg=cfg)
    oa = OptimizationAgent(qa, ra, medians, cfg=cfg, whatif_agent=whatif)
    sa = SafetyAgent(cfg=cfg)
    orch = Orchestrator(dq, qa, ra, oa, sa, cfg=cfg)
    return System(cfg, tel, lims, pak, y, ps, pdn, model, rel, orch)
