"""Numbers quoted in the documents must match the reports that produced them.

The documents are written by hand, so a figure can go stale the moment a run
changes it.  These checks read the reports and look for the rendered value in
the text: they catch the stale number, not the wording.
"""

import json

import pytest

from src.config import load_config, project_root

CFG = load_config()
REPORTS = project_root() / CFG.main["paths"]["reports"]
README = project_root() / "README.md"


def number(value, digits=2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def read(name: str):
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"нет отчёта {name}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_headline_forecast_metrics_match_the_report(readme):
    test = read("evaluation_report.json")["model"]["test"]
    for value, digits in (
        (test["regression"]["MAE"], 2),
        (test["regression"]["R2"], 2),
        (test["interval"]["coverage"], 2),
        (test["interval"]["mean_width"], 2),
    ):
        assert number(value, digits) in readme, (value, digits)


def test_architecture_table_matches_the_report(readme):
    for row in read("arch_compare.json")["configurations"]:
        share = f"{row['abstention_rate'] * 100:.0f} %"
        assert share in readme, (row["configuration"], share)
        assert str(row["unsafe_candidates_rejected"]) in readme, row["configuration"]


def test_unconstrained_response_coefficients_match_the_report(readme):
    report = read("response_unconstrained.json")
    for value in report["coefficients"].values():
        assert number(value, 4) in readme, value
    assert number(report["temperature_log_sensitivity"]["unconstrained_E"], 4) in readme


def test_closed_loop_table_matches_the_report(readme):
    for block in read("closed_loop_architectures.json")["scenarios"].values():
        for row in block["policies"].values():
            assert str(row["minutes_off_spec"]) in readme, row
            assert number(row["sulfur_final_mgkg"]) in readme, row


def test_feature_block_contributions_match_the_report(readme):
    for row in read("feature_blocks.json")["blocks"].values():
        assert number(row["test"]["MAE"], 3) in readme, row["test"]["MAE"]


def test_period_breakdown_matches_the_report(readme):
    periods = read("forecast_improvement.json")["by_period"]
    for name in ("базовый набор признаков", "блоки признаков"):
        for row in periods[name].values():
            assert number(row["MAE"], 3) in readme, (name, row["MAE"])


def test_architecture_e_explanation_matches_the_report(readme):
    """The figures explaining why configuration E abstains must be current."""
    for row in read("arch_compare.json")["configurations"]:
        if row["configuration"] not in ("C", "E"):
            continue
        assert number(row["median_best_upper_at_risk"]) in readme, row["configuration"]
    reasons = next(
        r for r in read("arch_compare.json")["configurations"] if r["configuration"] == "E"
    )["abstention_reasons"]
    limit_reason = next(v for k, v in reasons.items() if "верхней границе" in k)
    assert str(limit_reason) in readme


def test_decision_rule_figures_match_the_report(readme):
    rule = read("decision_rules.json")["threshold_rule"]
    assert number(rule["median_margin_below_limit"]) in readme
    assert f"{rule['abstention_rate']:.3f}".replace(".", ",") in readme
    assert f"{rule['action_rate_when_at_risk']:.3f}".replace(".", ",") in readme


def test_plant_model_validation_figures_match_the_report(readme):
    block = read("plant_model_validation.json")["blocks"]["test"]
    assert number(block["MAE_mgkg"]) in readme
    assert number(block["median_ratio"]) in readme


def test_temperature_path_figures_match_the_report(readme):
    report = read("temperature_path.json")
    slope = report["sensitivity_d_ln_s_dt"]["forecast_model_N"]
    assert f"{slope:.5f}".replace(".", ",").replace("-0", "-0") in readme
    for row in report["forecast_model_N"]:
        assert number(row["forecast_mgkg"], 3) in readme, row


def test_horizon_table_matches_the_report(readme):
    for row in read("horizons_evaluation.json")["horizons"]:
        if row.get("status") != "OK":
            continue
        for key, digits in (
            ("MAE", 3),
            ("baseline_previous_lims_MAE", 3),
            ("MedianAE", 3),
            ("RMSE", 3),
            ("R2", 3),
            ("interval_coverage", 3),
            ("brier_probability_above_limit", 3),
            ("brier_base_rate", 3),
        ):
            assert number(row[key], digits) in readme, (row["horizon_hours"], key)


def test_soft_sensor_table_matches_the_report(readme):
    for row in read("soft_sensors.json")["targets"]:
        assert row["metric"] in readme, row["metric"]
        assert str(row["n_observations"]) in readme, row["metric"]
        if row.get("MAE") is not None:
            assert number(row["MAE"], 3) in readme, row["metric"]
        assert row["status"] in readme, row["metric"]


def test_no_report_placeholder_leaked_into_the_summary():
    """A missing report must be visible, not silently rendered as a number."""
    summary = REPORTS / "results_summary.md"
    if not summary.is_file():
        pytest.skip("сводка не построена")
    assert "нет отчёта" not in summary.read_text(encoding="utf-8")


def test_every_markdown_file_passes_the_style_check():
    from scripts.check_docs import check, links, markdown_files

    problems = []
    for path in markdown_files():
        problems += check(path) + links(path)
    assert not problems, problems[:10]
