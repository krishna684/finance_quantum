"""Tests for pipeline.report — markdown report generation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_metrics_csv(path: Path, models: list[str], n_folds: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "fold", "n_train", "n_test",
                        "accuracy", "f1", "auc_roc", "precision", "recall"],
        )
        writer.writeheader()
        for model in models:
            for fold in range(n_folds):
                writer.writerow({
                    "model": model, "fold": fold,
                    "n_train": 100, "n_test": 20,
                    "accuracy": 0.62, "f1": 0.58,
                    "auc_roc": 0.65, "precision": 0.63, "recall": 0.55,
                })


def _write_summary_csv(path: Path, models: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "ann_return", "ann_vol", "sharpe", "max_drawdown", "n_days"],
        )
        writer.writeheader()
        for model in models:
            writer.writerow({
                "model": model,
                "ann_return": "0.085",
                "ann_vol": "0.15",
                "sharpe": "0.57",
                "max_drawdown": "-0.12",
                "n_days": "200",
            })


def _write_alphas_csv(path: Path, models: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "factor_model", "alpha", "alpha_tstat", "alpha_pvalue",
                "mkt_beta", "smb_beta", "hml_beta", "umd_beta", "rmw_beta", "cma_beta", "r_squared",
            ],
        )
        writer.writeheader()
        for model in models:
            for fm in ("capm", "ff3", "ff5", "ff5_umd"):
                writer.writerow({
                    "model": model, "factor_model": fm,
                    "alpha": "0.00020",
                    "alpha_tstat": "2.10",
                    "alpha_pvalue": "0.036",
                    "mkt_beta": "0.05",
                    "smb_beta": "0.02",
                    "hml_beta": "-0.01",
                    "umd_beta": "0.03",
                    "rmw_beta": "0.01",
                    "cma_beta": "0.01",
                    "r_squared": "0.12",
                })


# ---------------------------------------------------------------------------
# _aggregate_metrics
# ---------------------------------------------------------------------------

def test_aggregate_metrics_correct_mean():
    from pipeline.report import _aggregate_metrics

    rows = [
        {"model": "svm_rbf", "fold": "0", "accuracy": "0.60", "f1": "0.55",
         "auc_roc": "0.63", "precision": "0.61", "recall": "0.52"},
        {"model": "svm_rbf", "fold": "1", "accuracy": "0.62", "f1": "0.57",
         "auc_roc": "0.65", "precision": "0.63", "recall": "0.54"},
    ]
    agg = _aggregate_metrics(rows)
    assert "svm_rbf" in agg
    assert abs(agg["svm_rbf"]["accuracy"]["mean"] - 0.61) < 1e-9
    assert agg["svm_rbf"]["accuracy"]["n"] == 2.0


def test_aggregate_metrics_multiple_models():
    from pipeline.report import _aggregate_metrics

    rows = [
        {"model": "m1", "fold": "0", "accuracy": "0.60", "f1": "0.50",
         "auc_roc": "0.60", "precision": "0.60", "recall": "0.50"},
        {"model": "m2", "fold": "0", "accuracy": "0.70", "f1": "0.65",
         "auc_roc": "0.70", "precision": "0.70", "recall": "0.65"},
    ]
    agg = _aggregate_metrics(rows)
    assert "m1" in agg
    assert "m2" in agg
    assert agg["m2"]["accuracy"]["mean"] > agg["m1"]["accuracy"]["mean"]


def test_aggregate_metrics_empty():
    from pipeline.report import _aggregate_metrics

    assert _aggregate_metrics([]) == {}


# ---------------------------------------------------------------------------
# _best_in_col
# ---------------------------------------------------------------------------

def test_best_in_col():
    from pipeline.report import _best_in_col

    agg = {
        "svm": {"auc_roc": {"mean": 0.62}},
        "xgb": {"auc_roc": {"mean": 0.70}},
        "rf":  {"auc_roc": {"mean": 0.65}},
    }
    assert _best_in_col(agg, "auc_roc") == "xgb"


# ---------------------------------------------------------------------------
# _sig_stars
# ---------------------------------------------------------------------------

def test_sig_stars():
    from pipeline.report import _sig_stars

    assert _sig_stars(0.001) == "***"
    assert _sig_stars(0.03) == "**"
    assert _sig_stars(0.08) == "*"
    assert _sig_stars(0.20) == ""
    import math
    assert _sig_stars(math.nan) == ""


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------

def test_fmt_normal():
    from pipeline.report import _fmt

    assert _fmt(0.12345, 3) == "0.123"
    assert _fmt(-0.12345, 3) == "-0.123"


def test_fmt_nan():
    from pipeline.report import _fmt
    import math

    assert _fmt(math.nan) == "—"
    assert _fmt(math.inf) == "—"


# ---------------------------------------------------------------------------
# ReportGenerator.run
# ---------------------------------------------------------------------------

def test_report_generator_creates_file(tmp_path):
    from pipeline.report import ReportGenerator

    results_dir = tmp_path / "results"
    clf_dir = results_dir / "classifiers"
    port_dir = results_dir / "portfolio"

    models = ["svm_rbf", "xgboost", "random_forest"]
    _write_metrics_csv(clf_dir / "metrics_image.csv", models)
    _write_summary_csv(port_dir / "summary_stats_image.csv", models)
    _write_alphas_csv(port_dir / "factor_alphas_image.csv", models)

    gen = ReportGenerator(results_dir=results_dir)
    out = gen.run()

    assert out.exists(), "report.md not created"
    content = out.read_text(encoding="utf-8")
    assert "# Classical Analysis Pipeline" in content


def test_report_contains_all_sections(tmp_path):
    from pipeline.report import ReportGenerator

    results_dir = tmp_path / "results"
    clf_dir = results_dir / "classifiers"
    port_dir = results_dir / "portfolio"

    models = ["xgboost"]
    _write_metrics_csv(clf_dir / "metrics_image.csv", models)
    _write_summary_csv(port_dir / "summary_stats_image.csv", models)
    _write_alphas_csv(port_dir / "factor_alphas_image.csv", models)

    gen = ReportGenerator(results_dir=results_dir)
    out = gen.run()
    content = out.read_text()

    assert "## 1. Classification Performance" in content
    assert "## 2. Portfolio Performance" in content
    assert "## 3. Factor Model Alphas" in content
    assert "## 4. Key Findings" in content
    assert "## 5. Notes" in content


def test_report_model_names_in_output(tmp_path):
    from pipeline.report import ReportGenerator

    results_dir = tmp_path / "results"
    models = ["svm_rbf", "xgboost", "mlp"]
    _write_metrics_csv(results_dir / "classifiers" / "metrics_image.csv", models)

    gen = ReportGenerator(results_dir=results_dir)
    content = gen.run().read_text()

    for model in models:
        assert model in content, f"Model {model} not found in report"


def test_report_significant_alpha_highlighted(tmp_path):
    """Significant alphas (|t| >= 1.96) should appear in key findings."""
    from pipeline.report import ReportGenerator

    results_dir = tmp_path / "results"
    clf_dir = results_dir / "classifiers"
    port_dir = results_dir / "portfolio"

    models = ["xgboost"]
    _write_metrics_csv(clf_dir / "metrics_image.csv", models)
    _write_alphas_csv(port_dir / "factor_alphas_image.csv", models)  # t-stat = 2.10 > 1.96

    gen = ReportGenerator(results_dir=results_dir)
    content = gen.run().read_text()

    assert "Significant FF5+UMD alphas" in content


def test_report_empty_results(tmp_path):
    """Report should still render with placeholder text when no results exist."""
    from pipeline.report import ReportGenerator

    gen = ReportGenerator(results_dir=tmp_path / "empty_results")
    out = gen.run()
    content = out.read_text()

    assert "# Classical Analysis Pipeline" in content
    assert "No classification results found" in content or "No data" in content


def test_report_custom_output_path(tmp_path):
    from pipeline.report import ReportGenerator

    out_path = tmp_path / "custom_report.md"
    gen = ReportGenerator(results_dir=tmp_path / "r")
    written = gen.run(output_path=out_path)

    assert written == out_path
    assert out_path.exists()


def test_report_multiple_embedding_types(tmp_path):
    """Report should include a section for each embedding type found."""
    from pipeline.report import ReportGenerator

    results_dir = tmp_path / "results"
    for emb_type in ("image", "text", "multimodal"):
        _write_metrics_csv(
            results_dir / "classifiers" / f"metrics_{emb_type}.csv",
            ["xgboost"],
        )

    gen = ReportGenerator(results_dir=results_dir)
    content = gen.run().read_text()

    assert "Image" in content
    assert "Text" in content
    assert "Multimodal" in content
