from __future__ import annotations

"""Summary report generator for the classical analysis pipeline.

Reads the CSV outputs produced by classifiers.py and portfolio.py and
generates a single Markdown report at results/report.md.

Sections
--------
    1. Classification Performance
       — Per-embedding-type table: model × metric (mean ± std across folds)
       — Bold the best model per column

    2. Portfolio Performance
       — Table: model × (ann_return, ann_vol, sharpe, max_drawdown)

    3. Factor Model Alphas
       — Table: model × factor_model (alpha, t-stat, p-value)

    4. Key Findings
       — Best classical model by AUC-ROC
       — Highest Sharpe portfolio
       — Statistically significant alphas (|t| > 1.96)

Usage
-----
    from pipeline.report import ReportGenerator
    from pathlib import Path

    gen = ReportGenerator(results_dir=Path("results"))
    gen.run()                        # writes results/report.md
    gen.run(output_path=Path("report.md"))
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from pipeline.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(val: str | None, default: float = float("nan")) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _fmt(val: float, decimals: int = 4) -> str:
    if math.isnan(val) or math.isinf(val):
        return "—"
    return f"{val:.{decimals}f}"


def _sig_stars(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_metrics(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate per-fold metrics into {model: {metric: {mean, std}}}."""
    import statistics

    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_cols = ["accuracy", "f1", "auc_roc", "precision", "recall"]
    for row in rows:
        model = row.get("model", "unknown")
        for col in metric_cols:
            v = _float(row.get(col))
            if not math.isnan(v):
                by_model[model][col].append(v)

    result: dict[str, dict[str, dict[str, float]]] = {}
    for model, metrics in sorted(by_model.items()):
        result[model] = {}
        for col, vals in metrics.items():
            if vals:
                result[model][col] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "n": float(len(vals)),
                }
    return result


def _best_in_col(agg: dict[str, dict[str, dict[str, float]]], col: str) -> str:
    """Return the model name with the highest mean for *col*."""
    best_model = ""
    best_val = float("-inf")
    for model, metrics in agg.items():
        v = metrics.get(col, {}).get("mean", float("-inf"))
        if v > best_val:
            best_val = v
            best_model = model
    return best_model


# ---------------------------------------------------------------------------
# Markdown table builders
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "auc_roc": "AUC-ROC",
    "precision": "Precision",
    "recall": "Recall",
}


def _classification_table(agg: dict[str, dict[str, dict[str, float]]], caption: str) -> str:
    if not agg:
        return f"*No data for {caption}.*\n"

    cols = ["accuracy", "f1", "auc_roc", "precision", "recall"]
    best = {col: _best_in_col(agg, col) for col in cols}

    header = "| Model | " + " | ".join(_METRIC_LABELS[c] for c in cols) + " |"
    sep = "|-------|" + "|".join(["-------"] * len(cols)) + "|"
    lines = [f"**{caption}**", "", header, sep]

    for model, metrics in sorted(agg.items()):
        cells = []
        for col in cols:
            m = metrics.get(col, {}).get("mean", float("nan"))
            s = metrics.get(col, {}).get("std", float("nan"))
            cell = f"{_fmt(m, 3)}±{_fmt(s, 3)}"
            if model == best[col]:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _portfolio_table(rows: list[dict]) -> str:
    if not rows:
        return "*No portfolio results found.*\n"

    header = "| Model | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | N Days |"
    sep = "|-------|-------------|----------|--------|--------------|--------|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row.get('model', '?')} "
            f"| {_fmt(_float(row.get('ann_return')), 3)} "
            f"| {_fmt(_float(row.get('ann_vol')), 3)} "
            f"| {_fmt(_float(row.get('sharpe')), 3)} "
            f"| {_fmt(_float(row.get('max_drawdown')), 3)} "
            f"| {row.get('n_days', '?')} |"
        )
    return "\n".join(lines) + "\n"


def _alpha_table(rows: list[dict], factor_model: str = "ff5_umd") -> str:
    filt = [r for r in rows if r.get("factor_model") == factor_model]
    if not filt:
        return f"*No {factor_model.upper()} alpha results found.*\n"

    header = "| Model | α (daily) | t-stat | p-value | Mkt β | SMB β | HML β | UMD β | R² |"
    sep = "|-------|-----------|--------|---------|-------|-------|-------|-------|----|"
    lines = [header, sep]
    for row in filt:
        alpha = _float(row.get("alpha"))
        tstat = _float(row.get("alpha_tstat"))
        pval = _float(row.get("alpha_pvalue"))
        stars = _sig_stars(pval)
        lines.append(
            f"| {row.get('model', '?')} "
            f"| {_fmt(alpha, 5)}{stars} "
            f"| {_fmt(tstat, 3)} "
            f"| {_fmt(pval, 4)} "
            f"| {_fmt(_float(row.get('mkt_beta')), 3)} "
            f"| {_fmt(_float(row.get('smb_beta')), 3)} "
            f"| {_fmt(_float(row.get('hml_beta')), 3)} "
            f"| {_fmt(_float(row.get('umd_beta')), 3)} "
            f"| {_fmt(_float(row.get('r_squared')), 3)} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Key findings
# ---------------------------------------------------------------------------

def _key_findings(
    agg_by_type: dict[str, dict[str, dict[str, dict[str, float]]]],
    summary_rows: list[dict],
    alpha_rows: list[dict],
) -> str:
    lines = []

    # Best classifier by AUC-ROC
    best_auc_model = ""
    best_auc_val = float("-inf")
    best_auc_type = ""
    for emb_type, agg in agg_by_type.items():
        model = _best_in_col(agg, "auc_roc")
        if model:
            val = agg[model].get("auc_roc", {}).get("mean", float("-inf"))
            if val > best_auc_val:
                best_auc_val = val
                best_auc_model = model
                best_auc_type = emb_type

    if best_auc_model:
        lines.append(
            f"- **Best classifier**: `{best_auc_model}` on `{best_auc_type}` embeddings "
            f"(mean AUC-ROC = {_fmt(best_auc_val, 3)})"
        )

    # Highest Sharpe
    if summary_rows:
        best_row = max(summary_rows, key=lambda r: _float(r.get("sharpe"), float("-inf")))
        sharpe = _float(best_row.get("sharpe"))
        if not math.isnan(sharpe):
            lines.append(
                f"- **Highest Sharpe**: `{best_row.get('model', '?')}` "
                f"(Sharpe = {_fmt(sharpe, 3)}, "
                f"Ann. Return = {_fmt(_float(best_row.get('ann_return')), 3)})"
            )

    # Significant alphas
    sig_alphas = [
        r for r in alpha_rows
        if r.get("factor_model") == "ff5_umd" and abs(_float(r.get("alpha_tstat"))) >= 1.96
    ]
    if sig_alphas:
        lines.append(f"- **Significant FF5+UMD alphas** (|t| ≥ 1.96):")
        for r in sig_alphas:
            tstat = _float(r.get("alpha_tstat"))
            pval = _float(r.get("alpha_pvalue"))
            lines.append(
                f"  - `{r.get('model', '?')}`: "
                f"α = {_fmt(_float(r.get('alpha')), 5)}{_sig_stars(pval)}, "
                f"t = {_fmt(tstat, 2)}"
            )
    else:
        lines.append("- No statistically significant FF5+UMD alphas at 5% level.")

    return "\n".join(lines) if lines else "*Insufficient data for findings.*"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)

    def run(self, output_path: Optional[Path] = None) -> Path:
        """Generate report.md and return its path."""
        clf_dir = self.results_dir / "classifiers"
        port_dir = self.results_dir / "portfolio"
        out = output_path or (self.results_dir / "report.md")

        # Discover embedding types from metrics files
        emb_types = sorted(
            p.stem.replace("metrics_", "")
            for p in clf_dir.glob("metrics_*.csv")
        ) if clf_dir.exists() else []

        agg_by_type: dict[str, dict] = {}
        classification_sections: list[str] = []
        for emb_type in emb_types:
            rows = _load_csv(clf_dir / f"metrics_{emb_type}.csv")
            agg = _aggregate_metrics(rows)
            agg_by_type[emb_type] = agg
            classification_sections.append(
                f"### {emb_type.capitalize()} embeddings\n\n"
                + _classification_table(agg, f"{emb_type.capitalize()} — Walk-Forward CV")
            )

        # Portfolio stats (aggregate across all embedding types)
        all_summary_rows: list[dict] = []
        portfolio_sections: list[str] = []
        for emb_type in emb_types:
            rows = _load_csv(port_dir / f"summary_stats_{emb_type}.csv")
            if rows:
                all_summary_rows.extend(rows)
                portfolio_sections.append(
                    f"### {emb_type.capitalize()} embeddings\n\n" + _portfolio_table(rows)
                )

        # Alpha results
        all_alpha_rows: list[dict] = []
        alpha_sections: list[str] = []
        for emb_type in emb_types:
            rows = _load_csv(port_dir / f"factor_alphas_{emb_type}.csv")
            if rows:
                all_alpha_rows.extend(rows)
                alpha_sections.append(
                    f"### {emb_type.capitalize()} embeddings — FF5+UMD alphas\n\n"
                    + _alpha_table(rows, "ff5_umd")
                )

        findings = _key_findings(agg_by_type, all_summary_rows, all_alpha_rows)

        report = self._render(
            classification_sections=classification_sections,
            portfolio_sections=portfolio_sections,
            alpha_sections=alpha_sections,
            findings=findings,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        log.info("report_written", path=str(out))
        return out

    @staticmethod
    def _render(
        classification_sections: list[str],
        portfolio_sections: list[str],
        alpha_sections: list[str],
        findings: str,
    ) -> str:
        sections: list[str] = [
            "# Classical Analysis Pipeline — Results Report",
            "",
            "> Generated by `pipeline/report.py` (Obaid & Pukthuanthong 2022 replication framework)",
            "",
            "---",
            "",
            "## 1. Classification Performance",
            "",
            "Walk-forward expanding-window cross-validation.  "
            "Values are mean ± std across all folds.  "
            "**Bold** = best in column.",
            "",
        ]
        if classification_sections:
            sections.extend(classification_sections)
        else:
            sections.append("*No classification results found.  Run `classify` first.*\n")

        sections += [
            "---",
            "",
            "## 2. Portfolio Performance",
            "",
            "Long-short quintile portfolios (long Q5, short Q1) rebalanced daily.",
            "",
        ]
        if portfolio_sections:
            sections.extend(portfolio_sections)
        else:
            sections.append("*No portfolio results found.  Run `build-portfolio` first.*\n")

        sections += [
            "---",
            "",
            "## 3. Factor Model Alphas",
            "",
            "OLS regression with HC3 robust standard errors on Fama-French 5-factor + momentum (UMD) model.  "
            "Significance: \\* p<0.10, \\*\\* p<0.05, \\*\\*\\* p<0.01.",
            "",
        ]
        if alpha_sections:
            sections.extend(alpha_sections)
        else:
            sections.append(
                "*No alpha results found.  Ensure Fama-French factors are available and run `build-portfolio`.*\n"
            )

        sections += [
            "---",
            "",
            "## 4. Key Findings",
            "",
            findings,
            "",
            "---",
            "",
            "## 5. Notes",
            "",
            "- All classification uses strictly expanding walk-forward CV; no future data is seen during training.",
            "- Embeddings: CLIP ViT-B/32 (image), ProsusAI/FinBERT or all-MiniLM-L6-v2 (text).",
            "- Classifiers: SVM (RBF), SVM (poly), XGBoost, Random Forest, MLP.",
            "- Factor data: Kenneth French data library (F-F_Research_Data_5_Factors_2x3_daily + UMD).",
            "- Quantum spanning test placeholder is in `pipeline/portfolio.py::spanning_test()`.",
            "",
        ]

        return "\n".join(sections)
