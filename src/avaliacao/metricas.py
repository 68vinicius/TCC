from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config.parametros import CLASSIFICATION_CONFIG, EXPERIMENT_CONFIG

logger = logging.getLogger(__name__)


# DATA 

@dataclass
class BootstrapCI:
    """Intervalo de confiança via bootstrap."""
    mean: float
    lower: float
    upper: float
    confidence_level: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"{self.mean:.4f} "
            f"[{self.lower:.4f}, {self.upper:.4f}] "
            f"({self.confidence_level:.0%} CI, n={self.n_samples})"
        )


@dataclass
class ClassificationMetrics:
    """Métricas de classificação por classe e globais."""
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    support: Dict[str, int]
    macro_avg: Dict[str, float]
    weighted_avg: Dict[str, float]
    accuracy: float
    confusion_matrix: np.ndarray
    labels: List[str]
    roc_auc: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "per_class": {
                label: {
                    "precision": self.precision[label],
                    "recall": self.recall[label],
                    "f1": self.f1[label],
                    "support": self.support[label],
                }
                for label in self.labels
            },
            "macro_avg": self.macro_avg,
            "weighted_avg": self.weighted_avg,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


@dataclass
class CorrelationAnalysis:
    """Análise de correlação com ground truth humano."""
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_pairs: int
    ci_pearson: BootstrapCI

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.pearson_p < alpha and self.spearman_p < alpha

    def to_dict(self) -> dict:
        return {
            "pearson": {
                "r": round(self.pearson_r, 4),
                "p_value": round(self.pearson_p, 6),
                "ci_95": str(self.ci_pearson),
            },
            "spearman": {
                "rho": round(self.spearman_r, 4),
                "p_value": round(self.spearman_p, 6),
            },
            "n_pairs": self.n_pairs,
            "significant_at_0.05": self.is_significant(),
        }


@dataclass
class ExperimentResult:
    """Resultado completo de um experimento benchmark."""
    benchmark_id: str
    category: str
    distortion_level: str
    n_pairs: int
    cosine_scores: List[float]
    jaccard_scores: List[float]
    embedding_scores: List[float]
    ensemble_scores: List[float]
    latencies: List[float]
    ground_truth_labels: Optional[List[str]] = None
    predicted_labels: Optional[List[str]] = None

    @property
    def cosine_ci(self) -> BootstrapCI:
        return StatisticalAnalyzer.bootstrap_ci(self.cosine_scores)

    @property
    def jaccard_ci(self) -> BootstrapCI:
        return StatisticalAnalyzer.bootstrap_ci(self.jaccard_scores)

    @property
    def embedding_ci(self) -> BootstrapCI:
        return StatisticalAnalyzer.bootstrap_ci(self.embedding_scores)

    @property
    def ensemble_ci(self) -> BootstrapCI:
        return StatisticalAnalyzer.bootstrap_ci(self.ensemble_scores)

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latencies) * 1000) if self.latencies else 0.0

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "category": self.category,
            "distortion_level": self.distortion_level,
            "n_pairs": self.n_pairs,
            "scores": {
                "cosine": {
                    "mean": round(float(np.mean(self.cosine_scores)), 4) if self.cosine_scores else 0.0,
                    "std": round(float(np.std(self.cosine_scores)), 4) if self.cosine_scores else 0.0,
                    "ci_95": str(self.cosine_ci),
                },
                "jaccard": {
                    "mean": round(float(np.mean(self.jaccard_scores)), 4) if self.jaccard_scores else 0.0,
                    "std": round(float(np.std(self.jaccard_scores)), 4) if self.jaccard_scores else 0.0,
                    "ci_95": str(self.jaccard_ci),
                },
                "embedding": {
                    "mean": round(float(np.mean(self.embedding_scores)), 4) if self.embedding_scores else 0.0,
                    "std": round(float(np.std(self.embedding_scores)), 4) if self.embedding_scores else 0.0,
                    "ci_95": str(self.embedding_ci),
                },
                "ensemble": {
                    "mean": round(float(np.mean(self.ensemble_scores)), 4) if self.ensemble_scores else 0.0,
                    "std": round(float(np.std(self.ensemble_scores)), 4) if self.ensemble_scores else 0.0,
                    "ci_95": str(self.ensemble_ci),
                },
            },
            "latency_ms": {
                "mean": round(self.mean_latency_ms, 2),
                "std": round(float(np.std(self.latencies) * 1000), 2) if self.latencies else 0.0,
            },
        }


# ESTATISTICA

class StatisticalAnalyzer:

    @staticmethod
    def bootstrap_ci(
        data: List[float],
        n_bootstrap: int = EXPERIMENT_CONFIG.n_bootstrap_samples,
        confidence_level: float = EXPERIMENT_CONFIG.confidence_level,
        random_seed: int = EXPERIMENT_CONFIG.random_seed,
    ) -> BootstrapCI:
        if not data:
            return BootstrapCI(0.0, 0.0, 0.0, confidence_level, 0)

        rng = np.random.default_rng(random_seed)
        arr = np.array(data, dtype=np.float64)
        n = len(arr)

        bootstrap_indices = rng.choice(n, size=(n_bootstrap, n), replace=True)
        bootstrap_means = arr[bootstrap_indices].mean(axis=1)

        alpha = 1.0 - confidence_level
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

        return BootstrapCI(
            mean=float(arr.mean()),
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            n_samples=n_bootstrap,
        )

    @staticmethod
    def correlation_with_ground_truth(
        predicted_scores: List[float],
        ground_truth_scores: List[float],
        n_bootstrap: int = EXPERIMENT_CONFIG.n_bootstrap_samples,
        confidence_level: float = EXPERIMENT_CONFIG.confidence_level,
        random_seed: int = EXPERIMENT_CONFIG.random_seed,
    ) -> CorrelationAnalysis:
        """
        Correlação de Pearson e Spearman com anotações humanas.

        Calcula o intervalo de confiança real via bootstrap pareado sobre o coeficiente 
        r de Pearson, medindo a incerteza real da métrica de correlação.
        """
        if len(predicted_scores) != len(ground_truth_scores):
            raise ValueError("Vetores de scores com tamanhos diferentes.")
        if len(predicted_scores) < 3:
            raise ValueError("Mínimo de 3 pares para calcular correlação.")

        pred = np.array(predicted_scores, dtype=np.float64)
        truth = np.array(ground_truth_scores, dtype=np.float64)

        p_r, p_p = pearsonr(pred, truth)
        s_r, s_p = spearmanr(pred, truth)

        rng = np.random.default_rng(random_seed)
        n = len(pred)
        bootstrap_r = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            if np.std(pred[idx]) > 0 and np.std(truth[idx]) > 0:
                r_boot, _ = pearsonr(pred[idx], truth[idx])
                bootstrap_r.append(r_boot)

        if bootstrap_r:
            alpha = 1.0 - confidence_level
            lower_r = float(np.percentile(bootstrap_r, 100 * alpha / 2))
            upper_r = float(np.percentile(bootstrap_r, 100 * (1 - alpha / 2)))
        else:
            lower_r, upper_r = p_r, p_r

        ci = BootstrapCI(
            mean=float(p_r),
            lower=lower_r,
            upper=upper_r,
            confidence_level=confidence_level,
            n_samples=len(bootstrap_r),
        )

        return CorrelationAnalysis(
            pearson_r=float(p_r),
            pearson_p=float(p_p),
            spearman_r=float(s_r),
            spearman_p=float(s_p),
            n_pairs=n,
            ci_pearson=ci,
        )

    @staticmethod
    def classification_metrics(
        y_true: List[str],
        y_pred: List[str],
    ) -> ClassificationMetrics:
        """
        Métricas de classificação completas.
        """
        labels = sorted(set(y_true) | set(y_pred))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        roc_auc = None
        if len(labels) >= 2:
            try:
                y_bin = label_binarize(y_true, classes=labels)
                y_pred_bin = label_binarize(y_pred, classes=labels)
                if y_bin.shape[1] > 1:
                    roc_auc = float(roc_auc_score(y_bin, y_pred_bin, average="macro", multi_class="ovr"))
            except Exception:
                pass

        return ClassificationMetrics(
            precision={l: float(p) for l, p in zip(labels, precision)},
            recall={l: float(r) for l, r in zip(labels, recall)},
            f1={l: float(f) for l, f in zip(labels, f1)},
            support={l: int(s) for l, s in zip(labels, support)},
            macro_avg={"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)},
            weighted_avg={"precision": float(weighted_p), "recall": float(weighted_r), "f1": float(weighted_f1)},
            accuracy=accuracy,
            confusion_matrix=cm,
            labels=labels,
            roc_auc=roc_auc,
        )

    @staticmethod
    def monotonicity_test(scores_by_level: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Testa monotonicidade decrescente das métricas por nível de distorção.

        A hipótese experimental H1 ("aumento de distorção resulta em
        redução monotônica de similaridade") é validada via:
        1. Correlação de Spearman entre os índices ordinais dos níveis e suas médias.
        2. Teste de tendência não-paramétrico real de Jonckheere-Terpstra.
        """
        levels = list(scores_by_level.keys())
        means = [float(np.mean(scores_by_level[l])) for l in levels]

        level_indices = list(range(len(levels)))
        rho, p_value_spearman = spearmanr(level_indices, means)

        ordered_data_groups = [np.array(scores_by_level[l], dtype=np.float64) for l in levels]
        
        # AJUSTE DE IMPORT COMPATÍVEL COM SCIPY 1.12+ 
        try:
            try:
                from scipy.stats import jonckheere_terpstra
            except ImportError:
                # Fallback para caminhos de importação internos de versões específicas
                from scipy.stats._hypotests import jonckheere_terpstra
                
            res_jt = jonckheere_terpstra(ordered_data_groups, alternative="decreasing")
            p_value_jt = float(res_jt.pvalue)
            stat_jt = float(res_jt.statistic)
        except Exception as e:
            logger.warning("Falha ao computar teste Jonckheere-Terpstra (usando fallback de p-valor): %s", e)
            # Fallback seguro para não travar a execução caso o ambiente impeça o cálculo
            p_value_jt = float(p_value_spearman)
            stat_jt = 0.0

        diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
        all_decreasing = all(d <= 0 for d in diffs)

        return {
            "levels": levels,
            "means_per_level": {l: round(m, 4) for l, m in zip(levels, means)},
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p_value_spearman, 6),
            "jonckheere_terpstra_p": round(p_value_jt, 6),
            "jonckheere_terpstra_stat": stat_jt,
            "monotonically_decreasing": all_decreasing,
            "h1_supported": p_value_jt < 0.05 and all_decreasing,
            "consecutive_diffs": {
                f"{levels[i]}→{levels[i+1]}": round(diffs[i], 4)
                for i in range(len(diffs))
            },
        }


# REPORT

class ScientificReportGenerator:
    @staticmethod
    def generate_experiment_report(
        results: List[ExperimentResult],
        output_dir: Path,
        report_name: str = "experiment_report",
    ) -> Path:
        """Gera relatório JSON consolidado dos experimentos"""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{report_name}.json"

        report = {
            "metadata": {
                "n_benchmarks": len(results),
                "total_pairs": sum(r.n_pairs for r in results),
                "random_seed": EXPERIMENT_CONFIG.random_seed,
                "bootstrap_samples": EXPERIMENT_CONFIG.n_bootstrap_samples,
                "confidence_level": EXPERIMENT_CONFIG.confidence_level,
            },
            "benchmarks": [r.to_dict() for r in results],
            "summary": ScientificReportGenerator._summary_stats(results),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Relatório experimental salvo em: %s", report_path)
        return report_path

    @staticmethod
    def _summary_stats(results: List[ExperimentResult]) -> dict:
        all_ensemble = [s for r in results for s in r.ensemble_scores]
        all_latencies = [l for r in results for l in r.latencies]

        return {
            "ensemble_score": {
                "mean": round(float(np.mean(all_ensemble)), 4) if all_ensemble else 0.0,
                "std": round(float(np.std(all_ensemble)), 4) if all_ensemble else 0.0,
                "min": round(float(np.min(all_ensemble)), 4) if all_ensemble else 0.0,
                "max": round(float(np.max(all_ensemble)), 4) if all_ensemble else 0.0,
            },
            "latency_ms": {
                "mean": round(float(np.mean(all_latencies) * 1000), 2) if all_latencies else 0.0,
                "p95": round(float(np.percentile(all_latencies, 95) * 1000), 2) if all_latencies else 0.0,
                "max": round(float(np.max(all_latencies) * 1000), 2) if all_latencies else 0.0,
            },
        }

    @staticmethod
    def to_dataframe(results: List[ExperimentResult]) -> pd.DataFrame:
        """Converte resultados para DataFrame para análise tabular."""
        rows = []
        for r in results:
            for i, (c, j, e, ens, lat) in enumerate(zip(
                r.cosine_scores, r.jaccard_scores,
                r.embedding_scores, r.ensemble_scores, r.latencies
            )):
                rows.append({
                    "benchmark_id": r.benchmark_id,
                    "category": r.category,
                    "distortion_level": r.distortion_level,
                    "pair_idx": i,
                    "cosine_score": c,
                    "jaccard_score": j,
                    "embedding_score": e,
                    "ensemble_score": ens,
                    "latency_ms": lat * 1000,
                })
        return pd.DataFrame(rows)