import sys
from pathlib import Path

# =========================================================
# PATH SETUP
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# =========================================================
# IMPORTS
# =========================================================

import json
import statistics

from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from engine.extrator import DocumentFactory
from engine.processadores import TextCleaner
from engine.motor import TextAnalyzer

# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class BenchmarkResult:

    benchmark: str
    category: str

    cosseno: float
    jaccard: float
    semantic: float

    tempo: float
    memoria_mb: float

    status: str


# =========================================================
# BENCHMARK RUNNER
# =========================================================

class BenchmarkRunner:

    def __init__(self, analyzer: TextAnalyzer):

        self.analyzer = analyzer
        self.results: List[BenchmarkResult] = []

# =====================================================
# EXECUTA UM PAR
# =====================================================

    def run_pair(
        self,
        benchmark: str,
        category: str,
        file_a,
        file_b
    ) -> BenchmarkResult:

        raw_a = DocumentFactory.get_text(file_a)
        raw_b = DocumentFactory.get_text(file_b)

        res = self.analyzer.compare(raw_a, raw_b)

        result = BenchmarkResult(

            benchmark=benchmark,
            category=category,

            cosseno=res.get("cosseno", 0.0),
            jaccard=res.get("jaccard", 0.0),
            semantic=res.get("semantic", 0.0),

            tempo=res.get("tempo", 0.0),
            memoria_mb=res.get("memoria_mb", 0.0),

            status=res.get("status", "unknown")
        )

        self.results.append(result)

        return result

# =====================================================
# EXECUTA DATASET COMPLETO
# =====================================================

    def run_directory(
        self,
        benchmark_root: str
    ) -> List[BenchmarkResult]:

        benchmark_root = Path(benchmark_root)

        if not benchmark_root.exists():

            raise FileNotFoundError(
                f"Diretório não encontrado: {benchmark_root}"
            )

        # =================================================
        # ITERA BENCHMARKS
        # =================================================

        for benchmark_dir in sorted(benchmark_root.iterdir()):

            if not benchmark_dir.is_dir():
                continue

            benchmark_name = benchmark_dir.name

            print("\n================================================")
            print(f"EXECUTANDO {benchmark_name}")
            print("================================================")

            # =============================================
            # ITERA AMOSTRAS
            # =============================================

            for sample_dir in benchmark_dir.iterdir():

                if not sample_dir.is_dir():
                    continue

                file_a = sample_dir / "a.txt"
                file_b = sample_dir / "b.txt"

                if not file_a.exists() or not file_b.exists():

                    print(
                        f"[WARNING] Arquivos ausentes em {sample_dir}"
                    )

                    continue

                # =========================================
                # LABEL / METADATA
                # =========================================

                category = benchmark_name

                label_path = sample_dir / "label.json"

                if label_path.exists():

                    try:

                        with open(
                            label_path,
                            "r",
                            encoding="utf-8"
                        ) as f:

                            metadata = json.load(f)

                        category = metadata.get(
                            "category",
                            benchmark_name
                        )

                    except Exception as e:

                        print(
                            f"[WARNING] Falha ao ler label.json: {e}"
                        )

                # =========================================
                # EXECUÇÃO
                # =========================================

                print(
                    f"[RUNNING] {benchmark_name} -> {sample_dir.name}"
                )

                try:

                    with open(file_a, "rb") as fa:
                        with open(file_b, "rb") as fb:

                            result = self.run_pair(
                                benchmark=benchmark_name,
                                category=category,
                                file_a=fa,
                                file_b=fb
                            )

                    print(
                        f"[OK] "
                        f"Cosseno={result.cosseno:.4f} | "
                        f"Jaccard={result.jaccard:.4f} | "
                        f"Semantic={result.semantic:.4f} | "
                        f"Tempo={result.tempo:.4f}s"
                    )

                except Exception as e:

                    print(
                        f"[ERROR] {sample_dir.name}: {e}"
                    )

        return self.results

# =====================================================
# EXPORT JSON
# =====================================================

    def export_json(
        self,
        path="benchmark_results.json"
    ):

        data = [asdict(r) for r in self.results]

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

# =====================================================
# EXPORT CSV
# =====================================================

    def export_csv(
        self,
        path="benchmark_results.csv"
    ):

        df = pd.DataFrame(
            [asdict(r) for r in self.results]
        )

        df.to_csv(
            path,
            index=False,
            encoding="utf-8"
        )

# =====================================================
# DATAFRAME
# =====================================================

    def to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(
            [asdict(r) for r in self.results]
        )

    # =====================================================
    # ESTATÍSTICAS
    # =====================================================

    def statistics(self) -> Dict[str, Any]:

        if not self.results:
            return {}

        cossenos = [r.cosseno for r in self.results]
        jaccards = [r.jaccard for r in self.results]
        semantics = [r.semantic for r in self.results]
        tempos = [r.tempo for r in self.results]

        return {

            "samples": len(self.results),

            "cosseno_mean":
                statistics.mean(cossenos),

            "cosseno_std":
                statistics.stdev(cossenos)
                if len(cossenos) > 1 else 0,

            "jaccard_mean":
                statistics.mean(jaccards),

            "jaccard_std":
                statistics.stdev(jaccards)
                if len(jaccards) > 1 else 0,

            "semantic_mean":
                statistics.mean(semantics),

            "semantic_std":
                statistics.stdev(semantics)
                if len(semantics) > 1 else 0,

            "tempo_medio":
                statistics.mean(tempos)
        }

    # =====================================================
    # LIMPA RESULTADOS
    # =====================================================

    def clear(self):

        self.results.clear()


# =========================================================
# FUNÇÃO AUXILIAR
# =========================================================

def build_reference_corpus(dataset_root="dataset"):

    corpus = []

    dataset_root = Path(dataset_root)

    for benchmark_dir in dataset_root.iterdir():

        if not benchmark_dir.is_dir():
            continue

        for sample_dir in benchmark_dir.iterdir():

            if not sample_dir.is_dir():
                continue

            for file_name in ["a.txt", "b.txt"]:

                file_path = sample_dir / file_name

                if file_path.exists():

                    try:

                        text = file_path.read_text(
                            encoding="utf-8"
                        )

                        if text.strip():

                            corpus.append(text)

                    except Exception as e:

                        print(
                            f"[WARNING] Falha ao ler {file_path}: {e}"
                        )

    return corpus


# =========================================================
# EXECUÇÃO DIRETA
# =========================================================

if __name__ == "__main__":

    print("\n================================================")
    print("INICIALIZANDO ENGINE")
    print("================================================\n")

# =====================================================
# CORPUS REFERENCIAL
# =====================================================

    corpus_reference = build_reference_corpus("dataset")

    print(
        f"[INFO] Corpus carregado: "
        f"{len(corpus_reference)} documentos"
    )

    if len(corpus_reference) == 0:

        raise ValueError(
            "Corpus vazio. Verifique o dataset."
        )

# =====================================================
# VECTORIZER
# =====================================================

    vectorizer = TfidfVectorizer(

        stop_words=None,

        ngram_range=(1, 2),

        sublinear_tf=True,

        min_df=1
    )

# =====================================================
# CLEANER
# =====================================================

    cleaner = TextCleaner()

# =====================================================
# ANALYZER
# =====================================================

    analyzer = TextAnalyzer(

        vectorizer=vectorizer,

        cleaner=cleaner,

        corpus_reference=corpus_reference
    )

# =====================================================
# RUNNER
# =====================================================

    runner = BenchmarkRunner(analyzer)

# =====================================================
# EXECUÇÃO
# =====================================================

    print("\n================================================")
    print("INICIANDO BENCHMARKS")
    print("================================================\n")

    runner.run_directory(
        benchmark_root="dataset"
    )

# =====================================================
# DATAFRAME
# =====================================================

    print("\n================================================")
    print("RESULTADOS")
    print("================================================\n")

    df = runner.to_dataframe()

    print(df)

# =====================================================
# ESTATÍSTICAS
# =====================================================

    print("\n================================================")
    print("ESTATÍSTICAS")
    print("================================================\n")

    stats = runner.statistics()

    for k, v in stats.items():

        print(f"{k}: {v}")

# =====================================================
# EXPORTS
# =====================================================

    EXPORT_DIR = Path("src/experimentos")

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    runner.export_json(
        EXPORT_DIR / "resultados_benchmark.json"
    )

    runner.export_csv(
        EXPORT_DIR / "resultados_benchmark.csv"
    )

    print("\n================================================")
    print("EXPORT FINALIZADO")
    print(f"Arquivos salvos em: {EXPORT_DIR}")
    print("================================================\n")
