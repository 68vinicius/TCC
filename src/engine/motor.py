import time
import logging
import psutil
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Dict, Any

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================================================
# STRATEGY PATTERN
# =========================================================

class SimilarityStrategy(ABC):

    @abstractmethod
    def calculate(self, t1: str, t2: str) -> float:
        pass


# =========================================================
# TF-IDF + COSINE
# =========================================================

class CosineSimilarityStrategy(SimilarityStrategy):

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def calculate(self, t1: str, t2: str) -> float:
        matrix = self.vectorizer.transform([t1, t2])

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return float(score)


# =========================================================
# JACCARD NGRAM
# =========================================================

class JaccardSimilarityStrategy(SimilarityStrategy):

    def __init__(self, ngram_size: int = 3):
        self.ngram_size = ngram_size

    @lru_cache(maxsize=1024)
    def _get_char_ngrams(self, text: str):

        n = self.ngram_size

        return set(
            text[i:i+n]
            for i in range(len(text)-n+1)
        )

    def calculate(self, t1: str, t2: str) -> float:

        s1 = self._get_char_ngrams(t1)
        s2 = self._get_char_ngrams(t2)

        union = s1 | s2

        if not union:
            return 0.0

        return float(len(s1 & s2) / len(union))


# =========================================================
# SEMANTIC EMBEDDINGS
# =========================================================

class EmbeddingSimilarityStrategy(SimilarityStrategy):

    def __init__(self):
        self.model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )

    def calculate(self, t1: str, t2: str) -> float:

        embeddings = self.model.encode([t1, t2])

        score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return float(score)


# =========================================================
# ANALYZER
# =========================================================

class TextAnalyzer:

    def __init__(
        self,
        vectorizer,
        cleaner,
        corpus_reference
    ):

        self.cleaner = cleaner

        # Fit GLOBAL do espaço vetorial
        self.vectorizer = vectorizer
        self.vectorizer.fit(corpus_reference)

        self.cosine_strategy = CosineSimilarityStrategy(
            self.vectorizer
        )

        self.jaccard_strategy = JaccardSimilarityStrategy()

        self.embedding_strategy = EmbeddingSimilarityStrategy()

    # =====================================================
    # COMPOSITE SCORE
    # =====================================================

    def _final_score(
        self,
        cosine_score,
        jaccard_score,
        embedding_score
    ):

        return (
            cosine_score * 0.4 +
            jaccard_score * 0.2 +
            embedding_score * 0.4
        )

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    def _classify(self, score):

        if score >= 0.85:
            return "ALTO RISCO"

        elif score >= 0.65:
            return "SIMILARIDADE ELEVADA"

        elif score >= 0.40:
            return "SIMILARIDADE MODERADA"

        return "BAIXA SIMILARIDADE"

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def compare(
        self,
        t1_raw: str,
        t2_raw: str
    ) -> Dict[str, Any]:

        start = time.perf_counter()

        logger.info("Iniciando processamento textual")

        try:

            t1 = self.cleaner.clean(t1_raw)
            t2 = self.cleaner.clean(t2_raw)

            if not t1 or not t2:

                logger.warning("Texto vazio após limpeza")

                return {
                    "status": "error",
                    "message": "Texto vazio após limpeza"
                }

            # =============================================
            # METRICS
            # =============================================

            cosine_score = self.cosine_strategy.calculate(t1, t2)

            jaccard_score = self.jaccard_strategy.calculate(t1, t2)

            embedding_score = self.embedding_strategy.calculate(t1, t2)

            final_score = self._final_score(
                cosine_score,
                jaccard_score,
                embedding_score
            )

            classification = self._classify(final_score)

            elapsed = time.perf_counter() - start

            process = psutil.Process()

            memory_mb = process.memory_info().rss / 1024 / 1024

            logger.info(
                f"Processamento concluído em {elapsed:.4f}s"
            )

            return {

                "status": "success",

                "cosseno": cosine_score,

                "jaccard": jaccard_score,

                "embedding": embedding_score,

                "score_final": final_score,

                "classificacao": classification,

                "tempo": elapsed,

                "ram_mb": round(memory_mb, 2),

                "t1_limpo": t1,

                "t2_limpo": t2
            }

        except Exception as e:

            logger.exception("Falha crítica no motor")

            return {
                "status": "error",
                "message": str(e)
            }