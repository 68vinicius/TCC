from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config.parametros import NLP_CONFIG, ENSEMBLE_CONFIG, CLASSIFICATION_CONFIG
from src.processamento.saneamento import PipelineFactory, TextPipeline
from src.processamento.fragmentador import TextChunker

logger = logging.getLogger(__name__)

@dataclass
class SimilarityResult:

    # Scores individuais
    cosine_score: float
    jaccard_char_score: float
    jaccard_word_score: float
    embedding_score: float

    # Score agregado
    ensemble_score: float

    # Classificação
    classification: str

    # Textos processados (para auditoria)
    text_a_vectorized: str  
    text_b_vectorized: str
    text_a_embedded: str    
    text_b_embedded: str

    # Telemetria
    latency_seconds: float
    ram_usage_mb: float
    cpu_percent: float

    # Metadados do processamento
    text_a_chunks: int = 1
    text_b_chunks: int = 1
    embedding_model: str = ""
    tfidf_vocabulary_size: int = 0

    status: str = "success"
    warnings: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "scores": {
                "cosine_tfidf": round(self.cosine_score, 6),
                "jaccard_char_ngram": round(self.jaccard_char_score, 6),
                "jaccard_word_ngram": round(self.jaccard_word_score, 6),
                "semantic_embedding": round(self.embedding_score, 6),
                "ensemble_hybrid": round(self.ensemble_score, 6),
            },
            "classification": self.classification,
            "ensemble_weights": {
                "cosine": ENSEMBLE_CONFIG.cosine_weight,
                "jaccard": ENSEMBLE_CONFIG.jaccard_weight,
                "embedding": ENSEMBLE_CONFIG.embedding_weight,
            },
            "processing": {
                "latency_seconds": round(self.latency_seconds, 4),
                "ram_usage_mb": round(self.ram_usage_mb, 2),
                "cpu_percent": round(self.cpu_percent, 1),
                "text_a_chunks": self.text_a_chunks,
                "text_b_chunks": self.text_b_chunks,
                "tfidf_vocabulary_size": self.tfidf_vocabulary_size,
                "embedding_model": self.embedding_model,
            },
            "text_previews": {
                "text_a_vectorized_preview": self.text_a_vectorized[:500],
                "text_b_vectorized_preview": self.text_b_vectorized[:500],
            },
            "warnings": self.warnings,
        }


@dataclass
class ErrorResult:
    status: str = "error"
    message: str = ""
    stage: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "message": self.message, "stage": self.stage}


class SimilarityStrategy(ABC):

    @abstractmethod
    def compute(self, text_a: str, text_b: str) -> float:
        """
        Computa similaridade entre dois textos

        Returns:
            float em [0.0, 1.0] onde 1.0 = idênticos, 0.0 = sem sobreposição.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome descritivo da estratégia."""
        ...


# TF-IDF + COSSENO

class CosineTFIDFStrategy(SimilarityStrategy):

    def __init__(self):
        self._config = NLP_CONFIG

    @property
    def name(self) -> str:
        return "TF-IDF Cosine Similarity"

    def compute(self, text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0

        vectorizer = TfidfVectorizer(
            ngram_range=self._config.tfidf_ngram_range,
            max_features=self._config.tfidf_max_features,
            sublinear_tf=self._config.tfidf_sublinear_tf,
            min_df=1,
            max_df=1.0,
            analyzer="word",
        )

        try:
            matrix = vectorizer.fit_transform([text_a, text_b])
            score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            vocab_size = len(vectorizer.vocabulary_)
            logger.debug("TF-IDF vocab size: %d, cosine score: %.4f", vocab_size, score)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("CosineTFIDF falhou: %s", e)
            return 0.0

    def compute_with_vocab(self, text_a: str, text_b: str) -> Tuple[float, int]:
        """Retorna score e tamanho do vocabulário."""
        if not text_a.strip() or not text_b.strip():
            return 0.0, 0

        vectorizer = TfidfVectorizer(
            ngram_range=self._config.tfidf_ngram_range,
            max_features=self._config.tfidf_max_features,
            sublinear_tf=self._config.tfidf_sublinear_tf,
            min_df=1,
            max_df=1.0,
        )
        matrix = vectorizer.fit_transform([text_a, text_b])
        score = float(np.clip(cosine_similarity(matrix[0:1], matrix[1:2])[0][0], 0.0, 1.0))
        return score, len(vectorizer.vocabulary_)


# JACCARD COM N-GRAMS

@lru_cache(maxsize=512)
def _char_ngrams(text: str, n: int) -> frozenset:

    return frozenset(text[i:i+n] for i in range(len(text) - n + 1))


@lru_cache(maxsize=512)
def _word_ngrams(text: str, n: int) -> frozenset:
    words = text.split()
    return frozenset(
        " ".join(words[i:i+n]) for i in range(len(words) - n + 1)
    )


class JaccardStrategy(SimilarityStrategy):

    def __init__(
        self,
        char_ngram_size: int = NLP_CONFIG.jaccard_ngram_size,
        word_ngram_size: int = NLP_CONFIG.jaccard_word_ngram_size,
    ):
        self.char_n = char_ngram_size
        self.word_n = word_ngram_size

    @property
    def name(self) -> str:
        return f"Jaccard (char-{self.char_n}gram + word-{self.word_n}gram)"

    def _jaccard(self, set_a: frozenset, set_b: frozenset) -> float:
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    def compute_char(self, text_a: str, text_b: str) -> float:
        s_a = _char_ngrams(text_a, self.char_n)
        s_b = _char_ngrams(text_b, self.char_n)
        return self._jaccard(s_a, s_b)

    def compute_word(self, text_a: str, text_b: str) -> float:
        s_a = _word_ngrams(text_a, self.word_n)
        s_b = _word_ngrams(text_b, self.word_n)
        return self._jaccard(s_a, s_b)

    def compute(self, text_a: str, text_b: str) -> float:
        char_score = self.compute_char(text_a, text_b)
        word_score = self.compute_word(text_a, text_b)
        return float(0.6 * char_score + 0.4 * word_score)


# ─────────────────────────────────────────────────────────────
# STRATEGY: SENTENCE EMBEDDINGS
# ─────────────────────────────────────────────────────────────

class EmbeddingStrategy(SimilarityStrategy):

    def __init__(
        self,
        model_name: str = NLP_CONFIG.embedding_model_primary,
        chunker: Optional[TextChunker] = None,
    ):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._chunker = chunker or TextChunker()

    @property
    def name(self) -> str:
        return f"Semantic Embedding ({self.model_name})"

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Carregando modelo de embeddings: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compute(self, text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0

        model = self._get_model()
        chunks_a = self._chunker.chunk(text_a)
        chunks_b = self._chunker.chunk(text_b)

        if len(chunks_a) == 1 and len(chunks_b) == 1:
            return self._compare_single(model, text_a, text_b)

        return self._compare_chunked(model, chunks_a, chunks_b)

    def _compare_single(
        self,
        model: SentenceTransformer,
        text_a: str,
        text_b: str,
    ) -> float:
        embeddings = model.encode(
            [text_a, text_b],
            batch_size=NLP_CONFIG.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        # Com normalize_embeddings=True, produto escalar = cosseno
        score = float(np.dot(embeddings[0], embeddings[1]))
        return float(np.clip(score, 0.0, 1.0))

    def _compare_chunked(
        self,
        model: SentenceTransformer,
        chunks_a: list,
        chunks_b: list,
    ) -> float:
        texts_a = [c.text for c in chunks_a]
        texts_b = [c.text for c in chunks_b]

        all_texts = texts_a + texts_b
        all_embeddings = model.encode(
            all_texts,
            batch_size=NLP_CONFIG.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        emb_a = all_embeddings[:len(texts_a)]
        emb_b = all_embeddings[len(texts_a):]

        sim_matrix = cosine_similarity(emb_a, emb_b)
        max_scores_a = sim_matrix.max(axis=1)

        return float(np.clip(np.mean(max_scores_a), 0.0, 1.0))

    def get_chunk_counts(self, text_a: str, text_b: str) -> Tuple[int, int]:
        return len(self._chunker.chunk(text_a)), len(self._chunker.chunk(text_b))


# ENSEMBLE SCORER

class HybridEnsembleScorer:
    """ Os pesos são validados em tempo de importação  (config/settings.py) """

    def __init__(self, config=ENSEMBLE_CONFIG):
        self._config = config

    def combine(
        self,
        cosine: float,
        jaccard: float,
        embedding: float,
    ) -> float:
        score = (
            self._config.cosine_weight * cosine +
            self._config.jaccard_weight * jaccard +
            self._config.embedding_weight * embedding
        )
        return float(np.clip(score, 0.0, 1.0))


# CLASSIFIER

class DocumentClassifier:
    """ Os limiares são definidos em config/settings.py """

    def __init__(self, config=CLASSIFICATION_CONFIG):
        self._config = config

    def classify(self, score: float) -> str:
        if score >= self._config.high_risk_threshold:
            return self._config.LABEL_HIGH_RISK
        elif score >= self._config.high_similarity_threshold:
            return self._config.LABEL_HIGH
        elif score >= self._config.moderate_similarity_threshold:
            return self._config.LABEL_MODERATE
        return self._config.LABEL_LOW


# ANALYZER

class TextSimilarityAnalyzer:
    """

    Orquestrador principal do pipeline de análise de similaridade.

    1. Pipelines de pré-processamento duais (vetorização / embedding)
    2. Estratégias de similaridade via Strategy Pattern
    3. Ensemble híbrido ponderado
    4. Classificação baseada em limiares calibrados
    5. Telemetria e auditabilidade

    """

    def __init__(
        self,
        embedding_model: str = NLP_CONFIG.embedding_model_primary,
    ):
        # Pipelines duais — correção arquitetural central
        self._pipeline_vectorization = PipelineFactory.create_vectorization_pipeline()
        self._pipeline_embedding = PipelineFactory.create_embedding_pipeline()
        self._pipeline_jaccard = PipelineFactory.create_jaccard_pipeline()

        # Estratégias
        self._cosine = CosineTFIDFStrategy()
        self._jaccard = JaccardStrategy()
        self._embedding = EmbeddingStrategy(model_name=embedding_model)

        # Ensemble e classificador
        self._ensemble = HybridEnsembleScorer()
        self._classifier = DocumentClassifier()

        logger.info(
            "TextSimilarityAnalyzer inicializado com modelo '%s'",
            embedding_model
        )

    def analyze(
        self,
        text_a_raw: str,
        text_b_raw: str,
        warnings: Optional[List[str]] = None,
    ) -> SimilarityResult | ErrorResult:
        """
        Executa pipeline completo de análise de similaridade.

        Args:
            text_a_raw: Texto bruto do documento de referência
            text_b_raw: Texto bruto do documento de comparação
            warnings: Lista de avisos para acumular (modificada in-place)

        Returns:
            SimilarityResult em caso de sucesso, ErrorResult em falha.
        """
        if warnings is None:
            warnings = []

        t_start = time.perf_counter()
        process = psutil.Process()

        try:
            # Pré-processamento dual 
            text_a_vec, stats_a_vec = self._pipeline_vectorization.process(text_a_raw)
            text_b_vec, stats_b_vec = self._pipeline_vectorization.process(text_b_raw)

            text_a_emb, _ = self._pipeline_embedding.process(text_a_raw)
            text_b_emb, _ = self._pipeline_embedding.process(text_b_raw)

            text_a_jac, _ = self._pipeline_jaccard.process(text_a_raw)
            text_b_jac, _ = self._pipeline_jaccard.process(text_b_raw)

            if not text_a_vec.strip() or not text_b_vec.strip():
                return ErrorResult(
                    status="error",
                    message="Texto vazio após pré-processamento. Verifique se os documentos contêm texto extraível.",
                    stage="preprocessing",
                )

            for label, stats in [("Documento A", stats_a_vec), ("Documento B", stats_b_vec)]:
                if stats.compression_ratio > 0.85:
                    warnings.append(
                        f"{label}: alta taxa de compressão ({stats.compression_ratio:.0%}). "
                        "Texto pode ser muito curto ou conter principalmente stopwords/números."
                    )

            # Similaridades individuais 
            cosine_score, vocab_size = self._cosine.compute_with_vocab(text_a_vec, text_b_vec)
            jaccard_char = self._jaccard.compute_char(text_a_jac, text_b_jac)
            jaccard_word = self._jaccard.compute_word(text_a_jac, text_b_jac)
            jaccard_combined = self._jaccard.compute(text_a_jac, text_b_jac)
            embedding_score = self._embedding.compute(text_a_emb, text_b_emb)
            chunks_a, chunks_b = self._embedding.get_chunk_counts(text_a_emb, text_b_emb)

            # Ensemble 
            ensemble_score = self._ensemble.combine(cosine_score, jaccard_combined, embedding_score)
            classification = self._classifier.classify(ensemble_score)

            # Telemetria 
            latency = time.perf_counter() - t_start
            ram_mb = process.memory_info().rss / 1024 / 1024
            cpu_pct = process.cpu_percent(interval=None)

            logger.info(
                "Análise concluída em %.4fs | cosine=%.4f jaccard=%.4f emb=%.4f ensemble=%.4f [%s]",
                latency, cosine_score, jaccard_combined, embedding_score, ensemble_score, classification,
            )

            return SimilarityResult(
                cosine_score=cosine_score,
                jaccard_char_score=jaccard_char,
                jaccard_word_score=jaccard_word,
                embedding_score=embedding_score,
                ensemble_score=ensemble_score,
                classification=classification,
                text_a_vectorized=text_a_vec,
                text_b_vectorized=text_b_vec,
                text_a_embedded=text_a_emb,
                text_b_embedded=text_b_emb,
                latency_seconds=latency,
                ram_usage_mb=ram_mb,
                cpu_percent=cpu_pct,
                text_a_chunks=chunks_a,
                text_b_chunks=chunks_b,
                embedding_model=self._embedding.model_name,
                tfidf_vocabulary_size=vocab_size,
                warnings=warnings,
            )

        except Exception as e:
            logger.exception("Falha crítica no pipeline de análise")
            return ErrorResult(
                status="error",
                message=str(e),
                stage="pipeline",
            )