from __future__ import annotations

import io
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config.parametros import CLASSIFICATION_CONFIG, ENSEMBLE_CONFIG, NLP_CONFIG
from src.processamento.saneamento import (
    LowercaseStage,
    RemoveCitationStage,
    RemoveURLStage,
    StemmingStage,
    StopwordRemovalStage,
    TextPipeline,
    WhitespaceNormalizationStage,
    PipelineFactory,
)
from src.processamento.fragmentador import TextChunker
from src.motor.similaridade import (
    CosineTFIDFStrategy,
    DocumentClassifier,
    HybridEnsembleScorer,
    JaccardStrategy,
    TextSimilarityAnalyzer,
    SimilarityResult,
    ErrorResult,
)
from src.avaliacao.metricas import StatisticalAnalyzer, BootstrapCI


# FIXTURES

SAMPLE_TEXT_NLP = (
    "O processamento de linguagem natural é uma subárea da inteligência artificial "
    "que desenvolve sistemas capazes de compreender e gerar linguagem humana. "
    "Técnicas como TF-IDF e embeddings semânticos permitem comparar documentos "
    "com alta precisão em espaços vetoriais de alta dimensionalidade."
)

SAMPLE_TEXT_IR = (
    "A recuperação de informação busca documentos relevantes em grandes coleções. "
    "O modelo vetorial de Salton representa consultas e documentos como vetores. "
    "A similaridade de cosseno mede a proximidade angular entre vetores TF-IDF."
)

SAMPLE_TEXT_UNRELATED = (
    "O cultivo de café brasileiro inicia com a seleção de mudas de qualidade. "
    "O processo de beneficiamento envolve etapas de colheita, lavagem e secagem. "
    "A classificação por tamanho de grão influencia o preço de mercado."
)

SAMPLE_TEXT_IDENTICAL_COPY = SAMPLE_TEXT_NLP


# UNIT TESTS: PIPELINE STAGES

class TestPipelineStages:
    """Testes unitários para cada estágio do pipeline."""

    def test_lowercase_stage(self):
        stage = LowercaseStage()
        assert stage.process("TEXTO MAIÚSCULO") == "texto maiúsculo"
        assert stage.process("misto MIxTuRa") == "misto mixtura"

    def test_lowercase_preserves_unicode(self):
        stage = LowercaseStage()
        result = stage.process("AÇÃO ECONÔMICA")
        assert result == "ação econômica"

    def test_remove_url_http(self):
        stage = RemoveURLStage()
        result = stage.process("visitar https://example.com para mais informações")
        assert "https" not in result
        assert "example.com" not in result
        assert "informações" in result

    def test_remove_url_www(self):
        stage = RemoveURLStage()
        result = stage.process("site www.universidade.edu.br disponível")
        assert "www" not in result

    def test_remove_citation_abnt(self):
        stage = RemoveCitationStage()
        result = stage.process("conforme autores (SILVA, 2023) demonstraram")
        assert "(SILVA, 2023)" not in result
        assert "conforme" in result
        assert "demonstraram" in result

    def test_remove_citation_numeric(self):
        stage = RemoveCitationStage()
        result = stage.process("como proposto [1, 2, 3] na literatura")
        assert "[1, 2, 3]" not in result

    def test_stopword_removal_portuguese(self):
        stage = StopwordRemovalStage()
        result = stage.process("o sistema de processamento de texto é eficiente")
        # Stopwords "o", "de", "é" devem ser removidas
        assert " o " not in f" {result} "
        assert " de " not in f" {result} "
        # Termos conteúdo preservados
        assert "sistema" in result
        assert "processamento" in result

    def test_whitespace_normalization(self):
        stage = WhitespaceNormalizationStage()
        result = stage.process("  texto   com   espaços   extras  ")
        assert result == "texto com espaços extras"

    def test_whitespace_handles_newlines(self):
        stage = WhitespaceNormalizationStage()
        result = stage.process("linha1\n\nLinha2\t\tLinha3")
        assert "\n" not in result
        assert "\t" not in result


# UNIT TESTS: PIPELINE FACTORY

class TestPipelineFactory:
    """Testa a criação e diferenciação dos pipelines duais."""

    def test_vectorization_pipeline_applies_stemming(self):
        """Pipeline de vetorização deve aplicar stemming."""
        pipeline = PipelineFactory.create_vectorization_pipeline()
        result = pipeline.process_text("processamentos linguísticos avançados")
        # RSLP deve radicalizar as palavras
        # Verifica que houve redução (stemming comprime palavras)
        assert len(result) <= len("processamentos linguísticos avançados")

    def test_embedding_pipeline_preserves_natural_language(self):
        """Pipeline de embedding NÃO deve aplicar stemming."""
        from src.processamento.saneamento import StemmingStage
        pipeline = PipelineFactory.create_embedding_pipeline()
        # Verifica que StemmingStage não está na pipeline de embedding
        has_stemming = any(
            isinstance(stage, StemmingStage)
            for stage in pipeline._stages
        )
        assert not has_stemming, "Pipeline de embedding não deve conter StemmingStage"

    def test_vectorization_pipeline_not_equal_embedding_pipeline(self):
        """Os dois pipelines devem ter stages distintos."""
        vec = PipelineFactory.create_vectorization_pipeline()
        emb = PipelineFactory.create_embedding_pipeline()
        # Número de stages deve ser diferente (vec tem mais stages)
        assert len(vec._stages) > len(emb._stages)

    def test_pipeline_process_empty_string(self):
        pipeline = PipelineFactory.create_vectorization_pipeline()
        result = pipeline.process_text("")
        assert result == ""

    def test_pipeline_stats_compression(self):
        pipeline = PipelineFactory.create_vectorization_pipeline()
        _, stats = pipeline.process("o a e de que é um uma na no para com")
        # Stopwords devem ser removidas, causando compressão
        assert stats.compression_ratio > 0.0


# UNIT TESTS: TEXT CHUNKER

class TestTextChunker:
    """Testa a lógica de chunking para documentos longos."""

    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.chunk("texto curto")
        assert len(chunks) == 1
        assert chunks[0].text == "texto curto"

    def test_long_text_multiple_chunks(self):
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = " ".join(f"palavra{i}" for i in range(50))
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_overlap_creates_shared_content(self):
        chunker = TextChunker(chunk_size=5, chunk_overlap=2)
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_aggregate_mean(self):
        scores = [0.8, 0.6, 0.4]
        result = TextChunker.aggregate_similarity_scores(scores, method="mean")
        assert abs(result - 0.6) < 1e-9

    def test_aggregate_max(self):
        scores = [0.8, 0.6, 0.4]
        result = TextChunker.aggregate_similarity_scores(scores, method="max")
        assert abs(result - 0.8) < 1e-9

    def test_aggregate_empty(self):
        result = TextChunker.aggregate_similarity_scores([], method="mean")
        assert result == 0.0


# UNIT TESTS: SIMILARITY STRATEGIES

class TestCosineTFIDFStrategy:
    """
    Testa a correção crítica do fit do TF-IDF.
    O vectorizer DEVE ser fitado sobre os documentos do par, não um corpus externo.
    """

    def setup_method(self):
        self.strategy = CosineTFIDFStrategy()

    def test_identical_texts_score_one(self):
        score = self.strategy.compute(SAMPLE_TEXT_NLP, SAMPLE_TEXT_NLP)
        assert abs(score - 1.0) < 1e-6, f"Textos idênticos devem ter score=1.0, obtido: {score}"

    def test_unrelated_texts_low_score(self):
        score = self.strategy.compute(SAMPLE_TEXT_NLP, SAMPLE_TEXT_UNRELATED)
        assert score < 0.4, f"Textos não relacionados devem ter score baixo, obtido: {score}"

    def test_similar_texts_intermediate_score(self):
        score = self.strategy.compute(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        # Ambos são sobre NLP/IR — alguma sobreposição esperada
        assert 0.05 < score < 0.8, f"Textos relacionados devem ter score intermediário, obtido: {score}"

    def test_score_bounded_zero_one(self):
        score = self.strategy.compute(SAMPLE_TEXT_NLP, SAMPLE_TEXT_UNRELATED)
        assert 0.0 <= score <= 1.0

    def test_empty_text_returns_zero(self):
        assert self.strategy.compute("", SAMPLE_TEXT_NLP) == 0.0
        assert self.strategy.compute(SAMPLE_TEXT_NLP, "") == 0.0

    def test_symmetry(self):
        """Similaridade de cosseno é simétrica."""
        score_ab = self.strategy.compute(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        score_ba = self.strategy.compute(SAMPLE_TEXT_IR, SAMPLE_TEXT_NLP)
        assert abs(score_ab - score_ba) < 1e-6

    def test_vocabulary_size_returned(self):
        score, vocab_size = self.strategy.compute_with_vocab(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        assert vocab_size > 0
        assert isinstance(vocab_size, int)


class TestJaccardStrategy:
    """Testa as implementações de Jaccard char e word ngrams."""

    def setup_method(self):
        self.strategy = JaccardStrategy()

    def test_identical_texts_score_one(self):
        score = self.strategy.compute(SAMPLE_TEXT_NLP, SAMPLE_TEXT_NLP)
        assert abs(score - 1.0) < 1e-6

    def test_completely_different_texts(self):
        score = self.strategy.compute("abcdef ghijkl", "mnopqr stuvwx")
        assert score < 0.1

    def test_char_ngram_score_bounded(self):
        score = self.strategy.compute_char(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        assert 0.0 <= score <= 1.0

    def test_word_ngram_score_bounded(self):
        score = self.strategy.compute_word(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        assert 0.0 <= score <= 1.0

    def test_symmetry(self):
        s_ab = self.strategy.compute(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        s_ba = self.strategy.compute(SAMPLE_TEXT_IR, SAMPLE_TEXT_NLP)
        assert abs(s_ab - s_ba) < 1e-6


# UNIT TESTS: ENSEMBLE E CLASSIFIER

class TestHybridEnsembleScorer:
    """Testa a combinação ponderada das métricas."""

    def setup_method(self):
        self.scorer = HybridEnsembleScorer()

    def test_all_ones_gives_one(self):
        score = self.scorer.combine(1.0, 1.0, 1.0)
        assert abs(score - 1.0) < 1e-9

    def test_all_zeros_gives_zero(self):
        score = self.scorer.combine(0.0, 0.0, 0.0)
        assert abs(score - 0.0) < 1e-9

    def test_weighted_combination_correct(self):
        # cosine=0.35, jaccard=0.15, embedding=0.50
        score = self.scorer.combine(1.0, 0.0, 0.0)
        assert abs(score - ENSEMBLE_CONFIG.cosine_weight) < 1e-9

    def test_score_bounded(self):
        score = self.scorer.combine(0.5, 0.5, 0.5)
        assert 0.0 <= score <= 1.0

    def test_config_weights_sum_to_one(self):
        total = (
            ENSEMBLE_CONFIG.cosine_weight +
            ENSEMBLE_CONFIG.jaccard_weight +
            ENSEMBLE_CONFIG.embedding_weight
        )
        assert abs(total - 1.0) < 1e-9


class TestDocumentClassifier:
    """Testa os limiares de classificação."""

    def setup_method(self):
        self.classifier = DocumentClassifier()
        self.cfg = CLASSIFICATION_CONFIG

    def test_high_risk_threshold(self):
        assert self.classifier.classify(self.cfg.high_risk_threshold) == self.cfg.LABEL_HIGH_RISK
        assert self.classifier.classify(1.0) == self.cfg.LABEL_HIGH_RISK

    def test_high_similarity_threshold(self):
        score = self.cfg.high_similarity_threshold
        assert self.classifier.classify(score) == self.cfg.LABEL_HIGH

    def test_moderate_threshold(self):
        score = self.cfg.moderate_similarity_threshold
        assert self.classifier.classify(score) == self.cfg.LABEL_MODERATE

    def test_low_similarity(self):
        assert self.classifier.classify(0.0) == self.cfg.LABEL_LOW
        assert self.classifier.classify(0.10) == self.cfg.LABEL_LOW

    def test_boundary_conditions(self):
        """Testa condições de fronteira exatas."""
        # Exatamente no limiar de alto risco
        assert self.classifier.classify(0.80) == self.cfg.LABEL_HIGH_RISK
        # Imediatamente abaixo
        assert self.classifier.classify(0.799) == self.cfg.LABEL_HIGH


# UNIT TESTS: ANÁLISE ESTATÍSTICA

class TestStatisticalAnalyzer:
    """Testa os métodos de análise estatística."""

    def test_bootstrap_ci_returns_correct_structure(self):
        data = [0.8, 0.7, 0.9, 0.75, 0.85]
        ci = StatisticalAnalyzer.bootstrap_ci(data, n_bootstrap=100)
        assert isinstance(ci, BootstrapCI)
        assert ci.lower <= ci.mean <= ci.upper
        assert ci.n_samples == 100

    def test_bootstrap_ci_empty_data(self):
        ci = StatisticalAnalyzer.bootstrap_ci([])
        assert ci.mean == 0.0

    def test_bootstrap_ci_single_value(self):
        ci = StatisticalAnalyzer.bootstrap_ci([0.5], n_bootstrap=100)
        assert abs(ci.mean - 0.5) < 1e-9

    def test_correlation_perfect_positive(self):
        scores = [0.2, 0.4, 0.6, 0.8, 1.0]
        truth = [0.2, 0.4, 0.6, 0.8, 1.0]
        corr = StatisticalAnalyzer.correlation_with_ground_truth(scores, truth)
        assert abs(corr.pearson_r - 1.0) < 1e-6

    def test_correlation_perfect_negative(self):
        scores = [1.0, 0.8, 0.6, 0.4, 0.2]
        truth = [0.2, 0.4, 0.6, 0.8, 1.0]
        corr = StatisticalAnalyzer.correlation_with_ground_truth(scores, truth)
        assert abs(corr.pearson_r - (-1.0)) < 1e-6

    def test_classification_metrics_perfect(self):
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "A", "B"]
        metrics = StatisticalAnalyzer.classification_metrics(y_true, y_pred)
        assert metrics.accuracy == 1.0
        assert all(v == 1.0 for v in metrics.f1.values())

    def test_classification_metrics_all_wrong(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = ["B", "B", "A", "A"]
        metrics = StatisticalAnalyzer.classification_metrics(y_true, y_pred)
        assert metrics.accuracy == 0.0

    def test_monotonicity_decreasing(self):
        scores = {
            "identical": [0.95, 0.92, 0.98],
            "light": [0.75, 0.72, 0.78],
            "medium": [0.55, 0.52, 0.58],
            "heavy": [0.35, 0.32, 0.38],
            "unrelated": [0.10, 0.08, 0.12],
        }
        result = StatisticalAnalyzer.monotonicity_test(scores)
        assert result["monotonically_decreasing"] is True
        assert result["spearman_rho"] < -0.5


# INTEGRATION TESTS: ANALYZER

class TestTextSimilarityAnalyzerIntegration:
    """
    Testes de integração do motor completo.

    Estes testes executam o pipeline end-to-end com dados reais.
    São mais lentos que unitários mas validam a integração dos componentes.
    """

    @pytest.fixture(scope="class")
    def analyzer(self):
        """Fixture de escopo de classe para reutilizar o modelo carregado."""
        # Usa MiniLM para testes (mais rápido)
        return TextSimilarityAnalyzer(
            embedding_model=NLP_CONFIG.embedding_model_baseline
        )

    def test_identical_texts_high_ensemble_score(self, analyzer):
        result = analyzer.analyze(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IDENTICAL_COPY)
        assert isinstance(result, SimilarityResult)
        assert result.ensemble_score > 0.75, (
            f"Textos idênticos devem ter ensemble_score alto, obtido: {result.ensemble_score}"
        )

    def test_unrelated_texts_low_ensemble_score(self, analyzer):
        result = analyzer.analyze(SAMPLE_TEXT_NLP, SAMPLE_TEXT_UNRELATED)
        assert isinstance(result, SimilarityResult)
        assert result.ensemble_score < 0.5, (
            f"Textos não relacionados devem ter ensemble_score baixo, obtido: {result.ensemble_score}"
        )

    def test_result_structure_complete(self, analyzer):
        result = analyzer.analyze(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        assert isinstance(result, SimilarityResult)
        assert hasattr(result, "cosine_score")
        assert hasattr(result, "jaccard_char_score")
        assert hasattr(result, "jaccard_word_score")
        assert hasattr(result, "embedding_score")
        assert hasattr(result, "ensemble_score")
        assert hasattr(result, "classification")
        assert hasattr(result, "latency_seconds")

    def test_all_scores_bounded(self, analyzer):
        result = analyzer.analyze(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        assert isinstance(result, SimilarityResult)
        for score in [result.cosine_score, result.jaccard_char_score,
                      result.embedding_score, result.ensemble_score]:
            assert 0.0 <= score <= 1.0, f"Score fora de [0,1]: {score}"

    def test_empty_text_returns_error(self, analyzer):
        result = analyzer.analyze("", SAMPLE_TEXT_NLP)
        assert isinstance(result, ErrorResult)
        assert result.status == "error"

    def test_to_dict_serializable(self, analyzer):
        result = analyzer.analyze(SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR)
        assert isinstance(result, SimilarityResult)
        d = result.to_dict()
        # Deve ser serializável em JSON sem erros
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_monotonic_ordering(self, analyzer):
        """
        Teste crítico de monotonicidade:
        score(idêntico) > score(similar) > score(não-relacionado)
        """
        score_identical = analyzer.analyze(
            SAMPLE_TEXT_NLP, SAMPLE_TEXT_NLP
        ).ensemble_score

        score_similar = analyzer.analyze(
            SAMPLE_TEXT_NLP, SAMPLE_TEXT_IR
        ).ensemble_score

        score_unrelated = analyzer.analyze(
            SAMPLE_TEXT_NLP, SAMPLE_TEXT_UNRELATED
        ).ensemble_score

        assert score_identical > score_similar, (
            f"Idêntico ({score_identical:.4f}) deve ser > similar ({score_similar:.4f})"
        )
        assert score_similar > score_unrelated, (
            f"Similar ({score_similar:.4f}) deve ser > não-relacionado ({score_unrelated:.4f})"
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])