from __future__ import annotations

import re
import string
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

from config.parametros import NLP_CONFIG


# NLTK BOOTSTRAP

def _ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("stemmers/rslp", "rslp"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_resources()


# ABSTRACT BASE STAGE

class PipelineStage(ABC):
    """Contrato para estágios de processamento composíveis."""

    @abstractmethod
    def process(self, text: str) -> str:
        """Transforma o texto e retorna o resultado."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# STAGES COMPARTILHADOS (ambos os pipelines)

class LowercaseStage(PipelineStage):
    """Normalização de caixa. Etapa universal."""

    def process(self, text: str) -> str:
        return text.lower()


class UnicodeNormalizationStage(PipelineStage):

    def __init__(self, remove_accents: bool = False):
        self.remove_accents = remove_accents

    def process(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        if self.remove_accents:
            normalized = "".join(
                c for c in normalized
                if unicodedata.category(c) != "Mn"
            )
        return unicodedata.normalize("NFC", normalized)


class RemoveURLStage(PipelineStage):
    """Remove URLs e referências web."""

    _PATTERN = re.compile(
        r"https?://\S+|www\.\S+|ftp://\S+",
        re.IGNORECASE
    )

    def process(self, text: str) -> str:
        return self._PATTERN.sub(" ", text)


class RemoveCitationStage(PipelineStage):

    _ABNT = re.compile(r"\([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][A-ZÁÉÍÓÚÀÂÊÔÃÕÇa-z]+(?:\s+et\s+al\.?)?,\s*\d{4}(?:,\s*p\.\s*\d+(?:-\d+)?)?\)", re.UNICODE)
    _NUMERIC = re.compile(r"\[\d+(?:,\s*\d+)*\]")

    def process(self, text: str) -> str:
        text = self._ABNT.sub(" ", text)
        text = self._NUMERIC.sub(" ", text)
        return text


class RemoveNumbersStage(PipelineStage):
    """Remove sequências numéricas isoladas."""

    _PATTERN = re.compile(r"\b\d+\b")

    def process(self, text: str) -> str:
        return self._PATTERN.sub(" ", self._PATTERN.sub(" ", text))


class RemovePunctuationStage(PipelineStage):
    """Remove pontuação, preservando hífens em palavras compostas."""

    _PUNCT = re.compile(
        r"[^\w\s\-]|(?<!\w)\-(?!\w)",
        re.UNICODE
    )

    def process(self, text: str) -> str:
        return self._PUNCT.sub(" ", text)


class WhitespaceNormalizationStage(PipelineStage):
    """Colapsa espaços múltiplos e strip."""

    def process(self, text: str) -> str:
        return " ".join(text.split())


# STAGES EXCLUSIVOS DE VETORIZAÇÃO (TF-IDF)

class StopwordRemovalStage(PipelineStage):

    def __init__(self):
        _ensure_nltk_resources()
        self._stopwords = set(stopwords.words("portuguese"))
        self._stopwords.update(NLP_CONFIG.academic_stopwords)

    def process(self, text: str) -> str:
        tokens = text.split()
        return " ".join(t for t in tokens if t not in self._stopwords)


class StemmingStage(PipelineStage):

    def __init__(self):
        _ensure_nltk_resources()
        self._stemmer = RSLPStemmer()

    def process(self, text: str) -> str:
        tokens = text.split()
        return " ".join(self._stemmer.stem(t) for t in tokens)


# STAGES EXCLUSIVOS DE EMBEDDING

class SentenceBoundaryPreservationStage(PipelineStage):

    _EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
    _HYPHENATED_BREAK = re.compile(r"-\s*\n\s*")  # quebra de linha com hífen (PDF)

    def process(self, text: str) -> str:
        # Reconstrói palavras quebradas por hífen no final de linha (artefato PDF)
        text = self._HYPHENATED_BREAK.sub("", text)
        text = self._EXCESSIVE_NEWLINES.sub("\n\n", text)
        return text.strip()


# CLEANERS COMPOSTOS

@dataclass
class PipelineStats:
    """Estatísticas do processamento de texto."""
    input_length: int = 0
    output_length: int = 0
    stages_applied: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.input_length == 0:
            return 0.0
        return 1.0 - (self.output_length / self.input_length)


class TextPipeline:
    """
    Pipeline de processamento composível.

    Implementa o padrão Chain of Responsibility com coleta de
    estatísticas por estágio para auditabilidade e debugging.
    """

    def __init__(self, stages: List[PipelineStage]):
        self._stages = stages

    def process(self, text: str) -> tuple[str, PipelineStats]:
        stats = PipelineStats(input_length=len(text), stages_applied=len(self._stages))

        for stage in self._stages:
            text = stage.process(text)

        stats.output_length = len(text)
        return text, stats

    def process_text(self, text: str) -> str:
        """Interface simplificada sem stats."""
        result, _ = self.process(text)
        return result

    def __repr__(self) -> str:
        stages_str = " → ".join(repr(s) for s in self._stages)
        return f"TextPipeline([{stages_str}])"


# FACTORY DE PIPELINES


class PipelineFactory:
    @staticmethod
    def create_vectorization_pipeline() -> TextPipeline: # Pipeline para vetorização TF-IDF

        return TextPipeline([
            LowercaseStage(),
            UnicodeNormalizationStage(remove_accents=True),
            RemoveURLStage(),
            RemoveCitationStage(),
            RemoveNumbersStage(),
            RemovePunctuationStage(),
            StopwordRemovalStage(),
            StemmingStage(),
            WhitespaceNormalizationStage(),
        ])

    @staticmethod
    def create_embedding_pipeline() -> TextPipeline: # Pipeline para Sentence Embeddings
        return TextPipeline([
            SentenceBoundaryPreservationStage(),
            UnicodeNormalizationStage(remove_accents=False),
            RemoveURLStage(),
            RemoveCitationStage(),
            WhitespaceNormalizationStage(),
        ])

    @staticmethod
    def create_jaccard_pipeline() -> TextPipeline: # Pipeline para Jaccard N-Gram
        return TextPipeline([
            LowercaseStage(),
            UnicodeNormalizationStage(remove_accents=True),
            RemoveURLStage(),
            RemoveCitationStage(),
            RemovePunctuationStage(),
            WhitespaceNormalizationStage(),
        ])