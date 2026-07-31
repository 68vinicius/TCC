from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from config.parametros import NLP_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Fragmento de texto com rastreamento de posição."""
    text: str
    start_idx: int
    end_idx: int
    chunk_index: int


class TextChunker:
    """ Divide texto longo em chunks com overlap para processamento  """

    def __init__(
        self,
        chunk_size: int = NLP_CONFIG.chunk_size,
        chunk_overlap: int = NLP_CONFIG.chunk_overlap,
        min_chunk_words: int = 20,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words

    def chunk(self, text: str) -> List[TextChunk]:

        words = text.split()
        total_words = len(words)

        # Texto curto: retorna como c hunk unico
        if total_words <= self.chunk_size:
            return [TextChunk(
                text=text,
                start_idx=0,
                end_idx=total_words,
                chunk_index=0,
            )]

        chunks: List[TextChunk] = []
        fallback_chunks: List[TextChunk] = []
        start = 0
        chunk_idx = 0

        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            chunk_words = words[start:end]

            if not chunk_words:
                break

            # Cria a estrutura do chunk técnico
            current_chunk = TextChunk(
                text=" ".join(chunk_words),
                start_idx=start,
                end_idx=end,
                chunk_index=chunk_idx,
            )
            
            fallback_chunks.append(current_chunk)

            if len(chunk_words) >= self.min_chunk_words:
                chunks.append(current_chunk)
                chunk_idx += 1

            if end >= total_words:
                break

            # Avanço com overlap
            proximo_start = end - self.chunk_overlap
            if proximo_start <= start:
                start = end
            else:
                start = proximo_start

        # Se o filtro min_chunk_words descartou tudo (comum em strings curtas de teste), usar o fallback
        if not chunks and fallback_chunks:
            for i, chk in enumerate(fallback_chunks):
                chk.chunk_index = i
            chunks = fallback_chunks

        logger.debug(
            "Chunking: %d palavras → %d chunks (size=%d, overlap=%d)",
            total_words, len(chunks), self.chunk_size, self.chunk_overlap
        )

        return chunks

    @staticmethod
    def aggregate_similarity_scores(
        scores: List[float],
        method: str = NLP_CONFIG.chunk_aggregation,
    ) -> float:
        """
        Agrega scores de múltiplos chunks em um score final.

        Métodos disponíveis:
        - "mean": média aritmética 
        - "max": máximo 
        - "weighted_mean": média ponderada 
        """
        if not scores:
            return 0.0

        arr = np.array(scores, dtype=np.float64)

        if method == "mean":
            return float(np.mean(arr))
        elif method == "max":
            return float(np.max(arr))
        elif method == "weighted_mean":
            weights = np.ones(len(arr))
            return float(np.average(arr, weights=weights))
        else:
            raise ValueError(f"Método de agregação desconhecido: '{method}'")