import time
import logging
from typing import Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self, vectorizer, cleaner):
        self.vectorizer = vectorizer
        self.cleaner = cleaner

    def _get_char_ngrams(self, text: str, n=3):
        return set([text[i:i+n] for i in range(len(text)-n+1)])

    def compare(self, t1_raw: str, t2_raw: str) -> Dict[str, Any]:
        start = time.perf_counter()
        
        t1 = self.cleaner.clean(t1_raw)
        t2 = self.cleaner.clean(t2_raw)

        if not t1 or not t2:
            return {"status": "error", "message": "Texto vazio após limpeza"}

        # Cálculo Cosseno
        matrix = self.vectorizer.fit_transform([t1, t2])
        cos_sim = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])

        # Cálculo Jaccard
        s1, s2 = self._get_char_ngrams(t1), self._get_char_ngrams(t2)
        jac_sim = float(len(s1 & s2) / len(s1 | s2)) if (s1 | s2) else 0

        return {
            "cosseno": cos_sim,
            "jaccard": jac_sim,
            "t1_limpo": t1,
            "t2_limpo": t2,
            "tempo": time.perf_counter() - start,
            "status": "success"
        }