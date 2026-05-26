import re
import string

import nltk

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer


# =========================================================
# PIPELINE STAGES
# =========================================================

class PipelineStage:

    def process(self, text: str) -> str:
        raise NotImplementedError


# =========================================================
# LOWERCASE
# =========================================================

class LowercaseStage(PipelineStage):

    def process(self, text: str) -> str:
        return text.lower()


# =========================================================
# REMOVE URL
# =========================================================

class RemoveURLStage(PipelineStage):

    def process(self, text: str) -> str:

        return re.sub(
            r'https?://\S+|www\.\S+',
            ' ',
            text
        )


# =========================================================
# REMOVE CITATIONS
# =========================================================

class RemoveCitationStage(PipelineStage):

    def process(self, text: str) -> str:

        # Remove padrões ABNT básicos
        text = re.sub(
            r'\([A-Z]+,\s?\d{4}\)',
            ' ',
            text
        )

        return text


# =========================================================
# REMOVE NUMBERS
# =========================================================

class RemoveNumbersStage(PipelineStage):

    def process(self, text: str) -> str:

        return re.sub(r'\d+', ' ', text)


# =========================================================
# REMOVE PUNCT
# =========================================================

class RemovePunctuationStage(PipelineStage):

    def process(self, text: str) -> str:

        return re.sub(
            f"[{re.escape(string.punctuation)}]",
            " ",
            text
        )


# =========================================================
# STOPWORDS
# =========================================================

class StopwordStage(PipelineStage):

    def __init__(self):

        nltk.download('stopwords', quiet=True)

        self.stopwords = set(
            stopwords.words('portuguese')
        )

        self.stopwords.update([
            'figura',
            'tabela',
            'capítulo',
            'seção',
            'página',
            'autor',
            'ano',
            'referências',
            'metodologia',
            'resultados'
        ])

    def process(self, text: str) -> str:

        return " ".join(

            token

            for token in text.split()

            if token not in self.stopwords
        )


# =========================================================
# STEMMING
# =========================================================

class StemmingStage(PipelineStage):

    def __init__(self):

        nltk.download('rslp', quiet=True)

        self.stemmer = RSLPStemmer()

    def process(self, text: str) -> str:

        return " ".join(

            self.stemmer.stem(token)

            for token in text.split()
        )


# =========================================================
# WHITESPACE
# =========================================================

class WhitespaceStage(PipelineStage):

    def process(self, text: str) -> str:

        return " ".join(text.split())


# =========================================================
# MAIN CLEANER
# =========================================================

class TextCleaner:

    def __init__(self):

        self.pipeline = [

            LowercaseStage(),

            RemoveURLStage(),

            RemoveCitationStage(),

            RemoveNumbersStage(),

            RemovePunctuationStage(),

            StopwordStage(),

            StemmingStage(),

            WhitespaceStage()
        ]

    def clean(self, text: str) -> str:

        if not text:
            return ""

        for stage in self.pipeline:

            text = stage.process(text)

        return text