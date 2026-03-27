import re
import string

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        # Remove pontuação usando regex eficiente
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        # Remove números
        text = re.sub(r'\d+', '', text)
        return " ".join(text.split())