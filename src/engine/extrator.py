from abc import ABC, abstractmethod

from pypdf import PdfReader
from docx import Document

import pdfplumber
from charset_normalizer import from_bytes


# =========================================================
# ABSTRACT
# =========================================================

class DocumentHandler(ABC):

    @abstractmethod
    def extract(self, file) -> str:
        pass


# =========================================================
# PDF
# =========================================================

class PDFHandler(DocumentHandler):

    def extract(self, file) -> str:

        text = []

        with pdfplumber.open(file) as pdf:

            for page in pdf.pages:

                content = page.extract_text()

                if content:
                    text.append(content)

        return "\n".join(text)


# =========================================================
# DOCX
# =========================================================

class DocxHandler(DocumentHandler):

    def extract(self, file) -> str:

        doc = Document(file)

        return "\n".join(
            para.text
            for para in doc.paragraphs
        )


# =========================================================
# TXT
# =========================================================

class TextHandler(DocumentHandler):

    def extract(self, file) -> str:

        raw = file.read()

        detected = from_bytes(raw).best()

        return str(detected)


# =========================================================
# FACTORY
# =========================================================

class DocumentFactory:

    _map = {

        "pdf": PDFHandler(),

        "docx": DocxHandler(),

        "txt": TextHandler()
    }

    @classmethod
    def get_text(cls, file) -> str:

        ext = file.name.split(".")[-1].lower()

        handler = cls._map.get(ext)

        if not handler:
            raise ValueError(
                f"Extensão .{ext} não suportada."
            )

        return handler.extract(file)