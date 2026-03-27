from abc import ABC, abstractmethod
from pypdf import PdfReader
from docx import Document

class DocumentHandler(ABC):
    @abstractmethod
    def extract(self, file) -> str:
        pass

class PDFHandler(DocumentHandler):
    def extract(self, file) -> str:
        reader = PdfReader(file)
        return "".join([p.extract_text() for p in reader.pages if p.extract_text()])

class DocxHandler(DocumentHandler):
    def extract(self, file) -> str:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

class TextHandler(DocumentHandler):
    def extract(self, file) -> str:
        return file.read().decode("utf-8")

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
            raise ValueError(f"Extensão .{ext} não suportada.")
        return handler.extract(file)