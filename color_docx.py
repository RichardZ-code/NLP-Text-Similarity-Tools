"""Single-paragraph report with the original 0.8/0.9 highlighting thresholds."""
from docx import Document

from extend_docx import BRIGHT_GREEN, YELLOW, RED, _append_paragraph


def generate_docx(los: list[str], lof: list[float], doc_name="colored_docx.docx"):
    document = Document()
    if los or lof:
        _append_paragraph(document, los, lof, 0.8, 0.9)
    document.save(doc_name)
