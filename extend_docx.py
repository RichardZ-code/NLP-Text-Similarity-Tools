"""Create word-score reports while preserving caller-owned input sequences."""
from math import isfinite

from docx import Document
from docx.enum.text import WD_COLOR_INDEX

RED_VALUE = 0.3
YELLOW_VALUE = 0.7
BRIGHT_GREEN = WD_COLOR_INDEX.BRIGHT_GREEN
YELLOW = WD_COLOR_INDEX.YELLOW
RED = WD_COLOR_INDEX.RED


def _append_paragraph(doc, words, scores, red_value, yellow_value):
    if len(words) != len(scores):
        raise ValueError("words and scores must have equal lengths")
    if any(not isinstance(word, str) or not word for word in words):
        raise ValueError("each token must be a nonempty string")
    if any(not isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    paragraph = doc.add_paragraph()
    opening = {"(", "[", "{", "<", "``"}
    closing = {".", "?", "!", ",", ";", ":", ")", "]", "}", ">", '"'}
    for index, (word, score) in enumerate(zip(words, scores)):
        text = word.capitalize() if index == 0 else word
        if index + 1 < len(words):
            following = words[index + 1]
            if word not in opening and following not in closing and not following.startswith("'"):
                text += " "
        run = paragraph.add_run(text)
        if yellow_value < score <= 1:
            run.font.highlight_color = BRIGHT_GREEN
        elif red_value < score <= yellow_value:
            run.font.highlight_color = YELLOW
        elif -1 < score <= red_value:
            run.font.highlight_color = RED
    return paragraph


def partial_docx(los: list[str], lof: list[float], doc):
    """Append one paragraph using the original 0.3/0.7 highlighting thresholds."""
    _append_paragraph(doc, los, lof, RED_VALUE, YELLOW_VALUE)


def whole_docx(los: list[list[str]], lof: list[list[float]], doc_name="color_doc.docx"):
    if len(los) != len(lof):
        raise ValueError("paragraphs and score rows must have equal lengths")
    document = Document()
    for words, scores in zip(los, lof):
        partial_docx(words, scores, document)
    document.save(doc_name)
