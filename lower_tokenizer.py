"""Tokenization helpers without import-time downloads or example execution."""
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def lower_tokenizer(sentence: str):
    return [word.lower() for word in word_tokenize(sentence, preserve_line=True)]


def sp_remover(arr: list, stop_words=None):
    """Supply a stop-word collection for offline use, or install NLTK stopwords."""
    if stop_words is None:
        try:
            stop_words = stopwords.words("english")
        except LookupError as error:
            raise LookupError("Install stopwords with 'python -m nltk.downloader stopwords', "
                              "or pass stop_words explicitly.") from error
    excluded = set(stop_words)
    return [word for word in arr if word not in excluded]


def word_stemmer(arr: list):
    porter = PorterStemmer()
    return [porter.stem(word) for word in arr]
