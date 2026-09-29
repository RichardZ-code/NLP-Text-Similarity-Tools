"""Tokenization/stemming without network or print side effects at import."""
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def lower_tokenizer(sentence: str):
    return [word.lower() for word in TreebankWordTokenizer().tokenize(sentence)]


def sp_remover(arr: list, stop_words=None):
    """Use a supplied stopword list, or explicitly installed NLTK stopwords."""
    if stop_words is None:
        try:
            stop_words = stopwords.words("english")
        except LookupError as error:
            raise LookupError(
                "Pass stop_words explicitly or run: python -m nltk.downloader stopwords"
            ) from error
    excluded = set(stop_words)
    return [word for word in arr if word not in excluded]


def word_stemmer(arr: list):
    porter = PorterStemmer()
    return [porter.stem(word) for word in arr]
