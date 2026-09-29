"""Token helpers with no downloads or demonstration side effects at import time."""
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer


def lower_tokenizer(sentence: str):
    return [token.lower() for token in TreebankWordTokenizer().tokenize(sentence)]


def sp_remover(arr: list, stop_words=None):
    """Accept a supplied stop list, or use an explicitly installed NLTK corpus."""
    if stop_words is None:
        from nltk.corpus import stopwords
        try:
            stop_words = stopwords.words("english")
        except LookupError as error:
            raise LookupError("Install stopwords with `python -m nltk.downloader stopwords`, or pass stop_words explicitly.") from error
    excluded = set(stop_words)
    return [word for word in arr if word not in excluded]


def word_stemmer(arr: list):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in arr]


if __name__ == "__main__":
    tokens = lower_tokenizer("THIS IS WHAT IT IS GOING TO BE")
    filtered = sp_remover(tokens, {"this", "is", "what", "it", "to", "be"})
    print(tokens, filtered, word_stemmer(filtered), sep="\n")
