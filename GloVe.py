"""Small shared-embedding GloVe-style model for unordered co-occurrence pairs.

This keeps the original project's one-vector/one-bias-per-word design, rather
than claiming to reproduce Stanford GloVe's separate word/context tables.
Pickle persistence is for trusted local files only.
"""
import pickle
from numbers import Integral

import numpy as np
import scipy.sparse as sp


def cosine(vec1, vec2):
    """Return cosine similarity; define similarity involving a zero vector as 0."""
    a, b = np.asarray(vec1, dtype=float), np.asarray(vec2, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("vectors must have matching one-dimensional finite values")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)) if denominator else 0.0


class GloVe:
    def __init__(self, dim=5, max_count=100, alpha=0.5, max_loss=10.0, learning_rate=0.02, seed=None):
        if isinstance(dim, bool) or not isinstance(dim, Integral) or dim < 1:
            raise ValueError("dim must be a positive integer")
        for name, value in (("max_count", max_count), ("alpha", alpha), ("max_loss", max_loss), ("learning_rate", learning_rate)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if alpha > 1:
            raise ValueError("alpha must not exceed 1")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32):
            raise ValueError("seed must be None or an integer in [0, 2**32)")
        self.dim, self.max_count, self.alpha = dim, max_count, alpha
        self.max_loss, self.learning_rate, self.seed = max_loss, learning_rate, seed
        self.word_vectors = self.word_biases = None
        self.vectors_sum_gradients = self.biases_sum_gradients = None
        self.dictionary = self.inverse_dict = None
        self.loss_history = []

    @classmethod
    def load_file(cls, filepath):
        """Load only a file you trust: pickle can execute code during loading."""
        with open(filepath, "rb") as stream:
            state = pickle.load(stream)
        instance = cls()
        instance.__dict__.update(state)
        return instance

    def save_file(self, filename):
        with open(filename, "wb") as stream:
            pickle.dump(self.__dict__, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def _pair_step(self, i, j, count):
        # Both gradients must use the same pre-update parameter snapshot.
        left, right = self.word_vectors[i].copy(), self.word_vectors[j].copy()
        weight = min(1.0, count / self.max_count) ** self.alpha
        residual = np.dot(left, right) + self.word_biases[i] + self.word_biases[j] - np.log(count)
        derivative = float(np.clip(weight * residual, -self.max_loss, self.max_loss))
        left_gradient, right_gradient = derivative * right, derivative * left
        self.word_vectors[i] -= self.learning_rate * left_gradient / np.sqrt(self.vectors_sum_gradients[i])
        self.word_vectors[j] -= self.learning_rate * right_gradient / np.sqrt(self.vectors_sum_gradients[j])
        self.vectors_sum_gradients[i] += left_gradient**2
        self.vectors_sum_gradients[j] += right_gradient**2
        for index in (i, j):
            self.word_biases[index] -= self.learning_rate * derivative / np.sqrt(self.biases_sum_gradients[index])
            self.biases_sum_gradients[index] += derivative**2

    def fit_vectors(self, matrix, epochs=5):
        if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        if not sp.issparse(matrix) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square and sparse")
        entries = matrix.tocoo(copy=True)
        if not np.all(np.isfinite(entries.data)) or np.any(entries.data < 0):
            raise ValueError("co-occurrence counts must be finite and nonnegative")
        entries.sum_duplicates()
        entries.eliminate_zeros()
        if not entries.nnz or np.any(entries.row == entries.col):
            raise ValueError("matrix must contain positive, off-diagonal co-occurrences")
        if self.dictionary is not None and len(self.dictionary) != entries.shape[0]:
            raise ValueError("dictionary size must match matrix dimensions")
        rng = np.random.RandomState(self.seed)
        self.word_vectors = (rng.rand(entries.shape[0], self.dim) - 0.5) / self.dim
        self.word_biases = np.zeros(entries.shape[0])
        self.vectors_sum_gradients = np.ones_like(self.word_vectors)
        self.biases_sum_gradients = np.ones_like(self.word_biases)
        self.loss_history = [self.objective(entries)]
        indices = np.arange(entries.nnz)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for _ in range(epochs):
                rng.shuffle(indices)
                for index in indices:
                    self._pair_step(entries.row[index], entries.col[index], entries.data[index])
                self.loss_history.append(self.objective(entries))
        return self

    def objective(self, matrix):
        """Unclipped weighted least-squares objective; not a semantic quality score."""
        self._require_fitted()
        entries = matrix.tocoo()
        residuals = (np.sum(self.word_vectors[entries.row] * self.word_vectors[entries.col], axis=1)
                     + self.word_biases[entries.row] + self.word_biases[entries.col] - np.log(entries.data))
        weights = np.minimum(1.0, entries.data / self.max_count) ** self.alpha
        return float(0.5 * np.sum(weights * residuals**2))

    def _require_fitted(self):
        if self.word_vectors is None or self.word_biases is None:
            raise ValueError("fit or load a model first")

    def add_dictionary(self, dictionary):
        if (not isinstance(dictionary, dict) or not all(isinstance(word, str) for word in dictionary)
                or not all(isinstance(index, Integral) and not isinstance(index, bool) for index in dictionary.values())
                or set(dictionary.values()) != set(range(len(dictionary)))):
            raise ValueError("dictionary must map words to unique contiguous integer IDs")
        if self.word_vectors is not None and len(dictionary) != len(self.word_vectors):
            raise ValueError("dictionary size must match the trained model")
        self.dictionary = dict(dictionary)
        self.inverse_dict = {index: word for word, index in dictionary.items()}

    def _word_id(self, word, ignore_missing):
        self._require_fitted()
        if self.dictionary is None:
            raise ValueError("attach a dictionary first")
        if word not in self.dictionary and not ignore_missing:
            raise KeyError(word)
        return self.dictionary.get(word)

    def get_most_similar(self, word, number=5, ignore_missing=True):
        if isinstance(number, bool) or not isinstance(number, Integral) or number < 0:
            raise ValueError("number must be a nonnegative integer")
        index = self._word_id(word, ignore_missing)
        if index is None:
            return []
        scores = [cosine(vector, self.word_vectors[index]) for vector in self.word_vectors]
        # Exclude the query by identity, not by discarding the first sorted item.
        order = sorted((i for i in range(len(scores)) if i != index), key=lambda i: (-scores[i], i))
        return [(self.inverse_dict[i], scores[i]) for i in order[:number]]

    def check_similarity(self, word_a, word_b, ignore_missing=True):
        a, b = self._word_id(word_a, ignore_missing), self._word_id(word_b, ignore_missing)
        if a is None or b is None:
            return -1.0  # Preserve the original missing-word sentinel.
        return cosine(self.word_vectors[a], self.word_vectors[b])
