"""Small tied-embedding GloVe-style learner for unordered co-occurrence pairs.

This preserves the original project's single embedding table. It is not the
canonical two-table Stanford implementation. Pickle files must be trusted.
"""

import pickle

import numpy as np
import scipy.sparse as sp


def cosine(vec1, vec2):
    """Cosine similarity; zero-norm vectors have similarity zero."""
    a, b = np.asarray(vec1, dtype=float), np.asarray(vec2, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.all(np.isfinite([a, b])):
        raise ValueError("cosine requires equally sized finite vectors")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


class GloVe:
    def __init__(self, dim=5, max_count=100, alpha=0.5, max_loss=10.0,
                 learning_rate=0.02, seed=None):
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        for name, value in (("max_count", max_count), ("alpha", alpha),
                            ("max_loss", max_loss), ("learning_rate", learning_rate)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                                 or not 0 <= seed < 2**32):
            raise ValueError("seed must be None or an integer in [0, 2**32)")
        self.seed = seed
        self.dim = dim
        self.max_count = max_count
        self.alpha = alpha
        self.max_loss = max_loss
        self.learning_rate = learning_rate
        self.word_vectors = None
        self.word_biases = None
        self.vectors_sum_gradients = None
        self.biases_sum_gradients = None
        self.dictionary = None
        self.inverse_dict = None
        self.loss_history = []

    @classmethod
    def load_file(cls, filepath):
        """Load a locally trusted pickle. Never load a downloaded unknown model."""
        with open(filepath, "rb") as stream:
            state = pickle.load(stream)
        instance = cls()
        instance.__dict__.update(state)
        return instance

    def save_file(self, filename):
        with open(filename, "wb") as stream:
            pickle.dump(self.__dict__, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _matrix(matrix):
        if not sp.issparse(matrix) or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square and sparse")
        matrix = matrix.tocoo(copy=True).astype(np.float64)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if not matrix.nnz or not np.all(np.isfinite(matrix.data)) or np.any(matrix.data <= 0):
            raise ValueError("matrix must contain finite positive co-occurrence counts")
        if np.any(matrix.row == matrix.col):
            raise ValueError("this unordered-pair model excludes self co-occurrences")
        return matrix

    def _step(self, i, j, count):
        # Both derivatives must use the same pre-update parameter snapshot.
        left = self.word_vectors[i].copy()
        right = self.word_vectors[j].copy()
        residual = np.dot(left, right) + self.word_biases[i] + self.word_biases[j] - np.log(count)
        weight = min(1.0, count / self.max_count) ** self.alpha
        gradient = float(np.clip(weight * residual, -self.max_loss, self.max_loss))
        grad_i, grad_j = gradient * right, gradient * left
        self.word_vectors[i] -= self.learning_rate * grad_i / np.sqrt(self.vectors_sum_gradients[i])
        self.word_vectors[j] -= self.learning_rate * grad_j / np.sqrt(self.vectors_sum_gradients[j])
        self.vectors_sum_gradients[i] += grad_i**2
        self.vectors_sum_gradients[j] += grad_j**2
        for index in (i, j):
            self.word_biases[index] -= self.learning_rate * gradient / np.sqrt(self.biases_sum_gradients[index])
            self.biases_sum_gradients[index] += gradient**2

    def objective_loss(self, matrix):
        """Half weighted squared residual, before gradient clipping."""
        if self.word_vectors is None:
            raise ValueError("train the model first")
        matrix = self._matrix(matrix)
        if matrix.shape[0] != len(self.word_vectors):
            raise ValueError("matrix size differs from the trained vocabulary")
        residual = (np.sum(self.word_vectors[matrix.row] * self.word_vectors[matrix.col], axis=1)
                    + self.word_biases[matrix.row] + self.word_biases[matrix.col]
                    - np.log(matrix.data))
        weights = np.minimum(1.0, matrix.data / self.max_count) ** self.alpha
        return float(0.5 * np.sum(weights * residual**2))

    def fit_vectors(self, matrix, epochs=5):
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        matrix = self._matrix(matrix)
        if self.dictionary is not None and len(self.dictionary) != matrix.shape[0]:
            raise ValueError("dictionary size differs from the matrix")
        random_state = np.random.RandomState(self.seed)
        self.word_vectors = (random_state.rand(matrix.shape[0], self.dim) - 0.5) / self.dim
        self.word_biases = np.zeros(matrix.shape[0], dtype=np.float64)
        self.vectors_sum_gradients = np.ones_like(self.word_vectors)
        self.biases_sum_gradients = np.ones_like(self.word_biases)
        self.loss_history = [self.objective_loss(matrix)]
        order = np.arange(matrix.nnz)
        for _ in range(epochs):
            random_state.shuffle(order)
            for item in order:
                self._step(matrix.row[item], matrix.col[item], matrix.data[item])
            loss = self.objective_loss(matrix)
            if not np.isfinite(loss):
                raise FloatingPointError("non-finite training loss; reduce the learning rate")
            self.loss_history.append(loss)
        return self

    def add_dictionary(self, dictionary):
        if (not all(isinstance(word, str) for word in dictionary)
                or not all(isinstance(index, int) and not isinstance(index, bool) for index in dictionary.values())
                or sorted(dictionary.values()) != list(range(len(dictionary)))):
            raise ValueError("dictionary must map words to contiguous unique integer IDs")
        if self.word_vectors is not None and len(dictionary) != len(self.word_vectors):
            raise ValueError("dictionary size differs from trained vectors")
        self.dictionary = dict(dictionary)
        self.inverse_dict = {index: word for word, index in dictionary.items()}

    def _word_id(self, word, ignore_missing):
        if self.word_vectors is None or self.dictionary is None:
            raise ValueError("train the model and attach its dictionary first")
        if word not in self.dictionary:
            if ignore_missing:
                return None
            raise KeyError(f"word not in vocabulary: {word}")
        return self.dictionary[word]

    def get_most_similar(self, word, number=5, ignore_missing=True):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError("number must be a nonnegative integer")
        index = self._word_id(word, ignore_missing)
        if index is None:
            return []
        matches = [(i, cosine(vec, self.word_vectors[index]))
                   for i, vec in enumerate(self.word_vectors) if i != index]
        matches.sort(key=lambda pair: (-pair[1], pair[0]))
        return [(self.inverse_dict[i], similarity) for i, similarity in matches[:number]]

    def check_similarity(self, word_a, word_b, ignore_missing=True):
        a = self._word_id(word_a, ignore_missing)
        b = self._word_id(word_b, ignore_missing)
        if a is None or b is None:
            return -1.0  # Retained for compatibility with the original API.
        return cosine(self.word_vectors[a], self.word_vectors[b])
