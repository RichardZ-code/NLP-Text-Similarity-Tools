"""Small tied-embedding GloVe-style model; not Stanford's two-table implementation."""
import pickle

import numpy as np
import scipy.sparse as sp


def cosine(vec1, vec2):
    """Return cosine similarity, defining similarity to a zero vector as zero."""
    a, b = np.asarray(vec1, dtype=float), np.asarray(vec2, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("cosine requires equally sized finite vectors")
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a / norm_a, b / norm_b), -1.0, 1.0))


class GloVe:
    def __init__(self, dim=5, max_count=100, alpha=0.5, max_loss=10.0,
                 learning_rate=0.02, seed=None):
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 1:
            raise ValueError("dim must be a positive integer")
        for name, value in (("max_count", max_count), ("alpha", alpha),
                            ("max_loss", max_loss), ("learning_rate", learning_rate)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
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

    @classmethod
    def load_file(cls, filepath):
        """Load a trusted local pickle. Never load an untrusted model file."""
        with open(filepath, "rb") as stream:
            state = pickle.load(stream)
        instance = cls()
        instance.__dict__.update(state)
        return instance

    def save_file(self, filename):
        with open(filename, "wb") as stream:
            pickle.dump(self.__dict__, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def fit_vectors(self, matrix, epochs=5):
        """Fit positive off-diagonal co-occurrences, resetting weights each call.

        Corpus emits one entry per unordered word pair. Word and context roles
        share one vector table. A pair therefore uses both pre-update vectors
        when differentiating 0.5 * weight * (dot + biases - log(count))**2.
        """
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not sp.issparse(matrix) or len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square and sparse")
        matrix = matrix.astype(np.float64).tocoo(copy=True)
        if matrix.nnz == 0 or not np.all(np.isfinite(matrix.data)) or np.any(matrix.data <= 0):
            raise ValueError("matrix must contain finite positive co-occurrence counts")
        matrix.sum_duplicates()
        if np.any(matrix.row == matrix.col):
            raise ValueError("self-pairs are not supported; use Corpus to build the matrix")
        rng = np.random.RandomState(self.seed)
        self.word_vectors = (rng.rand(matrix.shape[0], self.dim) - 0.5) / self.dim
        self.word_biases = np.zeros(matrix.shape[0], dtype=np.float64)
        self.vectors_sum_gradients = np.ones_like(self.word_vectors)
        self.biases_sum_gradients = np.ones_like(self.word_biases)
        indices = np.arange(matrix.nnz)
        for _ in range(epochs):
            rng.shuffle(indices)
            for index in indices:
                i, j, count = matrix.row[index], matrix.col[index], matrix.data[index]
                old_i, old_j = self.word_vectors[i].copy(), self.word_vectors[j].copy()
                prediction = np.dot(old_i, old_j) + self.word_biases[i] + self.word_biases[j]
                weight = min(1.0, count / self.max_count) ** self.alpha
                residual = float(np.clip(weight * (prediction - np.log(count)),
                                         -self.max_loss, self.max_loss))
                grad_i, grad_j = residual * old_j, residual * old_i
                self.word_vectors[i] -= self.learning_rate * grad_i / np.sqrt(self.vectors_sum_gradients[i])
                self.word_vectors[j] -= self.learning_rate * grad_j / np.sqrt(self.vectors_sum_gradients[j])
                self.vectors_sum_gradients[i] += grad_i**2
                self.vectors_sum_gradients[j] += grad_j**2
                # Distinct word indices must each receive their own bias update.
                for word in (i, j):
                    self.word_biases[word] -= self.learning_rate * residual / np.sqrt(self.biases_sum_gradients[word])
                    self.biases_sum_gradients[word] += residual**2
        return self

    def add_dictionary(self, dictionary):
        copied = dict(dictionary)
        if (any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in copied.values())
                or sorted(copied.values()) != list(range(len(copied)))):
            raise ValueError("dictionary indices must be unique and contiguous from zero")
        if self.word_vectors is not None and len(copied) != len(self.word_vectors):
            raise ValueError("dictionary size does not match the trained model")
        self.dictionary = copied
        self.inverse_dict = {value: key for key, value in copied.items()}

    def _require_ready(self):
        if self.word_vectors is None or self.dictionary is None:
            raise ValueError("train a model and attach its dictionary before querying")
        if len(self.dictionary) != len(self.word_vectors):
            raise ValueError("dictionary size does not match the trained model")

    def get_most_similar(self, word, number=5, ignore_missing=True):
        self._require_ready()
        if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or number < 0:
            raise ValueError("number must be a nonnegative integer")
        if word not in self.dictionary:
            if ignore_missing:
                return []
            raise KeyError(word)
        word_index = self.dictionary[word]
        scores = np.array([cosine(vector, self.word_vectors[word_index]) for vector in self.word_vectors])
        ranked = np.argsort(-scores, kind="stable")
        # Do not assume that self is first: ties and zero vectors can break that.
        ranked = [index for index in ranked if index != word_index][:number]
        return [(self.inverse_dict[index], float(scores[index])) for index in ranked]

    def check_similarity(self, word_a, word_b, ignore_missing=True):
        self._require_ready()
        for word in (word_a, word_b):
            if word not in self.dictionary:
                if ignore_missing:
                    return -1.0
                raise KeyError(word)
        return cosine(self.word_vectors[self.dictionary[word_a]], self.word_vectors[self.dictionary[word_b]])
