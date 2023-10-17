
import scipy.sparse as sp
import numpy as np
import pickle
from math import *
from datetime import datetime

def cosine(vec1, vec2):
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

class GloVe():

    def __init__(self, dim=5, max_count=100, alpha=0.5, max_loss=10.0, learning_rate=0.02, seed=None) -> None:
        if not seed:
            self.seed = datetime.today().timestamp()
        else:
            self.seed = seed
        self.learning_rate = learning_rate
        self.dim = dim
        self.max_count = max_count
        self.alpha = alpha
        self.max_loss = max_loss

        # Value
        self.word_vectors = None
        self.word_biases = None
        # Gradient
        self.vectors_sum_gradients = None
        self.biases_sum_gradients = None
        
        self.dictionary = None
        self.inverse_dict = None

    @classmethod
    def load_file(cls, filepath):
        with open(filepath, 'rb') as f:
            instance = GloVe()
            instance.__dict__ = pickle.load(f)
            return instance

    def save_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
    
                

    def fit_vectors(self, matrix: sp.coo_matrix, epochs=5):

        
        random_state = np.random.RandomState()
        
        # Initialize vector matrixes
        self.word_vectors = ((random_state.rand(matrix.shape[0], self.dim)\
                             - 0.5) / self.dim)
        self.word_biases = np.zeros(matrix.shape[0], dtype=np.float64)

        self.vectors_sum_gradients = np.ones_like(self.word_vectors)
        self.biases_sum_gradients = np.ones_like(self.word_biases)

        # Initail idx list
        shuffle_indices = np.arange(matrix.nnz, dtype=np.int32)

        no_coocurrences = matrix.row.shape[0]

        # Machine Learning Steps
        for _ in range(epochs):
            # Shuffle the index list
            random_state.shuffle(shuffle_indices)

            for j in range(no_coocurrences):
                shuffle_idx = shuffle_indices[j]
                word_i = matrix.row[shuffle_idx]
                word_j = matrix.col[shuffle_idx]
                count = matrix.data[shuffle_idx]

                # Prediction step: Use GloVe Model to predict
                prediction = 0.0
                # Get the dot production
                for i in range(self.dim):
                    prediction += self.word_vectors[word_i, i] * \
                                        self.word_vectors[word_j, i]
                # Add the biases
                prediction += self.word_biases[word_i] + self.word_biases[word_j]

                # Compute the Count Weight
                weight = min(1.0, (count / self.max_count)) ** self.alpha

                
                # Compute the loss from prediction
                loss = weight * (prediction - log(count)) # ** 2

                # Clip the loss for numerical stability
                if loss < - self.max_loss:
                    loss = - self.max_loss
                elif loss > self.max_loss:
                    loss = self.max_loss
                
                # Update steps: Gradient Descent
                # Update vectors
                for i in range(self.dim):
                    # word_i is moving towards/away from word_j
                    adjusted_learning_rate = self.learning_rate / sqrt(self.vectors_sum_gradients[word_i, i])
                    gradient = loss * self.word_vectors[word_j, i]
                    self.word_vectors[word_i, i] -= adjusted_learning_rate * gradient
                    # Adjust the position gradient
                    self.vectors_sum_gradients[word_i, i] += gradient ** 2

                    # Same for word_j
                    adjusted_learning_rate = self.learning_rate / sqrt(self.vectors_sum_gradients[word_j, i])
                    gradient = loss * self.word_vectors[word_i, i]
                    self.word_vectors[word_j, i] -= adjusted_learning_rate * gradient

                    self.vectors_sum_gradients[word_j, i] += gradient ** 2
                
                # Update biases
                adjusted_learning_rate = self.learning_rate / sqrt(self.biases_sum_gradients[word_i])
                self.word_biases[word_i] -= adjusted_learning_rate * loss
                self.biases_sum_gradients[word_i] += loss ** 2

                adjusted_learning_rate = self.learning_rate / sqrt(self.biases_sum_gradients[word_i])
                self.word_biases[word_i] -= adjusted_learning_rate * loss
                self.biases_sum_gradients[word_i] += loss ** 2

                print(f'Training: {j/no_coocurrences} Epoch {_} / {epochs}', end='\r')

    def add_dictionary(self, dictionary):
        
        self.dictionary = dictionary
        self.inverse_dict = {v:k for k, v in self.dictionary.items()}

    def get_most_similar(self, word, number=5, ignore_missing=True):
        try:
            word_idx = self.dictionary[word]
        except KeyError:
            if not ignore_missing:
                raise Exception(f'ERROR: Unable to find {word} in dictionary')
            else:
                return []
        
        dst = np.array([cosine(vec, self.word_vectors[word_idx]) for vec in self.word_vectors])


        word_ids = np.argsort(-dst)

        return [(self.inverse_dict[x], dst[x]) for x in word_ids[1:number+1] if x in self.inverse_dict]

    def check_similarity(self, word_a, word_b, ignore_missing=True) -> float:
        try:
            idx_a = self.dictionary[word_a]
        except KeyError:
            if not ignore_missing:
                raise Exception(f'ERROR: Unable to find {word_a} in dictionary')
            else:
                return -1.
        try:
            idx_b = self.dictionary[word_b]
        except KeyError:
            if not ignore_missing:
                raise Exception(f'ERROR: Unable to find {word_b} in dictionary')
            else:
                return -1.
        
        return cosine(self.word_vectors[idx_a], self.word_vectors[idx_b])
            