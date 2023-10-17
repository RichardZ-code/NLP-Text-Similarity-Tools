from re import T
import numpy as np
import scipy.sparse as sp
from typing import Callable, List
import pickle
import nltk
from GloVe import *

nltk.download('punkt')

def search(arr: list, le, hi, x):
    while hi >= le:
        mid = (hi + le) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            hi = mid - 1
        else:
            le = mid + 1

    return hi + 1



class Corpus:

    def __init__(self, dictionary={}) -> None:
        self.dictionary = dictionary
        self.matrix = None

    @classmethod
    def load_file(cls, filepath):
        instance = Corpus()

        with open(filepath, 'rb') as f:
            instance.dictionary, instance.matrix = pickle.load(f)
        
        return instance
    
    def save_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.dictionary, self.matrix), f, protocol=pickle.HIGHEST_PROTOCOL)

    def fit_matrix(self, corpus: List[List[str]], window_size=5):
        self.matrix = con_coo_matrix(corpus, self.dictionary, window_size)

        

def words_to_ids(words, word_ids, dictionary):
    for word in words:
        word_id = dictionary.setdefault(word, len(dictionary))
        word_ids.append(word_id)

    return True

def increment_matrix(indice_mat: List[int], data_mat: List[List[float]], row: int, col: int, increment: float):
    while row >= len(indice_mat):
        indice_mat.append([])
        data_mat.append([])
    
    row_indices = indice_mat[row]
    row_data = data_mat[row]
    

    # Find where to insert the col
    idx = search(row_indices, 0, len(row_indices)-1, col)

    # If excceeded the list
    if idx == len(row_indices):
        row_indices.insert(idx, col)
        row_data.insert(idx, increment)
        return
    
    # Check if existed
    if row_indices[idx] == col:
        row_data[idx] += increment
    else:
        row_indices.insert(idx, col)
        row_data.insert(idx, increment)

def tokenizer(s: str) -> List[str]:
    return nltk.tokenize.word_tokenize(s)

def prepare_data(corpus: str, tokenizer: Callable=tokenizer, separater='\n') ->List[List[str]]:
    '''
    Args:
        str corpus: Texts to prepare 
        Callable(str) -> List[str] tokenzier: Callable tokenizer that tokenizes words
        char separator: The charactor that separates the texts
    Return:
        List[List[str]] Tokenized strings 
    '''
    list_str = corpus.split(separater)
    res = []
    for s in list_str:
        res.append(tokenizer(s))
    return res



def con_coo_matrix(corpus, dictionary, window_size):
    indices_matrix = []
    data_matrix = []
   
    idx = 0
    for words in corpus:
        
        word_ids = [] # Initialize word ids in each scope
        if (words_to_ids(words, word_ids, dictionary)): # convert words to ids

            wordslen = len(word_ids)
            
            for i in range(wordslen): # Iterate each words in this scope
                outer_word = word_ids[i]    

                window_size_ = min(i + window_size + 1, wordslen)
                # Consider the following words
                for j in range(i, window_size_):
                    inner_word = word_ids[j]

                    if inner_word == outer_word:
                        continue
                    
                    increment = 1.0 / (j - i)
                    
                    # Compare Words' Ids and make frequency increment
                    if inner_word < outer_word:
                        increment_matrix(indices_matrix, data_matrix, inner_word, outer_word, increment)
                    else:
                        increment_matrix(indices_matrix, data_matrix, outer_word, inner_word, increment)
        print(f'Generatring Matrix: {idx/len(corpus):.2f}', end='\r')
        idx += 1
                    
    return list_to_coo_matrix(indices_matrix, data_matrix, len(dictionary))

def list_to_coo_matrix(indices_mat, data_mat, shape):

    size = 0 
    for i in range(len(indices_mat)):
        size += len(indices_mat[i])

    row = np.empty(size, dtype=np.int32)
    col = np.empty(size, dtype=np.int32)
    data = np.empty(size, dtype=np.float64)
    
    k = 0
    for i in range(len(indices_mat)):
        for j in range(len(indices_mat[i])):
            row[k] = i
            col[k] = indices_mat[i][j]
            data[k] = data_mat[i][j]
            k += 1
    
    return sp.coo_matrix((data, (row, col)), shape=(shape, shape), dtype=np.float64)


if __name__ == "__main__":
    s = 'Deleted in liver cancer 1 (DLC1) is a tumor suppressor gene that was first discovered because it was deleted in hepatocarcinomas, and was later found to be downregulated, via genetic, epigenetic, and post-translational mechanisms, in many other tumor types, including colon/rectum, breast, prostate, and lung. The downregulation of DLC1 in tumors has been mostly attributed either to gene deletion or promoter DNA methylation, although other changes may also contribute to its decreased expression. Very recently, a fine detailed analysis of several tumor types in the TCGA dataset revealed that tumor-associated point mutations in DLC1 also occur frequently and can impair the biological functions of its encoded protein by several mechanisms.'
    print(prepare_data(s))

    # a = Corpus.load_file('saved-dict')
   
    # b = GloVe.load_model('saved-model')
    # b.add_dictionary(a.dictionary)

    # print(b.get_most_similar('aaa', 10, ignore_missing=True)) 
    # # output: [('4th', 0.9979431015431222), ('worldwide', 0.9978754554438284), ('malignancy', 0.995821489418576), ('upper', 0.9948895168668315),
    # # ('leading', 0.9940600146574153), ('fourth', 0.991413885448219), ('remain', 0.9888348240944643), 
    # # ('mechanism', 0.9879301238391083), ('carcinomas', 0.987503077756682), ('advances', 0.9869173792115726)]
    # print(b.check_similarity('Hepatocellular', 'carcinomas')) # output: 0.987503077756682
    # print(b.check_similarity('Hepatocellular', 'cancer')) # output: 0.26379474095785277
    # print(b.check_similarity('Hepatocellular', 'activated')) # output: -0.008187987591463093
    # print(b.check_similarity('aaa', 'activated', ignore_missing=False)) # output: -0.008187987591463093
