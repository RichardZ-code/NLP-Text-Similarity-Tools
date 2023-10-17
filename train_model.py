from Corpus import *
from GloVe import *
from lower_tokenizer import lower_tokenizer

nltk.download('punkt')
nltk.download('stopwords')

prepared_texts = None
with open('traindata.txt', 'r', encoding='utf-8') as f:
  texts = f.read()
  prepared_texts = prepare_data(texts, tokenizer = lower_tokenizer)

mycorpus = Corpus.load_file('saved-dict')
mycorpus.fit_matrix(prepared_texts)
mycorpus.save_file('saved-dict')

myglove = GloVe()
myglove.fit_vectors(mycorpus.matrix)
myglove.save_file('new_model')