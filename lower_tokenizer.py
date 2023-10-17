import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')


# This is a function that will tokenize a sentence into lowercase words
def lower_tokenizer(sentence: str):
    tokens = word_tokenize(sentence)
    lower_tokens = []
    for w in tokens:
        lower_tokens.append(w.lower())
    return lower_tokens



# This is a function that will remove all stopwords in a list of words
def sp_remover(arr: list):
    stop_words = set(stopwords.words('english'))
    arr = [w for w in arr if not w in stop_words]
    return arr



# This is a function that will stem all words in a list of words
def word_stemmer(arr: list):
    porter = PorterStemmer()
    stems = []
    for t in arr:    
        stems.append(porter.stem(t))
    return stems



# Examples:
ex = "THIS IS WHAT IT IS GOING TO BE"
tokens = lower_tokenizer(ex) # tokens: ['this', 'is', 'what', 'it', 'is', 'going', 'to', 'be']
after_removal = sp_remover(tokens) # after_removal: ['going']
stems = word_stemmer(after_removal) # stems: ['go']
print(tokens)
print(after_removal)
print(stems)