import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,words):
    
    """
    frase = ["Oi","Como","está",você"]
    palavras = ["oi","ola","Eu","vc","tchal","obrigado","legal"]
    bog = [0, 1, 0, 1, 0, 0, 0]

    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag


# frase = ["Oi","Como","está","você"]
# palavras = ["oi","ola","Eu","vc","tchal","obrigado","legal"]

# bag = bag_of_words(frase, palavras)
# print(bag)

# a = "how long does shipping take"
# print(a)
# a= tokenize(a)
# print(a)

#conhecido como caule das palavras
# words = ["organize","organizes","organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

