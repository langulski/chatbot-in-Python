import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenzied_sentence,todas_palavras):
    
    """
    frase = ["Oi","Como","está",você"]
    palavras = ["oi","ola","Eu","vc","tchal","obrigado","legal"]
    bog = [0, 1, 0, 1, 0, 0, 0]

    """
    tokenzied_sentence = [stem(w) for w in tokenzied_sentence]

    bag = np.zeros(len(todas_palavras),dtype=np.float32)
    for idx,w in enumerate(todas_palavras):
        if w in tokenzied_sentence:
            bag[idx]=1.0
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

