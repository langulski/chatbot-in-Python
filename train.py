import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
with open('frases.json','r') as f:
    frases = json.load(f)

# print(frases)

todas_palavras = []
tags = []
xy = []

for frase in frases['frases']:
    tag=frase['tag']
    tags.append(tag)
    for pattern in frase['patterns']:
        w = tokenize(pattern)
        todas_palavras.extend(w)
        xy.append((w,tag))
ignore_words = ['?','!',',','.','$','é','ê','á']
todas_palavras = [stem(w) for w in todas_palavras if w not in ignore_words]
todas_palavras = sorted(set(todas_palavras))
tags= sorted(set(tags))

# print(tags)

X_train = []
y_train = []
for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence, todas_palavras)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # 1hot cross entropy loss

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChataDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data= X_train
        self.y_data= y_train
    #dataset index
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

    batch_size=8

    dataset = ChataDataset()
    train_loader = DataLoader(dataset=dataset,bach_size=batch_size,shuffle=True,num_workers=2)


