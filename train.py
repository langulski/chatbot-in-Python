import numpy as np
import random
import json

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem

from model import NeuralNet





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


class ChatDataset (Dataset):
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
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
# print(input_size,len(todas_palavras))
# print(output_size,tags)
dataset = ChatDataset()

num_epochs = 1000


    
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

#perda e otimizacao

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#treino
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        
        
        loss = criterion(outputs,labels)

        #volta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "todas_palavras":todas_palavras,
    "tags":tags

}

FILE = "data.pth"
torch.save(data,FILE)

print(f'treinamento completo file salvo em {FILE}')

