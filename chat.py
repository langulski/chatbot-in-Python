import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('frases.json','r') as f:
    frases = json.load(f)


FILE = "data.pth"
data= torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

todas_palavras = data["todas_palavras"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)

model.load_state_dict(model_state)

model.eval()

bot_name =  "Helo"

print("Olá vamos conversar! digite 'sair' para sair do chat")
while True:
    sentence = input('Você:')
    if sentence == "sair":
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, todas_palavras)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)

    output= model(X)
    _, predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() >0.75:
        for frase in frases["frases"]:
            if tag == frase["tag"]:
                print(f"{bot_name}: {random.choice(frase['responses'])}")

    else:
        print(f"{bot_name}: Eu nao entendi o que vc falou...")
