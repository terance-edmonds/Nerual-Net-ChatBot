import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "trained_model_data.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data["tags"]
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print("Nural Net ChatBot. Type 'quit' to exit.")

while True:
    sentense = input('Text: ')

    if(sentense == 'quit'):
        break

    sentense = tokenizer(sentense)
    X = bag_of_words(sentense, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if(prob > 0.75):
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print("response:", random.choice(intent["responses"]))
    else:
        print("response: I don't understand...")

