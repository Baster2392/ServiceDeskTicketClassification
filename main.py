from collections import Counter
import nltk, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall
from classifier import TicketClassifier

nltk.download('punkt')

# Import data and labels
with open("data/words.json", 'r') as f1:
    words = json.load(f1)
with open("data/text.json", 'r') as f2:
    text = json.load(f2)
labels = np.load('data/labels.npy')

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o: i for i, o in enumerate(words)}
idx2word = {i: o for i, o in enumerate(words)}

# Looking up the mapping dictionary and assigning the index to the respective words
for i, sentence in enumerate(text):
    text[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]


# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


text = pad_input(text, 50)

# Splitting dataset
train_text, test_text, train_label, test_label = train_test_split(text, labels, test_size=0.2, random_state=42)

# Create data loaders
train_data = TensorDataset(torch.from_numpy(train_text), torch.from_numpy(train_label).long())
test_data = TensorDataset(torch.from_numpy(test_text), torch.from_numpy(test_label).long())

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# Train
model = TicketClassifier(len(word2idx) + 1, 64)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
model.train()
for epoch in range(30):
    total_loss = 0.0
    print(f"Running {epoch + 1} epoch...")
    for batch, labels in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} ended with total loss = {total_loss}")


# Evaluate model
acc = Accuracy(task="multiclass", num_classes=5)
prec = Precision(task="multiclass", num_classes=5, average=None)
rec = Recall(task="multiclass", num_classes=5, average=None)

model.eval()
print("Evaluating...")
predicted = []
with torch.no_grad():
    for batch, labels in test_loader:
        out = model(batch)
        cat = torch.argmax(out, dim=-1)
        print("Batch:", cat, labels)
        predicted.extend(cat.tolist())
        acc.update(cat, labels)
        prec.update(cat, labels)
        rec.update(cat, labels)

accuracy = acc.compute().item()
precision = prec.compute().tolist()
recall = rec.compute().tolist()

print("Model evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

print(predicted)
