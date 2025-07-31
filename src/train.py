import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from preprocess import (
    load_dataset_two,
    build_vocabulary,
    encode_and_pad,
    FakeNewsDataset,
    load_glove_embeddings
)
from model import FakeNewsClassifier

import numpy as np
import random
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Data
MAX_LEN = 50
MIN_FREQ = 1
BATCH_SIZE = 32
DROP_OUT = 0.5
# Model
EMBED_DIM = 100
HIDDEN_DIM = 32
NUM_CLASSES = 2
EPOCHS = 5
NUM_LAYERS = 2
LEARNING_RATE = 1e-4

set_seed(42)
train_df = load_dataset_two("data/True.csv", "data/Fake.csv")
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,        # or 0.2 for 80/20 split
    stratify=train_df["label"],  # preserve class distribution
    random_state=42       # ensures reproducibility
)




word2idx, idx2word = build_vocabulary(train_df["tokens"])

glove_path = "data/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
glove_embeddings = load_glove_embeddings(glove_path, word2idx, embed_dim=EMBED_DIM)


train_sequences = encode_and_pad(train_df["tokens"], word2idx, max_length=MAX_LEN)
val_sequences = encode_and_pad(val_df["tokens"], word2idx, max_length=MAX_LEN)


train_dataset = FakeNewsDataset(train_sequences, train_df["label"].values)
val_dataset = FakeNewsDataset(val_sequences, val_df["label"].values)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FakeNewsClassifier(
    vocab_size=len(word2idx),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    padding_idx=word2idx["<PAD>"],
    num_layers=NUM_LAYERS,
    drop_out=DROP_OUT,
    pretrained_embeddings=glove_embeddings
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)


def compute_accuracy(preds, labels):
    predicted_classes = torch.argmax(preds, dim=1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    return correct / total


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []



for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_acc = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(outputs, labels)

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")

    # ----- Validation -----
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += compute_accuracy(outputs, labels)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"          → Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "config": {
        "vocab_size": len(word2idx),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "padding_idx": word2idx["<PAD>"],
        "num_layers": NUM_LAYERS,
        "drop_out": DROP_OUT,
        "pretrained_embeddings": glove_embeddings
    },
    "word2idx": word2idx  # optional but useful for inference
}, "models/fake_news_checkpoint.pt")



test_df = load_dataset_two("data/true_test.csv", "data/fake_test.csv")
test_sequences = encode_and_pad(test_df["tokens"], word2idx, max_length=MAX_LEN)



test_data = FakeNewsDataset(test_sequences, test_df["label"].values)


test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        preds  = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
print(f"Test accuracy: {correct/total:.2%}")


