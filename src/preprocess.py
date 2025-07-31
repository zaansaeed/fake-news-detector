import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
nltk.download('punkt_tab')


def load_dataset_two(path_of_real, path_of_fake):
    df_real = pd.read_csv(path_of_real)
    df_real["label"] = 1  # Real news labeled as 1
    df_fake = pd.read_csv(path_of_fake)  
    df_fake["label"] = 0  # Fake news labeled as 0
    
    df = pd.concat([df_real, df_fake], ignore_index=True)
    df =  df[["title", "label"]]
    df.rename(columns={"title": "tokens"}, inplace=True)

    return df
    

def load_dataset_one(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = [
        "ID","label", "statement", "subject", "speaker", "job", "state",
        "party", "barely_true", "false", "half_true", "mostly_true",
        "pants_on_fire", "context"
    ]
    df = df[["statement", "label"]]
    
    # Map string labels to integers
    label_map = {
        "pants-fire": 0,
        "false": 1,
        "barely-true": 2,
        "half-true": 3,
        "mostly-true": 4,
        "true": 5
    }
    
    df["label"] = df["label"].map(label_map)
    df["tokens"] = df["statement"].apply(clean_and_tokenize)

    return df


def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    return tokens


def build_vocabulary(token_list,min_freq=2):
    counter =Counter()
    for tokens in token_list:
        counter.update(tokens) 

    #unique words with frequency >= min_freq
    vocab = {word for word, freq in counter.items() if freq >= min_freq}

    word2idx = {
    "<PAD>": 0,
    "<UNK>": 1
    }

    # Assign indices to words in the vocabulary
    for idx, word in enumerate(vocab, start=2):
        word2idx[word] = idx
    # switch keys and values
    # to create idx2word mapping
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def encode_and_pad(token_list,word2idx,max_length=200):
    encoded = []
    #token_list is a list of lists, where each inner list contains tokens for a statement
    #word2idx is a dictionary mapping words to their indices
    for tokens in token_list:
        # Encode tokens to indices, using <UNK> for unknown words
        # and <PAD> for padding
        encoded_tokens = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
        if len(encoded_tokens) < max_length:
            encoded_tokens += [word2idx["<PAD>"]] * (max_length - len(encoded_tokens))
        else:
            encoded_tokens = encoded_tokens[:max_length]
        encoded.append(encoded_tokens)

    return encoded


def load_glove_embeddings(glove_path, word2idx, embed_dim):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embed_dim))

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embed_dim + 1:
                continue  # skip malformed lines

            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue  # skip bad float conversion

            if word in word2idx:
                idx = word2idx[word]
                embeddings[idx] = vec

    return torch.tensor(embeddings, dtype=torch.float32)

import torch
from torch.utils.data import Dataset, DataLoader

# This class is used to create a PyTorch dataset for the fake news detection task
class FakeNewsDataset(Dataset):
    def __init__(self, encoded_sequences, labels):
        self.sequences = encoded_sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert to torch tensors
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # or float for binary
        return x, y
    
