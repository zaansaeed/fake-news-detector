import torch
import torch.nn as nn


class FakeNewsClassifier(nn.Module):
    def __init__(self,vocab_size, embed_dim,hidden_dim,num_classes,padding_idx,num_layers,drop_out,pretrained_embeddings=None):
        super(FakeNewsClassifier, self).__init__()
        #input as (32,200,200) where 32 is batch size, 200 is max length of sequence and 200 is the embedding dimension, ignore padding index
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=padding_idx)
        else:
            # Initialize embedding layer with random weights
            # vocab_size is the size of the vocabulary, embed_dim is the dimension of the embeddings
            self.embedding = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embed_dim,
                    padding_idx=padding_idx
                )     
               
        self.dropout = nn.Dropout(p=drop_out)  # 50% dropout

        # LSTM layer with bidirectional=True, which means it will have two hidden states for each time step
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True)        
        
        self.dropout = nn.Dropout(p=drop_out)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((h_n[-2], h_n[-1]), dim=1))
        logits = self.fc(hidden)
        return logits
