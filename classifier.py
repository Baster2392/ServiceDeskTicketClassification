import torch
import torch.nn as nn
import torch.nn.functional as F


class TicketClassifier(nn.Module):
    def __init__(self, num_embeddings=50, embedding_dim=50, conv_out_channels=16):
        super(TicketClassifier, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, stride=1)
        self.linear = nn.Linear(in_features=64, out_features=5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x)).mean(dim=2)
        x = self.linear(x)
        return x