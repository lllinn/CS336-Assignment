import torch
from torch import nn
from torch.nn import init
import math

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype), requires_grad=True)
        self.reset_parameters()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

    def reset_parameters(self):
        init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)



if __name__ == "__main__":
    vocab_size = 10000
    embedding_size = 256
    embedding = Embedding(vocab_size, embedding_size)
    x = torch.randint(low=0, high=vocab_size, size=(16, 500)).to(torch.long)
    y = embedding(x)
    print(y.shape)
