import torch
from torch import nn


def softmax(x: torch.Tensor, dim: int):
    v_max, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - v_max
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)

if __name__ == "__main__":
    x = torch.randn(4, 16, 32)
    y = softmax(x, -1)
    print(y.shape)


