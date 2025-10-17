import torch
from torch import nn
from torch.nn import init
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size=(out_features, in_features), dtype=dtype, device=device), requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    
    def reset_parameters(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        init.trunc_normal_(self.weight, mean=0, std=std, a = -3 * std, b = 3 * std)

    def flops(self, input_shape: torch.Tensor):
        # input_shape: [batch, seq, dim]
        return math.prod(input_shape) * self.out_features * 2
        


if __name__ == "__main__":
    in_features = 12
    out_features = 24
    linear = Linear(in_features, out_features)
    x = torch.randn(12, 213, 123, in_features)
    y = linear(x)
    print(y.shape)
    print(linear.flops((12,213,123,in_features)))

