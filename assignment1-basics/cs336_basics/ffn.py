import torch
from torch import nn
from torch.nn import init
import math
from .linear import Linear

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff = None, align_to=64, device=None, dtype=None):
        super().__init__()
        if not d_ff:
            hidden_dim = int(d_model * 8 / 3)
            hidden_dim += -hidden_dim % align_to
        hidden_dim = d_ff if d_ff else hidden_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.w1 = Linear(in_features=d_model, out_features=hidden_dim, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=hidden_dim, device=device, dtype=dtype)
        self.w2 = Linear(in_features=hidden_dim, out_features=d_model, device=device, dtype=dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * nn.functional.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((self.silu(self.w1(x))) * (self.w3(x)))

    def reset_parameters(self):
        self.init_linear(self.w1)
        self.init_linear(self.w2)
        self.init_linear(self.w3)
        

    def init_linear(self, x):
        std = math.sqrt(2 / (self.hidden_dim + self.d_model))
        init.trunc_normal_(x, mean=0, std=std, a = -3 * std, b = 3 * std)

if __name__ == "__main__":
    d_model = 64
    ffn = SwiGLUFFN(d_model=d_model)
    x = torch.randn(16, 100, d_model)
    y = ffn(x)
    print(y.shape)