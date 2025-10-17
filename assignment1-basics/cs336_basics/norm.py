import torch
from torch import nn
from torch.nn import init

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(data=torch.empty(size=(d_model,), device=device, dtype=dtype), requires_grad=True)
        self.eps = eps
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x =  x.to(torch.float32)
        x = x / self.RMS(x) * self.weight

        return x.to(in_dtype)
        
    def reset_parameters(self):
        init.constant_(self.weight, 1)

    def RMS(self, x: torch.Tensor):
        return torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)

    # def flops(self, input_shape: torch.Tensor) -> int:
    #     flops = 
        


if __name__ == "__main__":
    d_model = 256
    x = torch.rand(size=(1,1,d_model)) * 4 + 2
    norm = RMSNorm(d_model)
    print("mean:", torch.mean(x, dim=-1), "std:", torch.std(x, dim=-1))
    x = norm(x)
    print("mean:", torch.mean(x, dim=-1), "std:", torch.std(x, dim=-1))
    