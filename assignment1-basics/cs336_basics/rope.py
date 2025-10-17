import torch
from torch import nn



class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, **kwargs):
        super().__init__(**kwargs)
        self.k_list = torch.linspace(1, d_k // 2, steps=d_k // 2, device=device)
        self.seq_list = torch.linspace(0, max_seq_len - 1, steps=max_seq_len, device=device)
        self.k_list = self.k_list.view(1, d_k // 2)
        self.seq_list = self.seq_list.view(max_seq_len, 1)
        self.theta_k = torch.pow(theta, (2 - 2 * self.k_list) / d_k)
        self.theta_k_i = self.seq_list * self.theta_k
        self.register_buffer("cos", torch.cos(self.theta_k_i), persistent=False) # [seq, d / 2]
        self.register_buffer("sin", torch.sin(self.theta_k_i), persistent=False) # [seq, d / 2]
        self.max_seq_len = max_seq_len
        self.d_k = d_k

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotaty = x.view(*x.shape[:-1], self.d_k // 2, 2)
        rotaty_x = rotaty[..., 0]
        rotaty_y = rotaty[..., 1]
        rotaty = torch.stack([
            rotaty_x * self.cos[token_positions] - rotaty_y * self.sin[token_positions],
            rotaty_x * self.sin[token_positions] + rotaty_y * self.cos[token_positions]
        ], dim=-1)
        return rotaty.view_as(x)

if __name__ == "__main__":
    theta = 1e4
    d_k = 256
    max_seq_len = 100
    rope = RoPE(theta=theta, d_k=d_k, max_seq_len=100)
    x = torch.randn(4, max_seq_len, d_k)
    y = rope(x)
    print(y.shape)