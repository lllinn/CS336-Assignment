from torch import nn
import torch
from .norm import RMSNorm
from .multihead_selfattention import MultiHeadSelfAttentionWithRoPE
from .ffn import SwiGLUFFN
from .rope import RoPE

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE):
        super().__init__()
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model=d_model, num_heads=num_heads, rope=rope)
        self.ln1 = RMSNorm(d_model=d_model)
        
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        token_positions = torch.arange(0, S).view(1, S).repeat(B, 1).to(x.device)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
