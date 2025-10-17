from torch import nn
import torch
from torch.nn import init
import math
from einops import rearrange
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import RoPE
from .linear import Linear

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.q_proj = Linear(in_features=d_model, out_features=d_model)
        self.k_proj = Linear(in_features=d_model, out_features=d_model)
        self.v_proj = Linear(in_features=d_model, out_features=d_model)
        self.output_proj = Linear(in_features=d_model, out_features=d_model)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        qk_num = Q.shape[-2]
        mask = torch.triu(torch.ones(qk_num, qk_num), diagonal=1) == 0
        mask = mask.to(x.device)
        multi_q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        multi_k = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        multi_v = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)
        output = scaled_dot_product_attention(multi_q, multi_k, multi_v, mask)
        output = rearrange(output,  "... h s d -> ... s (h d)")
        return self.output_proj(output)


    def reset_parameters(self):
        std = math.sqrt(2 / (self.d_model * 2))
        init.trunc_normal_(self.q_proj, mean=0, std=std, a = -3 * std, b = 3 * std)
        init.trunc_normal_(self.k_proj, mean=0, std=std, a = -3 * std, b = 3 * std)
        init.trunc_normal_(self.v_proj, mean=0, std=std, a = -3 * std, b = 3 * std)
        init.trunc_normal_(self.output_proj, mean=0, std=std, a = -3 * std, b = 3 * std)


class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE,**kwargs):
        super().__init__(d_model=d_model, num_heads=num_heads, **kwargs)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        qk_num = Q.shape[-2]
        mask = torch.triu(torch.ones(qk_num, qk_num), diagonal=1) == 0
        mask = mask.to(x.device)
        multi_q = rearrange(Q, "... s (h d) -> h ... s d", h=self.num_heads)
        multi_k = rearrange(K, "... s (h d) -> h ... s d", h=self.num_heads)
        multi_v = rearrange(V, "... s (h d) -> h ... s d", h=self.num_heads)
        multi_q_rope = self.rope.forward(multi_q, token_positions)
        multi_k_rope = self.rope.forward(multi_k, token_positions)
        output = scaled_dot_product_attention(multi_q_rope, multi_k_rope, multi_v, mask)
        output = rearrange(output,  "h ... s d -> ... s (h d)")
        return self.output_proj(output)


if __name__ == "__main__":
    # mask = torch.triu(torch.ones(5, 5), diagonal=1) == 0
    # print(mask)
    d_model = 256
    num_heads = 4
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn((4, 100, d_model))
    y = mha(x)
    # print(x.shape, y.shape)
