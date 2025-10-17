import torch
import math
from .softmax import softmax
from einops import einsum

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    atten_score = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(Q.shape[-1])
    if mask is not None:
        atten_score.masked_fill_(~mask, -float('inf'))
    return einsum(softmax(atten_score, -1), V, "... q k, ... k d -> ... q d")

