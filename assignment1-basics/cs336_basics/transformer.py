from torch import nn
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .norm import RMSNorm
from .linear import Linear
from .softmax import softmax
import torch
from .rope import RoPE

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, rope_theta: float, max_seq_len: int):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.rope = RoPE(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=self.rope) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.lm_head(self.ln_final(x))
        return x
        


if __name__ == "__main__":
    vocab_size = 50257
    d_model = 1600
    num_layers = 48
    num_heads = 25
    d_ff = 6400
    max_seq_len = 1024
    model = Transformer(vocab_size=vocab_size, d_model=d_model,
                        num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, 
                        rope_theta=10000, max_seq_len=max_seq_len)
    # for param in model.parameters():
    #     print(param.shape, param.numel())
    for name, param in model.named_modules():
        print(name)
    for name, param in model.named_buffers():
        print(name)
    params_num = sum(param.numel() for param in model.parameters())
    trainable_num = sum(param.numel() for param in model.parameters() if param.requires_grad)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_buffers = sum(b.numel() for b in model.buffers())
    num_sum = num_params + num_buffers
    print("all:", num_sum / 1000 / 1000 / 1000, "billion params")
    print("trainable:", trainable_num / 1000 / 1000, "million params")
    print("all:", num_sum * 4 / 1024 ** 3, "GB")