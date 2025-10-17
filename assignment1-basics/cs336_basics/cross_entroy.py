import torch
import math


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    vocab_size = inputs.shape[-1]
    inputs = inputs.view(-1, vocab_size)
    batch = inputs.shape[0]
    targets = targets.view(-1)
    o_max, _ = torch.max(inputs, dim=-1)
    o_i = inputs[torch.arange(0, batch), targets]
    loss = o_max - o_i + torch.log(torch.sum(torch.exp(inputs - o_max.view(batch, 1)), dim=-1))
    return torch.mean(loss, dim=0)



if __name__ == "__main__":
    x = torch.randn(4, 256, 1000)
    targets = torch.randint(0, 999, size=(4, 256))
    print(targets.shape)
    cross_entropy(x, targets)


