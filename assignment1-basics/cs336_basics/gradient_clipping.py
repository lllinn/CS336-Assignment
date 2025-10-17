import torch
from typing import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps = 1e-6) -> None:
    with torch.no_grad():
        # 计算全局gradient
        total_norm = torch.sqrt(sum(torch.sum(p.grad.data ** 2) for p in parameters if p.grad is not None))
        if total_norm > max_l2_norm:
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(max_l2_norm / (total_norm + eps))



if __name__ == "__main__":
    x = torch.nn.Parameter(torch.randn(5, 5), requires_grad=True)
