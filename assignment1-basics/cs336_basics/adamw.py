import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta = group["betas"]
            beta1 = beta[0]
            beta2 = beta[1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', 0)
                v = state.get('v', 0)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                at = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= at * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        return loss

