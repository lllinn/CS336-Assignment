import torch
import math
import matplotlib.pyplot as plt


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1
        return loss

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=1)
    loss_list_1 = []
    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        # print(loss.cpu().item())
        loss_list_1.append(loss.cpu().item())
        loss.backward()
        opt.step()
    weights = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=10)
    loss_list_10 = []
    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        # print(loss.cpu().item())
        loss_list_10.append(loss.cpu().item())
        loss.backward()
        opt.step()
    weights = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=100)
    loss_list_100 = []
    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        # print(loss.cpu().item())
        loss_list_100.append(loss.cpu().item())
        loss.backward()
        opt.step()
    plt.plot(loss_list_1, loss_list_10, loss_list_100)
    plt.savefig("cmp.jpg")
    plt.show()
