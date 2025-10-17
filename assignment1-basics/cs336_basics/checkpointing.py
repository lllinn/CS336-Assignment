import torch
import typing
import os

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str) -> None:
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": iteration}
    torch.save(checkpoint, out)


def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    ckpt = torch.load(src)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['iter']


