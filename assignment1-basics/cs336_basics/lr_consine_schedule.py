import math

def lr_consine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, consine_cycle_iters: int):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= consine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (consine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

