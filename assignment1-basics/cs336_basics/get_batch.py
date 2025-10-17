import torch
import numpy as np
import numpy.typing as npt

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_input = []
    batch_target = []
    start_indices = np.random.randint(0, len(dataset) - context_length - 1, size=(batch_size,))

    for i in start_indices:
        input_sequence = torch.tensor(dataset[i : i + context_length], device=device, dtype=torch.long).view(1, -1)
        target_sequence = torch.tensor(dataset[i + 1 : i + 1 + context_length], device=device, dtype=torch.long).view(1, -1)
        batch_input.append(input_sequence)
        batch_target.append(target_sequence)
    return torch.concat(batch_input, dim=0), torch.concat(batch_target, dim=0)
 



if __name__ == "__main__":
    batch_input = []
    for _ in range(4):
        batch_input.append(torch.Tensor(torch.arange(start=0, end=10, dtype=torch.long)).view(1, -1))
    batch_tensor = torch.concat(batch_input, dim=-0)
    print(batch_tensor.shape)