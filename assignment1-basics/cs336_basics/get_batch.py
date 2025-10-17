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
    from functools import partial
    from .transformer import Transformer
    def get_memmap(filepath: str, dtype=np.int32):
        return np.memmap(filepath, dtype=dtype, mode='r')
    train_path = r"data\TinyStoriesV2-GPT4-train.dat"
    train_set = get_memmap(train_path)
    print(len(train_set))
    train_dataloader = partial(get_batch, dataset=train_set, 
                            batch_size=256, context_length=256, device="cuda")
    x = train_dataloader()
    print(x)
    print(type(x))
    x, label = train_dataloader()
    print(sum(x[:, 1:] != label[:, :-1]))
    model = Transformer(vocab_size=10000, d_model=512, num_heads=16,
                        d_ff=1344, rope_theta=10000, max_seq_len=256, num_layers=4).cuda()
    y = model(x)
    
    
    # for x, label in train_dataloader():
        # print(x, label)
        # exit()
    # batch_input = []
    # for _ in range(4):
    #     batch_input.append(torch.Tensor(torch.arange(start=0, end=10, dtype=torch.long)).view(1, -1))
    # batch_tensor = torch.concat(batch_input, dim=-0)
    # print(batch_tensor.shape)