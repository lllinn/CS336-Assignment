import torch
from .softmax import softmax



def top_p(x: torch.Tensor, p: float) -> torch.Tensor:
    # shape: (vocab_size, )
    index = torch.argsort(x, descending=True)
    cum_sum = 0
    for i in index:
        if cum_sum < p:
            cum_sum += x[i]
        else:
            x[i] = 0
            

def random_sample(x: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(x, 1, replacement=False)


def generate_text(model: torch.nn.Module, promote: torch.Tensor, max_output_len: int, end_of_text: int, temperature: int=1, p: float=0.9) -> torch.Tensor:
    end_of_text = torch.Tensor([end_of_text])
    while promote < max_output_len:
        predict = model(promote)[-1]
        predict = softmax(predict / temperature)
        top_p(predict, p)
        out = random_sample(predict)
        promote = torch.concat([promote, out], dim=0)
        if out == end_of_text:
            break
    return promote
    


