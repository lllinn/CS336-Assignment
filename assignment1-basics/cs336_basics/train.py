import numpy as np
import yaml
import argparse
from .transformer import Transformer
from .cross_entroy import cross_entropy
from .adamw import AdamW
from .lr_consine_schedule import lr_consine_schedule
from functools import partial
from tqdm import tqdm
from .get_batch import get_batch
from .gradient_clipping import gradient_clipping
import torch
from .checkpointing import save_checkpoint, load_checkpoint
import os
import numpy.typing as npt
import wandb
from datetime import datetime
import math


def get_memmap(filepath: str, dtype=np.int32):
    return np.memmap(filepath, dtype=dtype, mode='r')


def get_config(filepath: str = "config.yaml"):
    with open(filepath, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_val_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_input = []
    batch_target = []
    for i in range(0, len(dataset) - context_length, context_length):
        input_sequence = torch.tensor(dataset[i : i + context_length], device=device, dtype=torch.long).view(1, -1)
        target_sequence = torch.tensor(dataset[i + 1 : i + 1 + context_length], device=device, dtype=torch.long).view(1, -1)
        batch_input.append(input_sequence)
        batch_target.append(target_sequence)
        if len(batch_input) == batch_size:
            yield torch.concat(batch_input, dim=0), torch.concat(batch_target, dim=0)
            batch_input = []
            batch_target = []

    if len(batch_input) > 0:
        yield torch.concat(batch_input, dim=0), torch.concat(batch_target, dim=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 配置文件路径参数
    parser.add_argument('--config', type=str, default="CS336-Assignment/assignment1-basics/cs336_basics/config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    config = get_config(args.config)
    train_config = config['Train']
    device = train_config['device']

    # WandB
    wandb_config = config['WandB']
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + wandb_config['name']
    wandb.login()
    wandb.init(project=wandb_config['project'], config=config, name=nowtime, notes=wandb_config['notes'])
    
    # 数据集
    dataset_config = config['Dataset']
    train_path = dataset_config['train_path']
    val_path = dataset_config['val_path']
    train_set = get_memmap(train_path)
    valid_set = get_memmap(val_path)
    train_dataloader = partial(get_batch, dataset=train_set, 
                               batch_size=train_config['batch_size'], context_length=train_config['context_length'], device=device)
    val_dataloader = partial(get_val_batch, dataset=valid_set, 
                               batch_size=train_config['batch_size'], context_length=train_config['context_length'], device=device, )
    
    # model
    model = Transformer(**config["Model"]).to(device)
    # criterion
    criterion = cross_entropy
    # optimizer
    optim_config = config["Optimizer"]
    optim = AdamW(params=model.parameters(), lr=float(optim_config['lr']), betas=tuple(optim_config['betas']), 
                  weight_decay=float(optim_config['weight_decay']), eps=float(optim_config['eps']))
    
    # scheduler
    scheduler_config = config['Scheduler']
    scheduler = partial(lr_consine_schedule, max_learning_rate=float(scheduler_config['max_learning_rate']),
                        min_learning_rate=float(scheduler_config['min_learning_rate']), warmup_iters=scheduler_config['warmup_iters'],
                        consine_cycle_iters=scheduler_config['consine_cycle_iters'])
    
    
    test_config = config['Test']
    os.makedirs(test_config['save_checkpoint_folder'], exist_ok=True)
    start_it = 0
    if test_config['resume']:
        start_it = load_checkpoint(test_config['resume_checkpoint_path'], model, optim)
        
    # 使用tqdm上下文管理器
    pbar = tqdm(range(start_it, train_config['steps']), desc="training llm", unit="it", ncols=150)
    
    min_loss = float('+inf')
    
    for i in pbar:
        # train
        x, labels = train_dataloader()
        y = model(x)
        loss = criterion(y, labels)
        optim.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), **train_config['grad_clip'])
        lr = scheduler(i)
        # 更新lr
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        optim.step()
        # 更新进度条信息
        pbar.set_postfix(loss=loss.item())


        if (i + 1) % train_config['log_period'] == 0: #  TODO: 后面测试下是否是这个瓶颈
            wandb.log({'step': i + 1, 'train_loss': loss.item(), 'lr': lr})


        if (i + 1) % test_config['save_checkpoint_period'] == 0:
            # 保存最近的模型
            last_path = os.path.join(test_config['save_checkpoint_folder'], "last.ckpt")
            save_checkpoint(model, optim, i + 1, last_path)
            wandb.save(last_path)
            
        if (i + 1) % test_config['valid_period'] == 0:
            with torch.no_grad():
                model.eval()
                loss_sum = 0
                loss_num = 0
                test_pbar = tqdm(val_dataloader(), desc="valid llm", unit="it", ncols=80, leave=False, total=math.ceil((len(valid_set) - train_config['context_length']) // (train_config['batch_size'] * train_config['context_length'])))
                for x, labels in test_pbar:
                    y = model(x)
                    loss_sum += criterion(y, labels).item()
                    loss_num += 1
                loss = loss_sum / loss_num
                if loss <= min_loss:
                    min_loss = loss
                    # 保存最好的模型
                    save_path = os.path.join(test_config['save_checkpoint_folder'], "best.ckpt")
                    save_checkpoint(model, optim, i, save_path)
                    # wandb.save(save_path)
                    wandb.log({'val_step': i + 1, 'val_loss': loss, "min_loss": min_loss})
                model.train()
    wandb.finish()
        