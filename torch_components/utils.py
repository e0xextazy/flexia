import torch
from torch import Optimizer
import numpy as np
from typing import Any, Union, Optional
import random
import os


def get_random_seed(min_value:int=0, max_value:int=50) -> int:
    return random.randint(min_value, max_value)

def seed_everything(seed:Optional[int]=None) -> int:
    if seed is None:
        seed = get_random_seed()
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PC_SEED'] = str(seed)
    
    return seed
    

def get_lr(optimizer:Optimizer, only_last:bool=False) -> Union[int, list]:
    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"The given `optimizer` type is not supported, it must be instance of Optimizer.")
    
    lrs = [param_group["lr"] for param_group in optimizer.param_groups]
    return lrs[-1] if only_last else lrs


def to_tensor(input:Any):
    if not isinstance(input, torch.Tensor):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        else:
            input = torch.tensor(input)
        
    return input


def to_tensors(*inputs:Any):
    return tuple(map(lambda input: to_tensor(input), inputs))