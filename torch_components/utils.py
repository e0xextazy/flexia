import os
import random
import torch
import numpy as np
import transformers as T
from typing import Any



def make_directory(directory):
    if os.path.exists(directory):
        os.mkdir(directory)

        return True
    
    return False


def seed_everything(seed:int=42) -> None:
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    return seed



def get_optimizer(name, parameters, model_parameters):
    try:
        instance = getattr(torch.optim, name)
    except:
        instance = getattr(T, name)

    optimizer = instance(params=model_parameters, **parameters)

    return optimizer


def get_scheduler(name, parameters, optimizer):
    try:
        instance = getattr(torch.optim.lr_scheduler, name)
    except:
        instance = getattr(T, name)
    
    scheduler = instance(optimizer=optimizer, **parameters)
    return scheduler



def to_tensor(input:Any):
    if not isinstance(input, torch.Tensor):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        else:
            input = torch.tensor(input)
        
    return input


def to_tensors(*inputs:Any):
    return tuple(map(lambda input: to_tensor(input), inputs))