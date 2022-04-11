import torch
from torch import nn, optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any, Union, Optional
import warnings
import random
import os
from .import_utils import is_transformers_available


if is_transformers_available():
    import transformers


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
    """
    Returns optimizer's leearning rates for each group.
    """

    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"The given `optimizer` type is not supported, it must be instance of Optimizer.")
    
    lrs = [param_group["lr"] for param_group in optimizer.param_groups]
    return lrs[-1] if only_last else lrs


def to_tensor(input:Any) -> torch.Tensor:
    """
    Converts input to torch.Tensor.
    """

    if not isinstance(input, torch.Tensor):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        else:
            input = torch.tensor(input)
        
    return input


def to_tensors(*inputs:Any) -> tuple:
    """
    Converts all inputs to torch.Tensor.
    """
    return tuple(map(lambda input: to_tensor(input), inputs))



def load_checkpoint(path:str, 
                    model:nn.Module, 
                    optimizer:Optional[Optimizer]=None, 
                    scheduler:Optional[_LRScheduler]=None, 
                    strict:bool=True, 
                    ignore_warnings:bool=False, 
                    custom_keys:Optional["dict[str, str]"]=None) -> dict:

        """
        Loads checkpoint and then load state for model, optimizer or scheduler, if they are set. 

        Inputs:
            path: str - checkpoint's path.
            model: nn.Module - PyTorch's module.
            optimizer: Optional[Optimizer] - PyTorch's or HuggingFace Transformers's optimizer. Default: None.
            scheduler: Optional[_LRScheduler] - PyTorch's or HuggingFace Transformers's scheduler. Default: None.
            strict: bool - whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict() function. Default: True
            ignore_warnings: bool - if True the further warnings will be ignored. Default: False.
            custom_keys: Optional["dict[str, str]"] - sets keys for the checkpoint.

        Outputs:
            checkpoint: dict - loaded checkpoint.
            
        """

        if custom_keys is None:
            custom_keys = dict(model="model_state", 
                        optimizer="optimizer_state",
                        scheduler="scheduler_state")

        checkpoint = torch.load(path) if torch.cuda.is_available() else torch.load(path, map_location=torch.device("cpu"))
        
        model_key = custom_keys.get("model", "model_state")
        model_state = checkpoint[model_key]
        model.load_state_dict(model_state, strict=strict)

        if optimizer is not None:
            optimizer_key = custom_keys.get("optimizer", "optimizer_state")
            optimizer_state = checkpoint[optimizer_key]
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state, strict=strict)
            else:
                warnings.warn(f"`optimizer` was set, but checkpoint has not optimizer's state, hence it will be ignored.")
        

        if scheduler is not None:
            scheduler_key = custom_keys.get("scheduler", "scheduler_state")
            scheduler_state = checkpoint[scheduler_key]
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state, strict=strict)
            else:
                if not ignore_warnings:
                    warnings.warn(f"`scheduler` was set, but checkpoint has not scheduler's state, hence it will be ignored.")


        return checkpoint


def get_random_sample(dataset:Dataset) -> Any:
    index = random.randint(0, len(dataset)-1)
    sample = dataset[index]
    return sample


def get_batch(loader:DataLoader) -> Any:
    batch = next(iter(loader))
    return batch


def get_scheduler(name:str, parameters:Any, optimizer:_LRScheduler) -> _LRScheduler:
    """
    Returns scheduler with given name and parameters. 
    If failed to import from PyTorch, the function will try to import from HuggingFace Transformers library (if available).

    Raises:
        AttributeError - raised when given scheduler is not available/provided.
    """

    try:
        instance = getattr(lr_scheduler, name)
    except AttributeError as exception:
        if is_transformers_available():
            instance = getattr(transformers, name)
        else:
            raise AttributeError(f"Given scheduler's name is not provided.")
 
    scheduler = instance(optimizer=optimizer, **parameters)
    return scheduler


def get_optimizer(name:str, parameters:Any, model_parameters:Any) -> Optimizer:
    """
    Returns optimizer with given name and parameters. 
    If failed to import from PyTorch, the function will try to import from HuggingFace Transformers library.

    Raises:
        AttributeError - raised when given optimizer is not available/provided.
    """

    try:
        instance = getattr(optim, name)
    except AttributeError as exception:
        if is_transformers_available():
            instance = getattr(transformers, name)
        else:
            raise AttributeError(f"Given optimizer's name is not provided.")

    optimizer = instance(params=model_parameters, **parameters)
    return optimizer