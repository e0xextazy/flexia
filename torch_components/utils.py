from pickle import BINBYTES
import torch
from torch import nn, optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any, Union, Optional
from enum import Enum
import warnings
import random
import os
from tqdm import tqdm


from .import_utils import is_transformers_available, is_bitsandbytes_available
from .exceptions import LibraryException




if is_transformers_available():
    import transformers

if is_bitsandbytes_available():
    import bitsandbytes as bnb


class ExplicitEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        keys = list(cls._value2member_map_.keys())
        raise ValueError(f"`{value}` is not a valid `{cls.__name__}`, select one of `{keys}`.")



def get_random_seed(min_value:int=0, max_value:int=50) -> int:
    """
    Returns random value from [`min_value`, `max_value`] range.
    """
    
    return random.randint(min_value, max_value)

def seed_everything(seed:Optional[int]=None) -> int:
    """
    Sets seed for `torch`, `numpy` and `random` libraries to have opportunity to reproduce results.
    """
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
    

def get_lr(optimizer:Optimizer, only_last:bool=False, key="lr") -> Union[int, list]:
    """
    Returns optimizer's leearning rates for each group.
    """

    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"The given `optimizer` type is not supported, it must be instance of Optimizer.")
    
    lrs = []
    for param_group in optimizer.param_groups:
        if key not in param_group:
            key = "lr"
        
        lr = param_group[key]
        lrs.append(lr)
        
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


def to_list(input:Any) -> list:
    if not isinstance(input, list):
        if isinstance(input, torch.Tensor):
            input = input.detach().to("cpu").tolist()
        elif isinstance(input, np.ndarray):
            input = input.tolist()
        else:
            try:
                input = list(input)
            except TypeError:
                input = list([input])
            
    return input


def to_lists(*inputs:Any) -> tuple:
    """
    Converts all inputs to torch.Tensor.
    """
    return tuple(map(lambda input: to_list(input), inputs))


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
    """
    Returns random sample from dataset.
    """

    index = random.randint(0, len(dataset)-1)
    sample = dataset[index]
    return sample


def get_batch(loader:DataLoader) -> Any:
    """
    Returns batch from loader.
    """

    batch = next(iter(loader))
    return batch


def __get_from_library(library, name, parameters, **kwargs):
    instance = getattr(library, name)
    instance = instance(**kwargs, **parameters)

    return instance


class SchedulerLibraries(ExplicitEnum):
    TRANSFORMERS = "transformers"
    TORCH = "torch"


def get_scheduler(optimizer:Optimizer, name:str="LinearLR", parameters:dict={}, library="torch") -> _LRScheduler:
    """
    Returns instance of scheduler.

    Inputs:
        name:str - name of scheduler, e.g ReduceLROnPlateau, CosineAnnealingWarmRestarts, get_cosine_schedule_with_warmup.
        parameters:dict - parameters of scheduler, e.g num_training_steps, T_mult, last_epoch. Default: {}.
        optimizer:Any - instance of optimizer to schedule the learning rate.
        library:str - library from which the scheduler will be used. Possible values: ["torch", "transformers"]. Default: "torch".
    
    Returns:
        scheduler:_LRScheduler - instance of scheduler.

    """


    library = SchedulerLibraries(library)

    if library == SchedulerLibraries.TORCH:
        scheduler = __get_from_library(library=lr_scheduler, 
                                       name=name, 
                                       parameters=parameters, 
                                       optimizer=optimizer)

    elif library == SchedulerLibraries.TRANSFORMERS:
        if is_transformers_available():
            scheduler = __get_from_library(library=transformers, 
                                           name=name, 
                                           parameters=parameters, 
                                           optimizer=optimizer)
        else:
            raise LibraryException("transformers")

    return scheduler


class OptimizerLibraries(ExplicitEnum):
    TRANSFORMERS = "transformers"
    TORCH = "torch"
    BITSANDBYTES = "bitsandbytes"


def get_optimizer(model_parameters:Any, name:str="AdamW", parameters:dict={}, library:str="torch") -> Optimizer:
    """
    Returns instance of optimizer.

    Inputs:
        name:str - name of optimizer, e.g AdamW, SGD, RMSprop.
        parameters:dict - parameters of optimizer, e.g lr, weight_decay. Default: {}.
        model_parameters:Any - model's parameters to optimize.
        library:str - library from which the optimizer will be used. Possible values: ["torch", "transformers", "bitsandbytes"]. Default: "torch".
    
    Returns:
        optimizer:Optimizer - instance of optimizer.

    """


    library = OptimizerLibraries(library)

    if library == OptimizerLibraries.TORCH:
        optimizer = __get_from_library(library=optim, 
                                       name=name, 
                                       parameters=parameters, 
                                       params=model_parameters)

    elif library == OptimizerLibraries.TRANSFORMERS:
        if is_transformers_available():
            optimizer = __get_from_library(library=transformers, 
                                           name=name, 
                                           parameters=parameters, 
                                           params=model_parameters)
        else:
            raise LibraryException("transformers")

    elif library == OptimizerLibraries.BITSANDBYTES:
        if is_bitsandbytes_available():
            optimizer = __get_from_library(library=bnb.optim, 
                                           name=name, 
                                           parameters=parameters, 
                                           params=model_parameters)
        else:
            raise LibraryException("bitsandbytes")

    return optimizer



def freeze(module:nn.Module) -> None:
    """
    Freezes module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False
        
        
def get_freezed_parameters(module:nn.Module) -> list:
    """
    Returns names of freezed parameters of the given module.
    """
    
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters



def tqdm_loader_wrapper(self, loader:DataLoader, description:str="", color="#000") -> Any:
    """
    Wraps loader into `tqdm` loop.

    Inputs:
        loader: DataLoader - loader to wrap.
        description: str - description for `tqdm` loop.
     """

    bar_format = "{l_bar} {bar} {n_fmt}/{total_fmt} - elapsed: {elapsed} - remain: {remaining}{postfix}"
    steps = len(loader)
    loader = tqdm(iterable=loader, 
                  total=steps,
                  colour=color,
                  bar_format=bar_format)

    loader.set_description_str(description)

    return loader