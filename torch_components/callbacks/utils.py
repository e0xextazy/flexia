# torch_components.callbacks.utils

import warnings
import torch
from typing import Union
from torch_components.callbacks.constants import modes_options
from ..utils import  to_tensors


def get_mode_values(mode:str="min"):
    if mode not in modes_options:
        modes = list(modes_options.keys())
        raise ValueError(f"`{mode}` is not valid, choose one of {modes}.")
    else:
        mode_values = modes_options[mode]
        return mode_values


def get_delta_value(value:Union[torch.Tensor, float, int], delta:Union[torch.Tensor, float, int]=0.0, mode:str="min") -> torch.Tensor:
    value, delta = to_tensors(value, delta)
    
    *_, delta_operation = get_mode_values(mode)
    return delta_operation(value, delta)
    
    
def compare_values(value:Union[torch.Tensor, float, int], other:Union[torch.Tensor, float, int], mode:str="min") -> bool:    
    value, other = to_tensors(value, other)
    
    *_, compare_operation, _ = get_mode_values(mode)
    return compare_operation(value, other)