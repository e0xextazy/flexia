import torch
from typing import Union
from .constants import modes_options


def is_mode_valid(mode:str) -> bool:
    return mode in modes_options
    
    
def get_delta_value(value:torch.Tensor, delta:Union[float, int]=0.0, mode:str="min") -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    
    if not is_mode_valid(mode):
        raise ValueError(f"Given mode is not provided, possible modes: 'min' or 'max'.")
    
    default_best_value, compare_operation, delta_operation = modes_options[mode]
    delta_value = delta_operation(value, delta)
    return delta_value
    
    
def get_default_best_value(mode:str) -> torch.Tensor:
    if not is_mode_valid(mode):
        raise ValueError(f"Given mode is not provided, possible modes: 'min' or 'max'.")
    
    default_best_value, compare_operation, delta_operation = modes_options[mode]
    return default_best_value
    
    
def compare_values(value:torch.Tensor, other:torch.Tensor, mode:str="min") -> bool:
    if not is_mode_valid(mode):
        raise ValueError(f"Given mode is not provided, possible modes: 'min' or 'max'.")
    
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)
    
    default_best_value, compare_operation, delta_operation = modes_options[mode]
    result = compare_operation(value, other)
    
    return result