import warnings
import torch
from typing import Callable, Union, Optional
from .callback import Callback
from utils import is_mode_valid, get_default_best_value, get_delta_value, compare_values


class EarlyStopping(Callback):  
    """
    Class description
    
    Inputs:
        mode: str - description.
        delta: float - description.
        patience: int - description.
        stopping_threshold: float - description.
        check_finite: bool - description.
        logger: callable - description.
        ignore_warnings: bool - description.
    
    Errors:
        ValueError - description.
        
    Examples:
        >>> example
    
    """
    
    
    def __init__(self, 
                 mode:str="min", 
                 delta:Union[float, int]=0.0, 
                 patience:Union[float, int]=5, 
                 stopping_threshold:Optional[float]=None, 
                 check_finite:bool=False, 
                 logger:Callable[[str], str]=print, 
                 ignore_warnings:bool=False):
        
        super().__init__()
        
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.stopping_threshold = stopping_threshold
        self.check_finite = check_finite
        self.logger = logger
        self.ignore_warnings = ignore_warnings
        
        self.is_stopped = False
        self.training = False
        self.step = None
        self.best_step = None
        self.case = None
        self.fails = 0
        
        if not is_mode_valid(self.mode):
            raise ValueError(f"Given mode ('{self.mode}') is not provided, possible modes: 'min' or 'max'.")
        
        if self.patience < 0:
            raise ValueError(f"Patience parameter must be integer and be in range (0, +inf), but given {self.patience}.")
        else:
            if not isinstance(self.patience, int):
                if not self.ignore_warnings:
                    warnings.warn(f"Seems that given patience parameter is not integer, so it will be rounded.")
                
                self.patience = round(self.patience)
        
        self.best_value = get_default_best_value(self.mode)
        
    
    def state_dict(self) -> dict:
        state = {
            "patience": self.patience,
            "best_value": self.best_value.item(),
            "check_finite": self.check_finite,
            "stopping_threshold": self.stopping_threshold,
            "fails": self.fails,
            "delta": self.delta,
            "mode": self.mode,
            "is_stopped": self.is_stopped,
            "training": self.training,
            "case": self.case,
            "step": self.step,
            "best_step": self.best_step,
        }
    
        return state
    
    
    def load_state_dict(self, state_dict:dict):
        self.patience = state_dict["patience"]
        self.best_value = torch.tensor(state_dict["best_value"])
        self.check_finite = state_dict["check_finite"]
        self.stopping_threshold = state_dict["stopping_threshold"]
        self.fails = state_dict["fails"]
        self.delta = state_dict["delta"]
        self.mode = state_dict["mode"]
        self.best_step = state_dict["best_step"]
        self.is_stopped = state_dict["is_stopped"]
        self.training = state_dict["training"]
        self.case = state_dict["case"]
        self.step = state_dict["step"]
        
        return self
    
    
    def __call__(self, value:torch.Tensor, step:int=None, training:bool=False) -> bool:
        """
        Inputs:
            value: torch.Tensor - description.
            step: int - description.
            training: bool - description.
            
        Outputs:
            is_stopped: bool - description.
        
        """
        
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float)
        
        case = None
        delta_value = get_delta_value(value=value, delta=self.delta, mode=self.mode)        
        if not torch.isfinite(value):
            if self.check_finite:
                self.is_stopped = True
                case = f"The value is not finite, maybe problem of Gradient Exploding."
            else:
                if not self.ignore_warnings:
                    warnings.warn(f"Input value is infinite the training should be stopped.")
        
        if not training and not self.is_stopped:
            if self.stopping_threshold is not None:
                if compare_values(value=self.stopping_threshold, other=delta_value, mode=self.mode):
                    self.is_stopped = True
                    case = f"The value is not better than 'stopping_threshold'."
            else:
                if compare_values(value=delta_value, other=self.best_value, mode=self.mode):
                    improvement_delta = abs(value - self.best_value)
                    case = f"'best_value' is improved by {improvement_delta}! New 'best_value': {value}."
                    self.best_value = value
                    self.best_step = step
                    self.fails = 0
                else:
                    self.fails += 1
                    if self.fails >= self.patience:
                        self.is_stopped = True
                        case = f"Number of attempts is expired."
        
        if self.logger and case is not None:
            self.logger(case)
        
        self.case = case        
        self.step = step
        self.training = training
        
        return self.is_stopped
    
    
    def __str__(self):
        return f"EarlyStopping(mode='{self.mode}', delta={self.delta}, patience={self.patience}, stopping_threshold={self.stopping_threshold}, check_finite={self.check_finite}, fails={self.fails}, ignore_warnings={self.ignore_warnings})"
    
    __repr__ = __str__