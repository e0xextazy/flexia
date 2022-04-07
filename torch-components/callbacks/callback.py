from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field, asdict
import numpy as np
import os
import inspect
import logging
import torch
            

@dataclass
class Callback(metaclass=ABCMeta):
    delta: float = field(default=0.0, metadata={"help": "The small value with which the monitored value must be improved. Default: 0.0."})
    mode: str = field(default="min", metadata={"help": "Directs in which side the monitored value must improving. Default: 'min'."})
    ignore_warnings: bool = field(default=False, metadata={"help": "Ignores warnings if True. Default: False."})
    

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError(f"'mode' parameter shoud be 'min' or 'max', but given '{self.mode}'.")
        
        if not (0 <= self.delta):
            raise ValueError(f"'delta' parameter should be in range [0, +inf), but given '{self.eps}'.")
        
        self._set_default_best_value()
    
    
    def state_dict(self) -> dict:
        state = {
            "delta": self.delta,
            "mode": self.mode,
        }
        
        return state
        
    def load_state_dict(self, state_dict):
        self.delta = state_dict["delta"]
        self.mode = state_dict["mode"]
        
        return self
    
    @abstractmethod
    def __call__(self):
        pass
    
    def _get_delta_value(self, value:float) -> float:
        """
        Gets 'value' in limits of 'delta' value relatively on 'mode'.
        """
        delta_value = (value - self.delta) if self.mode == "max" else (value + self.delta)
        return float(delta_value)
    
    def _is_better(self, value, best_value):
        """
        Compares 'value' and 'best_value' relatively on 'mode'. 
        """
        
        condition = (best_value < value) if self.mode == "max" else (best_value > value)   
        return condition
    
    def _set_default_best_value(self):
        """
        Sets the minimum value for the 'best_value' relatively on 'mode'.
        """
        
        best_value = torch.tensor(float("-inf" if self.mode == "max" else "inf"))
        self.best_value = best_value