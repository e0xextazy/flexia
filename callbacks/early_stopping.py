from dataclasses import dataclass, field
import numpy as np
from .callback import Callback


@dataclass
class EarlyStopping(Callback):
    patience: int = field(default=5)
    past_value: float = field(default=np.inf)
    stopping_threshold: float = field(default=None)
    __fails: int = field(default=0)
    
    def __post_init__(self):
        assert 0 < self.patience, f"'patience' parameter should be in range (0, +inf), but given '{self.patience}'."
        self.past_value = 0 if self.mode == "max" else np.inf
    
    
    def is_worse_than_past(self, value):
        condition = (self.past_value < value) if self.mode == "max" else (self.past_value > value)
        return condition
    
    def check_threshold_condition(self, value):
        condition = (self.stopping_threshold >= value) if self.mode == "max" else (self.stopping_threshold <= value)
        return condition
    
    def __call__(self, value):
        breaking = False
        breaking = self.check_threshold_condition(value)
        
        eps_value = self.get_eps_value(value)
        condition = self.is_worse_than_past(eps_value)
        
        if not condition:
            self.__fails += 1
            if self.__fails >= self.patience:
                breaking = True
        else:
            self.__fails = 0
            
        self.past_value = value
        
        return breaking