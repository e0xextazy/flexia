from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field
import torch


@dataclass
class Callback(metaclass=ABCMeta):
    eps: float = field(default=0.0)
    mode: str = field(default="min")
    ignore_warnings: bool = field(default=False)
    verbose: bool = field(default=False)
    

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError(f"'mode' parameter shoud be 'min' or 'max', but given '{self.mode}'.")
        
        if not (0 <= self.eps):
            raise ValueError(f"'eps' parameter should be in range [0, +inf), but given '{self.eps}'.")
        
        self._set_default_best_value()
    
    
    def state_dict(self) -> dict:
        state = {
            "eps": self.eps,
            "mode": self.mode,
        }
        
        return state
        
    def load_state_dict(self, state_dict):
        self.eps = state_dict["eps"]
        self.mode = state_dict["mode"]
        
        return self
    
    @abstractmethod
    def __call__(self):
        pass
    
    def _get_eps_value(self, value:float) -> float:
        """
        Gets 'value' in limits of 'eps' value relatively on 'mode'.
        
        """
        eps_value = (value - self.eps) if self.mode == "max" else (value + self.eps)
        return float(eps_value)
    
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
