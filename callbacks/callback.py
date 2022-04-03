from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field


@dataclass
class Callback(metaclass=ABCMeta):
    eps: float = field(default=0.0)
    mode: str = field(default="min")
    ignore_warnings: bool = field(default=False)
    verbose: bool = field(default=False)
    
    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError(f"'mode' parameter shoud be 'min' or 'max', but given '{self.mode}'.")
        
        assert (0 <= self.eps), f"'eps' parameter should be in range [0, +inf), but given '{self.eps}'." 
    
    
    @abstractmethod
    def __call__(self):
        pass
    
    def get_eps_value(self, value):
        eps_value = (value - self.eps) if self.mode == "max" else (value + self.eps)
        return float(eps_value)
    
    def is_better(self, value):
        condition = self.best_value < value if self.mode == "max" else self.best_value > value      
        return condition