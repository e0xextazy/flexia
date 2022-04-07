import warnings
from callback import Callback
import torch


class EarlyStopping(Callback):
    modes = ("min", "max")
    
    def __init__(self, patience=5, mode="min", delta=0.0, stopping_threshold=None, check_gradient_exploding=False, fails=0, logger=print, ignore_warnings=False):
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.stopping_threshold = stopping_threshold
        self.check_gradient_exploding = check_gradient_exploding
        self.fails = fails
        self.logger = logger
        self.ignore_warnings = ignore_warnings
        
        self.is_stopped = False
        self.training = False
        self.step = None
        self.best_step = None
        self.case = None
        
        if self.mode not in self.modes:
            raise ValueError(f"'mode' parameter shoud be 'min' or 'max', but given '{self.mode}'.")
        
        if not (0 <= self.delta):
            raise ValueError(f"'delta' parameter should be in range [0, +inf), but given '{self.delta}'.")
        
        if not (0 < self.patience):
            raise ValueError(f"'patience' parameter should be in range (0, +inf), but given '{self.patience}'.")
        
        self._set_default_best_value()
            
    
    def state_dict(self) -> dict:
        state = {
            "patience": self.patience,
            "best_value": self.best_value.item(),
            "check_gradient_exploding": self.check_gradient_exploding,
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
    
    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.best_value = torch.tensor(state_dict["best_value"])
        self.check_gradient_exploding = state_dict["check_gradient_exploding"]
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
    
    
    def get_delta_value(self, value):
        delta_value = (value - self.delta) if self.mode == "max" else (value + self.delta)
        return delta_value
    
    
    def compare_values(self, value, other) -> bool:
        condition = (other < value) if self.mode == "max" else (other > value)   
        return condition
    
    
    def _set_default_best_value(self):
        infinity = torch.tensor(float("inf"))
        self.best_value = -infinity if self.mode == "max" else infinity
    
    
    def __call__(self, value, step=None, training=False) -> bool:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Input value must be instance of torch.Tensor.")
        
        delta_value = self.get_delta_value(value)        
        case = None
        if not torch.isfinite(value):
            if self.check_gradient_exploding:
                self.is_stopped = True
                case = f"The value is not finite, maybe problem of Gradient Exploding."
            else:
                if not self.ignore_warnings:
                    warnings.warn(f"Detected Gradient Exploding the training should be stopped.")
        
        if not training and not self.is_stopped:
            if self.stopping_threshold is not None:
                if self.compare_values(value=self.stopping_threshold, other=delta_value):
                    self.is_stopped = True
                    case = f"The value is not better than 'stopping_threshold'."
            else:
                if not self.compare_values(value=self.best_value, other=delta_value):
                    improvement_delta =  abs(value - self.best_value)
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
            self.case = case
            self.logger(self.case)
            
        self.step = step
        self.training = training
        
        return self.is_stopped
    
    
    def __str__(self):
        return f"EarlyStopping(mode='{self.mode}', delta={self.delta}, stopping_threshold={self.stopping_threshold}, check_gradient_exploding={self.check_gradient_exploding}, fails={self.fails}, ignore_warnings={self.ignore_warnings})"
    
    __repr__ = __str__