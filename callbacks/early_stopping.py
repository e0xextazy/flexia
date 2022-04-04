import torch
from dataclasses import dataclass, field
import logging
import warnings
from .callback import Callback




@dataclass
class EarlyStopping(Callback):
    """
    Inputs:
        patience: int - number of attempts to beat 'best_value'. Default: 5.
        best_value: float - the best value. Default: inf if 'mode' is 'min' otherwise -inf.
        stopping_threshold: float - the threshold, when crossing it returns 'True'. Default: None.
        check_gradient_vanishing: bool - checks for the problem Gradient Vanishing if True. Default: False.
        logger: bool - logs messages if True. Default: True.
    
    """
    
    patience: int = field(default=5)
    best_value: float = field(default=float("inf"))
    stopping_threshold: float = field(default=None)
    check_gradient_vanishing: bool = field(default=False)
    fails: int = field(default=0)
    logger: bool = field(default=True)
    
    def __post_init__(self) -> None:
        if not (0 < self.patience):
            raise ValueError(f"'patience' parameter should be in range (0, +inf), but given '{self.patience}'.")
         
        if not isinstance(self.logger, logging.Logger):
            if self.logger:
                self.logger = logging.getLogger(__name__)
        
        self._set_default_best_value()
        
    
    def state_dict(self) -> dict:
        state = {
            "patience": self.patience,
            "best_value": self.best_value,
            "check_gradient_vanishing": self.check_gradient_vanishing,
            "stopping_threshold": self.stopping_threshold,
            "fails": self.fails,
            "eps": self.eps,
            "mode": self.mode,
            "is_stopped": self.is_stopped,
        }
        
        if self.step is not None:
            state["stopping_step"] = self.stopping_step
        
        return state
    
    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.best_value = state_dict["best_value"]
        self.check_gradient_vanishing = state_dict["check_gradient_vanishing"]
        self.stopping_threshold = state_dict["stopping_threshold"]
        self.fails = state_dict["fails"]
        self.eps = state_dict["eps"]
        self.mode = state_dict["mode"]
        self.is_stopped = state_dict["is_stopped"]
        
        if "stopping_step" in state_dict:
            self.stopping_step = state_dict["stopping_step"]
        
        return self
    
    def __call__(self, value:torch.Tensor, step=None) -> bool:
        """
        Inputs:
            value: torch.Tensor - the value for controling Early Stopping.
            step: int - step, where was checking for Early Stopping. If 'step' is provided, it will be displayed in the state.
            
        Outputs:
            stop: bool - controls stopping, if True the training process should be stopped.
            
        """
        
        if not isinstance(value, torch.Tensor):
            raise "'value' must be torch.Tensor."
        
        eps_value = self._get_eps_value(value)
        stop = False
        if not torch.isfinite(value):
            if self.check_gradient_vanishing:
                stop = True
                message = f"The value is not finite, Gradient Vanishing."
            else:
                if not self.ignore_warnings:
                    warnings.warn(f"Detected Gradient Vanishing the training should be stopped.")
        elif self.stopping_threshold is not None:
            if self._is_better(best_value=eps_value, value=self.stopping_threshold):
                stop = True
                message = f"The value reaches the stopping threshold."
            else:
                message = f"The value doesn't reach the stopping threshold."
        else:
            if not self._is_better(best_value=eps_value, value=self.best_value):
                improvement_delta =  abs(value - self.best_value)
                message = f"'best_value' is improved by {improvement_delta}. New 'best_value': {value}."
                self.best_value = value
                self.fails = 0
            else:
                message = f"Attempt is failed to beat the 'best_value'."
                self.fails += 1
                if self.fails >= self.patience:
                    stop = True
                    message = f"Number of attempts is expired."
        
        if stop: 
            self.stopping_step = step
        
        if self.logger:
            self.logger.warning(message)
            
        self.is_stopped = stop
        
        return stop