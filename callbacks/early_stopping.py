import torch
import warnings
from dataclasses import dataclass, field
from .callback import Callback


@dataclass
class EarlyStopping(Callback):
    """
    Implementation of Early Stopping. Early Stopping stops training when monitored value has stopped improving during some time. 
    
    Inputs:
        patience: int - number of attempts to beat 'best_value'. Default: 5.
        best_value: float - the best value. Default: inf if 'mode' is 'min' otherwise -inf.
        stopping_threshold: float - the threshold, when crossing it returns 'True'. Default: None.
        check_gradient_exploding: bool - checks for the problem Gradient Vanishing if True. Default: False.
        logger: bool - logs messages if True. Default: True.
    
    """
    
    patience: int = field(default=5, metadata={"help": ""})
    best_value: float = field(default="default", metadata={"help": ""})
    stopping_threshold: float = field(default=None, metadata={"help": ""})
    check_gradient_exploding: bool = field(default=False, metadata={"help": "Checks for Gradient Exploding."})
    fails: int = field(default=0, metadata={"help": "Number of fails."})
    logger: bool = field(default=True, metadata={"help": ""})
    
    def __post_init__(self) -> None:
        if not (0 < self.patience):
            raise ValueError(f"'patience' parameter should be in range (0, +inf), but given '{self.patience}'.")
         
        if not isinstance(self.logger, bool):
            if not callable(self.logger):
                raise TypeError(f"'logger' must be callable.")
        else:
            self.logger = print if self.logger else False
            
                
        if self.best_value == "default":
            self._set_default_best_value()
        else:
            if not self.ignore_warnings:
                warnings.warn(f"When setting personal value for 'best_value', you should be very carefully to avoid very early stopping.")
                
        self.is_stopped = False
        self.training = False
        self.step = None
        self.case = None
    
    def state_dict(self) -> dict:
        state = {
            "patience": self.patience,
            "best_value": self.best_value,
            "check_gradient_exploding": self.check_gradient_exploding,
            "stopping_threshold": self.stopping_threshold,
            "fails": self.fails,
            "delta": self.delta,
            "mode": self.mode,
            "is_stopped": self.is_stopped,
            "training": self.training,
            "case": self.case,
        }
        
        if self.step is not None:
            state["step"] = self.step
        
        return state
    
    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.best_value = state_dict["best_value"]
        self.check_gradient_exploding = state_dict["check_gradient_exploding"]
        self.stopping_threshold = state_dict["stopping_threshold"]
        self.fails = state_dict["fails"]
        self.delta = state_dict["delta"]
        self.mode = state_dict["mode"]
        self.is_stopped = state_dict["is_stopped"]
        self.training = state_dict["training"]
        self.case = state_dict["case"]
        
        if "step" in state_dict:
            self.step = state_dict["step"]
        
        return self
    
    
    def __call__(self, value:torch.Tensor, step=None, training=False) -> bool:
        """
        Inputs:
            value: torch.Tensor - the value for controling Early Stopping.
            step: int - step, where was checking for Early Stopping. If 'step' is provided, it will be displayed in the state.
            training: bool - checks only for Gradient Exploding, useful for training if True. Default: False.
            
        Outputs:
            stop: bool - controls stopping, if True the training process should be stopped.
            
        """
        
        if not isinstance(value, torch.Tensor):
            raise TypeError("'value' must be torch.Tensor.")
        
        delta_value = self._get_delta_value(value)
        stop = False
        case = None
        
        if not torch.isfinite(value):
            if self.check_gradient_exploding:
                stop = True
                case = f"The value is not finite, maybe problem of Gradient Exploding."
            else:
                if not self.ignore_warnings:
                    warnings.warn(f"Detected Gradient Exploding the training should be stopped.")
        
        if not training and not stop:
            if self.stopping_threshold is not None:
                if self._is_better(best_value=delta_value, value=self.stopping_threshold):
                    stop = True
                    case = f"The value is not better than 'stopping_threshold'."
            else:
                if not self._is_better(best_value=delta_value, value=self.best_value):
                    improvement_delta =  abs(value - self.best_value)
                    case = f"'best_value' is improved by {improvement_delta}! New 'best_value': {value}."
                    self.best_value = value
                    self.fails = 0
                else:
                    self.fails += 1
                    if self.fails >= self.patience:
                        stop = True
                        case = f"Number of attempts is expired."
        
        if stop: 
            self.case = case
        
        if self.logger and case is not None:
            self.logger(case)
            
        self.is_stopped = stop
        self.step = step
        self.training = training
        
        return stop