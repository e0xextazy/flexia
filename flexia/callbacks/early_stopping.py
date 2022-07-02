import warnings
import torch
from typing import Callable, Union, Optional


from .callback import Callback
from .utils import compare_values, get_delta_value, get_mode_values
from ..utils import to_tensor


class EarlyStopping(Callback):  
    """
    Early Stopping is a simple method of regularization used to avoid overfitting during 
    training Machine Learning models by stopping the training when the monitored value has not been improved.
    
    Inputs:
        mode: str - the direction where the monitored value must be improving. Default: 'min'.
        delta: Union[float, int] - small value on which the monitored value must be improved. It is useful for preventing very small improvements, e.g +1e-7. Default: 0.0.
        patience: Union[float, int] - a number of attempts to beat the best written monitored value, if the limit is reached, the Early Stopping returns True. Default: 5.
        stopping_threshold: Optional[float] - if it is set,  if the monitored value reaches it, the Early Stopping returns True. Default: None.
        check_finite: bool - if `check_finite`=True, the Early Stopping will check the input value for the infinite (NaN, Infinity, etc.), if the value is infinite the Early Stopping returns True. Default: False.
        logger: Callable[[str], str] - logging method. Default: print.
        ignore_warnings: bool - if True the further warnings will be ignored. Default: False.
    
    Examples:
        # Example of settings for monitoring Accuracy score. 
        # The Accuracy score must be improving on 1% (0.01) during 3 attempts, 
        # In addition, Accuracy score must be greater than 50% otherwise the training will be stopped. 

        >>> from torch_components.callbacks import EarlyStopping
        >>> callback = EarlyStopping(mode="max", 
                                     delta=0.01, 
                                     patience=3, 
                                     stopping_threshold=50, 
                                     check_finite=False, 
                                     ignore_warnings=False)
    
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
        
        self.best_value, *_ = get_mode_values(self.mode)
        
        if self.patience < 0:
            raise ValueError(f"`patience` must be in range (0, +inf), but given {self.patience}.")
        else:
            if not isinstance(self.patience, int):
                self.patience = round(self.patience)
                if not self.ignore_warnings:
                    warnings.warn(f"Seems that given `patience` value is float type, so it will be rounded. New `patience` value: {self.patience}.")
                
        
    
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
    
    
    def __call__(self, value:Union[torch.Tensor, float, int], step:Optional[int]=None, training:bool=False) -> bool:
        """
        Inputs:
            value: Union[torch.Tensor, float, int] - monitored value.
            step: Optional[int] - step of calling Early Stopping.
            training: bool - checks monitored value only for infinite value if True. It is useful for embedding on batch level of training to avoid further not meaningful training.
            
        Outputs:
            is_stopped: bool - if True it means, that training should be stopped.
        
        """
        
        value = to_tensor(value)
        
        case = None
        delta_value = get_delta_value(value=value, delta=self.delta, mode=self.mode)
        
        if not torch.isfinite(value):
            if self.check_finite:
                self.is_stopped = True
                case = f"The value is not finite, maybe problem of Gradient Exploding."
            else:
                if not self.ignore_warnings:
                    warnings.warn(f"Monitored value is infinite, maybe problem of Gradient Exploding, the training should be stopped to avoid further not meaningful training.")
        
        if not training and not self.is_stopped:
            if self.stopping_threshold is not None:
                if compare_values(value=self.stopping_threshold, other=delta_value, mode=self.mode):
                    self.is_stopped = True
                    case = f"Monitored value reached `stopping_threshold`. Value: {self.value}. Stopping threshold: {self.stopping_threshold}."
            else:
                if compare_values(value=delta_value, other=self.best_value, mode=self.mode):
                    improvement_delta = abs(value - self.best_value)
                    case = f"Moniroted value is improved by {improvement_delta}! New `best_value`: {value}."
                    self.best_value = value
                    self.best_step = step
                    self.fails = 0
                else:
                    self.fails += 1
                    if self.fails >= self.patience:
                        self.is_stopped = True
                        case = f"Number of attempts has been expired. The best monitored value wasn't beaten."
        
        if self.logger and case is not None:
            self.logger(case)
        
        self.case = case        
        self.step = step
        self.training = training
        
        return self.is_stopped