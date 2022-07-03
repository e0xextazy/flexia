import torch
from typing import Callable, Union, Optional

from ..trainer.trainer_enums import TrainingStates
from .callback import Callback
from .utils import compare_values, get_delta_value, get_mode_values
from ..utils import to_tensor


class EarlyStopping(Callback):   
    def __init__(self, 
                 monitor_value="validation_loss",
                 mode:str="min", 
                 delta:Union[float, int]=0.0, 
                 patience:Union[float, int]=5, 
                 stopping_threshold:Optional[float]=None, 
                 check_finite:bool=False):
        
        super().__init__()
        
        self.monitor_value = monitor_value
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.stopping_threshold = stopping_threshold
        self.check_finite = check_finite
        
        self.is_stop = False
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


    def on_validation_end(self, trainer):
        value = trainer.history.get(self.monitor_value)
        if value is None:
            value = trainer.history["validation_metrics"].get(self.monitor_value)

        if value is None:
            raise KeyError(f"Disable to find `{self.monitor_value}` in Trainer history.")
        else:
            self.is_stop = self.check(value=value, trainer=trainer)

            if self.is_stop:
                trainer.state = TrainingStates.TRAINING_STOP

    
    def check(self, value:Union[torch.Tensor, float, int], trainer) -> bool:       
        value = to_tensor(value)
        
        delta_value = get_delta_value(value=value, delta=self.delta, mode=self.mode)
        
        if not self.is_stop:
            if not torch.isfinite(value):
                if self.check_finite:
                    self.is_stop = True
                    self.case = f"The value is not finite, maybe problem of Gradient Exploding."

            if self.stopping_threshold is not None:
                if compare_values(value=self.stopping_threshold, other=delta_value, mode=self.mode):
                    self.is_stop = True
                    self.case = f"Monitored value reached `stopping_threshold`. Value: {self.value}. Stopping threshold: {self.stopping_threshold}."
            else:
                if compare_values(value=delta_value, other=self.best_value, mode=self.mode):
                    improvement_delta = abs(value - self.best_value)
                    self.case = f"Moniroted value is improved by {improvement_delta}! New `best_value`: {value}."
                    self.best_value = value
                    self.fails = 0
                else:
                    self.fails += 1
                    if self.fails >= self.patience:
                        self.is_stop = True
                        self.case = f"Number of attempts has been expired. The best monitored value wasn't beaten." 
        
        return self.is_stop