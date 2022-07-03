from typing import Union, Callable
from datetime import timedelta, datetime
import warnings


from .callback import Callback
from ..third_party.pytimeparse.timeparse import timeparse
from ..trainer.trainer_enums import TrainingStates


class Timing(Callback):
    def __init__(self, 
                 duration:Union[str, timedelta]="01:00:00:00", 
                 ignore_warnings:bool=False, 
                 duration_separator:str=":", 
                 logger:Callable[[str], str]=print):
        
        super().__init__()
        
        self.duration = duration
        self.ignore_warnings = ignore_warnings
        self.duration_separator = duration_separator
        self.logger = logger

        if isinstance(self.duration, str):
            try:
                duration_values = self.duration.strip().split(self.duration_separator)
                duration_values = tuple([int(value) for value in duration_values])
                days, hours, minutes, seconds = duration_values 
            except:
                if not self.ingore_warnings:
                    warnings.warn(f"Failed to parse the given duration format, trying to understand format with `timeparse` module.")
                    
                seconds = timeparse(self.duration)
                days, hours, minutes = 0, 0, 0
            
            self.duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        elif isinstance(self.duration, dict):
            self.duration = timedelta(**self.duration)
        elif not isinstance(self.duration, timedelta):
            raise TypeError(f"Type of given `duration` is not supported.")
            
        self.start = datetime.now()
        self.remaining_time = self.duration
        self.elapsed_time = timedelta()
        self.is_stopped = False
        

    def on_epoch_end(self, trainer):
        self.is_stopped = self.check()
        
        if self.is_stopped:
            trainer.state = TrainingStates.TRAINING_STOP
        
    def check(self) -> bool:       
        now = datetime.now()
        self.elapsed_time = abs(now - self.start)
        self.remaining_time = timedelta(seconds=max(0, (self.duration - self.elapsed_time).total_seconds()))
        self.is_stopped = self.elapsed_time > self.duration
        
        if self.is_stopped:
            self.logger(f"Reached duration limit. Elapsed: {self.elapsed_time}.")
        
        return self.is_stopped