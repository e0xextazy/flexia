from typing import Union, Callable
from datetime import timedelta, datetime
import warnings
from pytimeparse.timeparse import timeparse
from .callback import Callback


class Timer(Callback):
    """
    Timer stops training when the duration of the training stage reaches a certain limit of time. 
    It is useful when you are using time-limit sources, e.g. Google Colab or Kaggle Kernels/Notebooks.
    
    Inputs:
        duration: Union[str, timedelta] - duration of time after reaching whom, the training should be stopped. Default: '01:00:00:00'.
        duration_separator: str - seperator for input duration's format. Default: ':'.
        ignore_warnings: bool - if True the further warnings will be ignored. Default: False.
        logger: Callable[[str], str] - logging method. Default: print.

    Examples:
        >>> example
    
    """
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
        
    def state_dict(self) -> dict:
        state = {
            "start": self.start,
            "duration": self.duration,
            "remaining_time": self.remaining_time,
            "elapsed_time": self.elapsed_time,
            "is_stopped": self.is_stopped,
            "duration_separator": self.duration_separator,
        }
        
        return state
    
    
    def load_state_dict(self, state_dict:dict):
        self.start = state_dict["start"]
        self.duration = state_dict["duration"]
        self.remaining_time = state_dict["remaining_time"]
        self.elapsed_time = state_dict["elapsed_time"]
        self.duration_separator = state_dict["duration_separator"]
        self.is_stopped = state_dict["is_stopped"]
        
        return self
        
    def update(self) -> bool:
        """
        Outputs:
            is_stopped: bool - True if duration time is reached.
        """
        
        now = datetime.now()
        self.elapsed_time = abs(now - self.start)
        self.remaining_time = timedelta(seconds=max(0, (self.duration - self.elapsed_time).total_seconds()))
        self.is_stopped = self.elapsed_time > self.duration
        
        if self.is_stopped:
            self.logger(f"Reached duration limit. Elapsed: {self.elapsed_time}.")
        
        return self.is_stopped
    
    def __str__(self):
        return f"Timer(duration='{self.duration}', ignore_warnings={self.ignore_warnings}, duration_separator='{self.duration_separator}')"
    
    __repr__ = __str__