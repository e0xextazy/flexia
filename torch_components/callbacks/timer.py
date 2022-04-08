from typing import Union, Optional, Callable
from datetime import timedelta, datetime
from pytimeparse.timeparse import timeparse
from .callback import Callback


class Timer(Callback):
    """
    Class description
    
    Inputs:
        duration: Union[str, timedelta] - description.
        duration_separator: str - description.
        ignore_warnings: bool - description.
        logger: Callable[[str], str] - description.
    
    Errors:
        TypeError - description.
        
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
        now = datetime.now()
        self.elapsed_time = abs(now - self.start)
        self.remaining_time = timedelta(seconds=max(0, (self.duration - self.elapsed_time).total_seconds()))
        self.is_stopped = self.elapsed_time > self.duration
        
        if self.is_stopped:
            self.logger(f"Reached duration limit. Elapsed: {self.elapsed_time}.")
        
        return self.is_stopped
    
    def __str__(self):
        return f"Timer(duration='{self.duration}', total={self.total}, ignore_warnings={self.ignore_warnings}, duration_separator={self.duration_separator})"
    
    __repr__ = __str__