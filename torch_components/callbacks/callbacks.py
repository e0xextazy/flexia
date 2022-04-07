from dataclasses import dataclass, field
import inspect
from callback import Callback
import warnings


@dataclass
class Callbacks:
    callbacks: list = field(default_factory=list)
    ignore_warnings: bool = field(default=False)
        
    def __post_init__(self):
        filtered_callbacks = []
        for callback in self.callbacks:
            if Callback not in callback.__class__.__bases__ and not inspect.isclass(callback):
                if self.ignore_warnings:
                    warnings.warn(f"Object '{callback}' isn't callback, so it will be ignored.") 
            else:
                filtered_callbacks.append(callback)
                
        self.callbacks = filtered_callbacks
        self.__callbacks_names = [callback.__class__ for callback in self.callbacks]
                
    def __getitem__(self, index):
        if inspect.isclass(index):
            index = self.__callbacks_names.index(index)
            callback = self.callbacks[index]  
        else:
            callback = self.callbacks[index]
                
        return callback