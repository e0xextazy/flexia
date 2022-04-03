import torch
from dataclasses import dataclass, field
import numpy as np
import os
from .callback import Callback
import warnings


@dataclass
class ModelCheckpoint(Callback):
    directory: str = field(default="./checkpoints")
    filename: str = field(default="epoch_{epoch}__{value_name}_{value}.pth")
    value_name: str = field(default="loss")
    best_value: float = field(default=np.inf)
    best_checkpoint_path: str = field(default=None)
    keys: set = field(default_factory=dict)
        
    
    def __post_init__(self):        
        if os.path.isdir(self.directory):
            if not os.path.exists(self.directory):
                raise FileNotFoundError(f"Directory '{self.directory}' does not exist.")
        else:
            raise NotADirectoryError(f"'{self.directory}' is not directory.")
        
        self.best_value = 0 if self.mode == "max" else np.inf
    
    
    def create_checkpoint(self, model, value, optimizer=None, epoch=None, step=None, scheduler=None):
        checkpoint = {}
        checkpoint[self.value_name] = value
        
        model_state_key = self.keys.get("model_state", "model_state")
        checkpoint[model_state_key] = model.state_dict()

        if optimizer is not None:
            optimizer_state_key = self.keys.get("optimizer_state", "optimizer_state")
            checkpoint[optimizer_state_key] = optimizer
                    
        if epoch is not None:
            epoch_key = self.keys.get("epoch", "epoch")
            checkpoint[epoch_key] = epoch
            
        if step is not None:
            step_key = self.keys.get("step", "step")
            checkpoint[step_key] = step
                
        if scheduler is not None:
            scheduler_key = self.__keys.get("scheduler_state", "scheduler_state")
            checkpoint[scheduler_key] = scheduler.state_dict()
            
        return checkpoint
        
    
        
    def __call__(self, model, value, optimizer=None, epoch=None, step=None, scheduler=None):
        if optimizer is None:
            if not self.ignore_warnings:
                warnings.warn("When saving checkpoint, better additional save the optimizer's state to continue training.")
        
        
        eps_value = self.get_eps_value(value)
        condition = self.is_better(eps_value)
        
        if condition:
            checkpoint_filename = self.filename.format(value=value, value_name=self.value_name, epoch=epoch)
            checkpoint_path = os.path.join(self.directory, checkpoint_filename)
            checkpoint = self.create_checkpoint(model=model, 
                                                value=value, 
                                                optimizer=optimizer, 
                                                epoch=epoch, 
                                                step=step, 
                                                scheduler=scheduler)
            
            torch.save(checkpoint, checkpoint_path)
            self.best_value = value
            self.best_checkpoint_path = checkpoint_path
            
            if self.verbose:
                print(f"Saved checkpoint: '{checkpoint_path}'.")
            
        return condition