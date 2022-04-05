import torch
import warnings
import shutil
from dataclasses import dataclass, field
import os
import numpy as np
from .callback import Callback



@dataclass
class ModelCheckpoint(Callback):
    directory: str = field(default="./checkpoints")
    overwriting: bool = field(default=False)
    best_value: float = field(default="default")
    filename_format: str = field(default="best_checkpoint.pth")
    canditates: int = field(default=1)
    
    def __post_init__(self):        
        if not os.path.exists(self.directory):
            if self.overwriting:
                os.mkdir(self.directory)
            else:
                raise FileNotFoundError(f"Directory '{self.directory}' does not exist.")
                
        else:
            if os.path.isdir(self.directory):
                possible_checkpoints = os.listdir(self.directory)
                if len(possible_checkpoints) > 0:
                    if self.overwriting:
                        self._remove_files_from_directory(self.directory)
                    else:
                        string_possible_checkpoints = "', '".join(possible_checkpoints)
                        if not self.ignore_warnings:
                            warnings.warn(f"Found files: {string_possible_checkpoints} in '{self.directory}'"
                                          "Be carefully with setting 'filename_format' to avoid overwritting other files.")
            else:
                raise NotADirectoryError(f"'{self.directory}' is not directory.")
                        
        
        if self.best_value == "default":
            self._set_default_best_value()
        else:
            if not self.ignore_warnings:
                warnings.warn(f"When setting personal value for 'best_value', you should be very carefully to avoid very early stopping.")
        
        self.__candidates_info = []
        self.best_checkpoint_path = None
        self.step = None
        self.best_step = None
    
    
    def _remove_files_from_directory(self, directory):
        filenames = os.listdir(directory)
        pathes = [os.path.join(directory, filename) for filename in filenames]
        
        for path in pathes:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
                
                
    def state_dict(self):
        state = {
            "directory": self.directory,
            "best_checkpoint_path": self.best_checkpoint_path,
            "best_value": self.best_value,
            "filename_format": self.filename_format,
            "step": self.step,
            "best_step": self.best_step,
        }
        
        return state
    
    
    def load_state_dict(self, state_dict):
        self.directory = state_dict["directory"]
        self.best_checkpoint_path = state_dict["best_checkpoint_path"]
        self.best_value = state_dict["best_value"]
        self.filename_format = state_dict["filename_format"]
        self.step = state_dict["step"]
        self.best_step = state_dict["best_step"]
        
        return self
    
    
    def _create_candidate(self):
        state = self.state_dict()
        candidate = {
            "path": state["best_checkpoint_path"],
            "value": state["best_value"],
            "step": state["best_step"],
        }
        
        return candidate
    
    
    def _filter_candidates(self, candidates):
        values = [candidate["value"].item() for candidate in candidates]
        sorted_indexes = np.argsort(values)
        sorted_candidates = candidates[sorted_indexes]
        
        return sorted_candidates
    
    
    def _format_filename(self, value, step=None):
        value = value.item()
        
        if step is not None:
            filename = self.filename_format.format(value=value, step=step)
        else:
            filename = self.filename_format.format(value=value, step="")
            
        return filename
        
    
    def create_checkpoint(self, value, model, optimizer=None, scheduler=None, step=None):
        if optimizer is None:
            if not self.ignore_warnings:
                warnings.warn("When saving checkpoint, better additional save the optimizer's state to continue training.")
        
        checkpoint = {
            "value": value,
            "step": step,
            "model_state": model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state"] = scheduler.state_dict()
            
        return checkpoint
        
        
    def __call__(self, value, model, optimizer=None, scheduler=None, step=None):
        if not isinstance(value, torch.Tensor):
            raise TypeError("'value' must be torch.Tensor.")
        
        delta_value = self._get_delta_value(value)
        
        is_saved = False
        if self._is_better(value=delta_value, best_value=self.best_value):
            improvement_delta = abs(value - self.best_value)
            checkpoint_filename = self._format_filename(value=value, step=step)
            checkpoint_path = os.path.join(self.directory, checkpoint_filename)
            
            checkpoint = self.create_checkpoint(value=value, 
                                                model=model, 
                                                optimizer=optimizer, 
                                                scheduler=scheduler, 
                                                step=step)
            
            torch.save(checkpoint, checkpoint_path)
            
            print(f"'best_value' is improved by {improvement_delta}! New 'best_value': {value}. Checkpoint path: '{checkpoint_path}'.")
            
            self.best_value = value
            self.best_step = step
            self.best_checkpoint_path = checkpoint_path
            
            candidate = self._create_candidate()
            self.__candidates_info.append(candidate)
            
            is_saved = True
        
        
        self.step = step
        
        return is_saved