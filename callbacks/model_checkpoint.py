import torch
import warnings
import shutil
from dataclasses import dataclass, field
import os
from .callback import Callback
import numpy as np



@dataclass
class ModelCheckpoint(Callback):
    directory: str = field(default="./checkpoints")
    overwriting: bool = field(default=False)
    best_value: float = field(default="default")
    filename_format: str = field(default="best_checkpoint.pth")
    candidates: int = field(default="all")
    
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
                        self.__remove_files_from_directory(self.directory)
                    else:
                        string_possible_checkpoints = "', '".join(possible_checkpoints)
                        if not self.ignore_warnings:
                            warnings.warn(f"Found files: {string_possible_checkpoints} in '{self.directory}'"
                                          "Be carefully with setting 'filename_format' to avoid overwritting other files.")
            else:
                raise NotADirectoryError(f"'{self.directory}' is not directory.")
        
        
        if isinstance(self.candidates, str):
            if self.candidates != "all":
                raise ValueError(f"'candidates' can be a string, but only with 1 value: 'all', but given '{self.candidates}'")
        elif self.candidates < 0:
            if not self.ignore_warnings:
                print(f"Parameter 'candidates' is lower than 0, so it will be setted to 0 (no saving checkpoints during training).")
            self.candidates = 0
            
        elif self.candidates == 0:
            if not self.ignore_warnings:
                print(f"Parameter 'candidates' was setted to '0', which means that no saving checkpoints during training.")
        
        if self.filename_format.count(".") > 1:
            raise ValueError(f"'filename_format' must not has '.' in filename, but given '{self.filename_format}'.")
        
        if self.is_filename_format_unique(self.filename_format):
            if not self.ignore_warnings:
                warnings.warn(f"Seems that 'filename_format' is not unique, maybe will be overwrited some useful checkpoints during training.")
            
            if self.candidates > 1:
                if not self.ignore_warnings:
                    warnings.warn(f"When 'filename_format' is not unique, number of candidates does not affect anything.")
        
        
        if self.best_value == "default":
            self._set_default_best_value()
        else:
            if not self.ignore_warnings:
                warnings.warn(f"When setting personal value for 'best_value', you should be very carefully to avoid very early stopping.")
        
        self.candidates_info = np.array([])
        self.best_checkpoint_path = None
        self.step = None
        self.best_step = None
    
    
    def is_filename_format_unique(self, format_):
        return not ("{value}" in format_ or "{step}" in format_)
    
    
    def __remove_files_from_directory(self, directory):
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
    
    
    def append_candidate(self, path=None, value=None, step=None):
        if (path is not None and value is None) or (value is not None and path is None):
            raise ValueError(f"'path' and 'value' must be not 'None' at the same time.")
        
        if path is None and value is None:
            state = self.state_dict()
            path = state["best_checkpoint_path"]
            value = state["best_value"]
            step = state["step"]
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Candidate's path '{path}' does not exist.")
        
        candidate = {
            "path": path,
            "value": value,
            "step": step,
        }
        
        self.candidates_info = list(self.candidates_info)
        self.candidates_info.append(candidate)
        self.candidates_info = np.array(self.candidates_info)
    
    
    def __filter_candidates(self):
        values = [candidate["value"].item() for candidate in self.candidates_info]
        sorted_indexes = np.argsort(values)
        
        if self.mode == "min":
            sorted_indexes = sorted_indexes[::-1]
        
        
        self.candidates_info = self.candidates_info[sorted_indexes]

    def __select_candidates(self):
        self.__filter_candidates()
        
        if self.candidates != "all":
            candidates_to_remove = self.candidates_info[:self.candidates]
            
            removed_pathes = []
            removed_count = 0
            for candidate in candidates_to_remove:
                path = candidate["path"]
                if os.path.exists(path):
                    os.remove(path)

                removed_pathes.append(path)
                removed_count += 1

            if removed_count > 0:
                removed_pathes = "', '".join(removed_pathes)
                print(f"Removed {removed_count} excesses candidates: {removed_pathes}.")
                    
        
    def _format_filename(self, value, step=None):
        value = value.item()
        value = str(value).replace(".", "")
        
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
        if self._is_better(value=delta_value, best_value=self.best_value) and self.candidates != 0:
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
            
            self.append_candidate(value=value, path=checkpoint_path, step=step)
            
            self.best_value = value
            self.best_step = step
            self.best_checkpoint_path = checkpoint_path
            
            is_saved = True
            
            self.__select_candidates()
        
        self.step = step
        
        return is_saved