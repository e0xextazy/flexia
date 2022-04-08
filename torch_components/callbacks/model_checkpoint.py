import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import warnings
import shutil
import os
from typing import  Union, Optional, Callable
from .callback import Callback
from .utils import is_mode_valid, get_default_best_value, get_delta_value, compare_values


class ModelCheckpoint(Callback):  
    """
    Class description
    
    Inputs:
        mode: str - description.
        delta: Union[float, int] - description.
        directory: str - description.
        overwriting: bool - description.
        filename_format: str - description.
        candidates: Union[int, float, str] - description.
        ignore_warnings: bool - description.
    
    Errors:
        ValueError - description.
        NotADirectoryError - description.
        FileNotFoundError - description.
        
        
    Examples:
        >>> example
    
    """
    
    def __init__(self, 
                 mode:str="min", 
                 delta:Union[float, int]=0.0, 
                 directory:str="./", 
                 overwriting:bool=False, 
                 filename_format:str="checkpoint_{step}.pth", 
                 candidates:Union[str, float, int]="all", 
                 ignore_warnings:bool=False, 
                 logger:Callable[[str], str]=print):
        
        super().__init__()
        
        self.mode = mode
        self.delta = delta
        self.directory = directory
        self.overwriting = overwriting
        self.filename_format = filename_format
        self.candidates = candidates
        self.ignore_warnings = ignore_warnings
        self.logger = logger
        
        
        if not is_mode_valid(self.mode):
            raise ValueError(f"'mode' parameter shoud be 'min' or 'max', but given '{self.mode}'.")
        
        
        if isinstance(self.candidates, str):
            if self.candidates != "all":
                raise ValueError(f"'candidates' can be a string, but only with 1 value: 'all', but given '{self.candidates}'")
        else:
            if self.candidates < 0:
                if not self.ignore_warnings:
                    print(f"Parameter 'candidates' is lower than 0, so it will be setted to 0 (no saving checkpoints during training).")
                self.candidates = 0
            elif self.candidates == 0:
                if not self.ignore_warnings:
                    print(f"Parameter 'candidates' was setted to '0', which means that no saving checkpoints during training.")
        
        
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
                            warnings.warn(f"Found files in '{self.directory}' Be carefully with setting 'filename_format' to avoid overwriting other files.")
            else:
                raise NotADirectoryError(f"'{self.directory}' is not directory.")
        
        if self.filename_format.count(".") > 1:
            raise ValueError(f"'filename_format' must not has '.' in filename, but given '{self.filename_format}'.")
        
        if not self.is_filename_format_unique(self.filename_format):
            if not self.ignore_warnings:
                warnings.warn(f"Seems that 'filename_format' is not unique, maybe will be overwrited some useful checkpoints during training.")
            
            if not isinstance(self.candidates, str):
                if self.candidates > 1:
                    if not self.ignore_warnings:
                        warnings.warn(f"When 'filename_format' is not unique, number of candidates does not affect anything.")
            
            
        self.best_value = get_default_best_value(self.mode)
        self.all_candidates = []
        self.best_checkpoint_path = None
        self.step = None
        self.best_step = None
    
    
    def is_filename_format_unique(self, format_:str) -> bool:
        """
        description
        """
        return "{value}" in format_ or "{step}" in format_
    
    
    def __remove_files_from_directory(self, directory:str) -> None:
        """
        description
        """
        
        filenames = os.listdir(directory)
        pathes = [os.path.join(directory, filename) for filename in filenames]
        
        for path in pathes:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
                
    
    def state_dict(self) -> dict:
        state = {
            "directory": self.directory,
            "best_checkpoint_path": self.best_checkpoint_path,
            "best_value": self.best_value.item(),
            "filename_format": self.filename_format,
            "step": self.step,
            "best_step": self.best_step,
        }
        
        return state
    
    
    def load_state_dict(self, state_dict:dict):
        self.directory = state_dict["directory"]
        self.best_checkpoint_path = state_dict["best_checkpoint_path"]
        self.best_value = torch.tensor(state_dict["best_value"])
        self.filename_format = state_dict["filename_format"]
        self.step = state_dict["step"]
        self.best_step = state_dict["best_step"]
        
        return self     
    
    
    def append_candidate(self, path:str, value:Union[float, torch.Tensor, int]) -> None:   
        """
        description
        """
        
        if not os.path.exists(path):
            raise FileNotFoundError("`path` does not exist.")
        
        self.all_candidates.append([path, value])
        
    
    def __select_candidates(self) -> None:
        """
        description
        """
        if self.candidates != "all":
            if len(self.all_candidates) >= self.candidates:
                selected_candidates = self.all_candidates[-self.candidates:]
                deleted_candidates = 0
                for candidate in self.all_candidates:
                    if candidate not in selected_candidates:
                        path, value = candidate
                        
                        if os.path.exists(path):
                            os.remove(path)

                        deleted_candidates += 1
                
                self.all_candidates = self.all_candidates[-self.candidates:]
                print(f"Deleted {deleted_candidates} candidates from '{self.directory}'.")
                
            
        
    def format_filename(self, value:Union[float, torch.Tensor, int], step:int=None) -> str:
        """
        description
        """
        
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        value = str(value).replace(".", "")
        
        if step is not None:
            filename = self.filename_format.format(value=value, step=step)
        else:
            filename = self.filename_format.format(value=value, step="")
            
        return filename
        
    
    def create_checkpoint(self, value:Union[float, torch.Tensor, int], model:nn.Module, optimizer:Optional[Optimizer]=None, scheduler:Optional[_LRScheduler]=None, step:int=None) -> dict:
        """
        description
        """
        
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
        
        
    def __call__(self, value:Union[float, torch.Tensor, int], model:nn.Module, optimizer:Optional[Optimizer]=None, scheduler:Optional[_LRScheduler]=None, step:int=None) -> bool:
        """
        Inputs:
            value - description.
            model - description.
            optimizer - description.
            scheduler - description.
            step - description.
        
        Outputs:
            is_saved: bool - description.
            
        """
        
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        
        delta_value = get_delta_value(value=value, delta=self.delta, mode=self.mode)
        
        is_saved = False
        if compare_values(value=delta_value, other=self.best_value, mode=self.mode) and self.candidates != 0:
            checkpoint_filename = self.format_filename(value=value, step=step)
            checkpoint_path = os.path.join(self.directory, checkpoint_filename)
            
            checkpoint = self.create_checkpoint(value=value, 
                                                model=model, 
                                                optimizer=optimizer, 
                                                scheduler=scheduler, 
                                                step=step)
            
            torch.save(checkpoint, checkpoint_path)
            
            improvement_delta = abs(value - self.best_value)
            self.logger(f"'best_value' is improved by {improvement_delta}! New 'best_value': {value}. Checkpoint path: '{checkpoint_path}'.")
            
            self.append_candidate(value=value.item(), path=checkpoint_path)
            
            self.best_value = value
            self.best_step = step
            self.best_checkpoint_path = checkpoint_path
            
            self.__select_candidates()
            is_saved = True
        
        self.step = step
        
        return is_saved
    
    
    def __str__(self):
        return f"ModelCheckpoint(mode='{self.mode}', delta={self.delta}, directory='{self.directory}', overwriting={self.overwriting}, filename_format='{self.filename_format}', candidates={self.candidates}, ignore_warnings={self.ignore_warnings})"

    __repr__ = __str__