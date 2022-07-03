import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import warnings
import shutil
import os
import gc
from typing import Callable, Union, Optional
from .callback import Callback
from .utils import compare_values, get_delta_value, get_mode_values
from ..utils import to_tensor


class ModelCheckpoint(Callback):  
    def __init__(self, 
                 mode:str="min", 
                 delta:Union[float, int]=0.0, 
                 directory:str="./", 
                 overwriting:bool=False, 
                 filename_format:str="checkpoint_{step}_{value}.pth", 
                 num_candidates:Union[str, float, int]=1, 
                 ignore_warnings:bool=False, 
                 logger:Callable[[str], str]=print):
        
        super().__init__()
        
        self.mode = mode
        self.delta = delta
        self.directory = directory
        self.overwriting = overwriting
        self.filename_format = filename_format
        self.num_candidates = num_candidates
        self.ignore_warnings = ignore_warnings
        self.logger = logger
        
        self.best_value, *_ = get_mode_values(self.mode)
        
        if isinstance(self.num_candidates, str):
            if self.num_candidates != "all":
                raise ValueError(f"`num_candidates` can be a string, but only with 1 value: 'all', but given '{self.num_candidates}'")
        else:
            if self.num_candidates < 0:
                if not self.ignore_warnings:
                    warnings.warn(f"`num_candidates` is lower than 0, so it will be setted to 0 (no saving checkpoints during training).")
                self.num_candidates = 0
            elif self.num_candidates == 0:
                if not self.ignore_warnings:
                    warnings.warn(f"`num_candidates` was setted to '0', which means that no saving checkpoints during training.")
        
        
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
            
            if not isinstance(self.num_candidates, str):
                if self.num_candidates > 1:
                    if not self.ignore_warnings:
                        warnings.warn(f"When 'filename_format' is not unique, number of candidates does not affect anything.")
            
        self.all_candidates = []
        self.best_checkpoint_path = None
        self.step = None
        self.best_step = None
    
    
    def is_filename_format_unique(self, format_:str) -> bool:
        """
        Checks for the uniqueness of the format.
        """
        return "{value}" in format_ or "{step}" in format_
    
    
    def __remove_files_from_directory(self, directory:str) -> None:
        """
        Removes all files and folders from directory.
        """
        
        filenames = os.listdir(directory)
        pathes = [os.path.join(directory, filename) for filename in filenames]
        
        for path in pathes:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    
    
    def append_candidate(self, path:str, value:Union[float, torch.Tensor, int]) -> None:   
        """
        Appends new candidate.
        """
        
        if not os.path.exists(path):
            raise FileNotFoundError("`path` does not exist.")
        
        self.all_candidates.append([path, value])
        
    
    def __select_candidates(self) -> None:
        """
        Deleted not selected candidates.
        """
        if self.num_candidates != "all":
            if len(self.all_candidates) >= self.num_candidates and self.is_filename_format_unique(self.filename_format):
                selected_candidates = self.all_candidates[-self.num_candidates:]
                deleted_candidates = 0
                for candidate in self.all_candidates:
                    if candidate not in selected_candidates:
                        path, value = candidate
                        
                        if os.path.exists(path):
                            os.remove(path)

                        deleted_candidates += 1
                
                self.all_candidates = self.all_candidates[-self.num_candidates:]
                self.logger(f"Deleted {deleted_candidates} candidates from '{self.directory}'.")
                
            
        
    def format_filename(self, value:Union[float, torch.Tensor, int], step:Optional[int]=None) -> str:
        """
        Formats checkpoint's filename with the given value and step.
        """
        
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        value = str(value).replace(".", "")
        
        if step is not None:
            filename = self.filename_format.format(value=value, step=step)
        else:
            filename = self.filename_format.format(value=value, step="")
            
        return filename
        
    
    def create_checkpoint(self, 
                          value:Union[torch.Tensor, float, int], 
                          model:nn.Module, 
                          optimizer:Optional[Optimizer]=None, 
                          scheduler:Optional[_LRScheduler]=None, 
                          step:Optional[int]=None) -> dict:
        """
        Creates checkpoints dictionary for further saving.
        
        Inputs:
            value: Union[torch.Tensor, float, int] - monitored value.
            model: nn.Module - PyTorch's module.
            optimizer: Optional[Optimizer] - PyTorch's or HuggingFace Transformers's optimizer.
            scheduler: Optional[_LRScheduler] - PyTorch's or HuggingFace Transformers's scheduler.
            step: Optional[int] - step of calling Model Checkpoint.
            
        Outputs:
            checkpoint: dict - created checkpoint.
        
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
        
        
    def __call__(self, 
                 value:Union[torch.Tensor, float, int], 
                 model:nn.Module, 
                 optimizer:Optional[Optimizer]=None, 
                 scheduler:Optional[_LRScheduler]=None, 
                 step:Optional[int]=None) -> bool:
        """
        Inputs:
            value: Union[torch.Tensor, float, int] - monitored value.
            model: nn.Module - PyTorch's module.
            optimizer: Optional[Optimizer] - PyTorch's or HuggingFace Transformers's optimizer.
            scheduler: Optional[_LRScheduler] - PyTorch's or HuggingFace Transformers's scheduler.
            step: Optional[int] - step of calling Model Checkpoint.
        
        Outputs:
            is_saved: bool - returns True if checkpoint was saved.
            
        """
        
        value = to_tensor(value)
        
        delta_value = get_delta_value(value=value, delta=self.delta, mode=self.mode)

        is_saved = False
        if compare_values(value=delta_value, other=self.best_value, mode=self.mode) and self.num_candidates != 0:
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

            # removing checkpoint from memory
            del checkpoint
            gc.collect()
        
        self.step = step
        
        return is_saved