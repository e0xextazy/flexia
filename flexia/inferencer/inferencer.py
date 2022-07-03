import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Optional, Union, Any
from torch.utils.data import DataLoader
import numpy as np
from datetime import timedelta
import gc

from ..timer import Timer


class Inferencer:
    def __init__(self, 
                 model:nn.Module, 
                 device:Optional[Union[str, torch.device]]="cpu", 
                 amp:bool=False, 
                 logger:Union[str, list]="print", 
                 verbose:int=1, 
                 time_format:str="{hours}:{minutes}:{seconds}", 
                 logging_filename:str="inference_logs.log", 
                 logging_format:str="%(message)s"):

        self.model = model
        self.device = device
        self.amp = amp
        self.logger = logger
        self.time_format = time_format
        self.verbose = verbose
        self.logging_filename = logging_filename
        self.logging_format = logging_format
        
        if "logging" in self.logger:
            self.logging_logger = get_logger(name="inferencer", 
                                             format=self.logging_format,  
                                             filename=self.logging_filename)

        self.is_cuda = torch.cuda.is_available()
        self.__numpy_dtype = np.float16 if self.amp else np.float32
        self.__torch_dtype = torch.float16 if self.amp else torch.float32

        if self.device is None:
            if self.is_cuda:
                self.device = "cuda"
            else:
                self.device = "cpu"

        self.passed_steps = 0


    def prediction_step(self, batch:Any):
        """
        Returns outputs.
        """

        raise NotImplementedError(f"`prediction_step` function is not implemented.")
        
    def __call__(self, loader:DataLoader):
        """
        Runs inference.
        """
        self.model.to(self.device)
        self.model.eval()         

        steps = len(loader)
        outputs = []
        
        total_time = timedelta(seconds=0)

        timer = Timer(self.time_format)

        if "tqdm" in self.logger: loader = tqdm_loader_wrapper(loader, f"Inference")

        for step, batch in enumerate(loader, 1):
            self.passed_steps += 1
            with torch.no_grad():
                with autocast(enabled=self.amp):
                    batch_size = len(batch)
                    batch_outputs = self.prediction_step(batch=batch)

                    if "print" in self.logger or "logging" in self.logger:
                        if step % self.verbose == 0 or step == steps and self.verbose > 0:
                            elapsed, remain = timer(step/steps)
                            log_message = f"[Prediction] {step}/{steps} - elapsed: {elapsed} - remain: {remain}"
                            self.log(log_message)
                     
                    batch_outputs = batch_outputs.to("cpu").numpy().astype(self.__numpy_dtype)
                    outputs.extend(batch_outputs)
                    
        if "tqdm" in self.logger and "print" not in self.logger:
            elapsed, remain = timer(1/1)

        elapsed_seconds = timer.elapsed_time.total_seconds()
        total_time += timedelta(seconds=elapsed_seconds)

        if "tqdm" in self.logger: loader.close()

        outputs = torch.tensor(outputs, dtype=self.__torch_dtype)
        outputs = outputs.to("cpu").numpy().astype(self.__numpy_dtype)

        gc.collect()
        
        if "print" in self.logger or "logging" in self.logger:
            total_time_string = Timer.format_time(total_time, time_format=self.time_format)
            log_message = f"Total time: {total_time_string}"
            self.log(log_message)

        return outputs


    def log(self, message:str, end:str="\n") -> None:
        if "print" in self.logger:
            print(message, end=end)

        if "logging" in self.logger:
            self.logging_logger.debug(message)