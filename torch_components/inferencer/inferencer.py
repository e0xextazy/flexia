import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Optional, Union, Any
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import gc

from ..timer import Timer
from ..import_utils import is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

class Inferencer:
    def __init__(self, 
                 model:nn.Module, 
                 device:Optional[Union[str, torch.device]]="cpu", 
                 amp:bool=False, 
                 logger:Union[str, list]="print", 
                 verbose:int=1, 
                 time_format:str="{hours}:{minutes}:{seconds}"):

        """
        
        Inputs:

            model: nn.Module - model to train.
            amp:bool - if True, the training will use Auto Mixed Precision training, i.e training with half precision (16 bytes). Default: False.
            verbose:int - number of steps to print the results. Default: 1.
            device: Optional[Union[str, torch.device]] - device for model and batch's data. Default: torch.device("cpu").
            logger: Union[str, list] - logger or loggers for logging training process, it can recieve list or just string of loggers. 
            Possible values: ["wandb", "print", "tqdm"]. Default: "print".
            time_format:str - format for printing the elapsed time. Default: "{hours}:{minutes}:{seconds}".
        

        """

        self.model = model
        self.device = device
        self.amp = amp
        self.logger = logger
        self.time_format = time_format
        self.verbose = verbose
        
        self.is_tpu = is_torch_xla_available()
        self.is_cuda = torch.cuda.is_available()
        self.__numpy_dtype = np.float16 if self.amp else np.float32
        self.__torch_dtype = torch.float16 if self.amp else torch.float32

        if self.device is None:
            if self.is_cuda:
                self.device = "cuda"
            elif self.is_tpu:
                self.device = xm.xla_device()
            else:
                self.device = "cpu"

        self.passed_steps = 0

    
    def __tqdm_loader_wrapper(self, loader:DataLoader, description:str="") -> Any:
        """
        Wraps loader into `tqdm` loop.

        Inputs:
            loader: DataLoader - loader to wrap.
            description: str - description for `tqdm` loop.
        """

        bar_format = "{l_bar} {bar} {n_fmt}/{total_fmt} - remain: {remaining}{postfix}"
        loader = tqdm(iterable=loader, 
                      total=len(loader),
                      colour="#000",
                      bar_format=bar_format)

        loader.set_description_str(description)
        return loader


    def prediction_step(self, 
                        batch:Any, 
                        model:nn.Module, 
                        device:Optional[Union[str, torch.device]]="cpu"):
        """
        Returns outputs.
        """

        raise NotImplementedError(f"`prediction_step` function is not implemented.")
        
    def __call__(self, loader:DataLoader):
        """
        Runs inference.
        """
        
        steps = len(loader)
        outputs = []
        
        total_time = timedelta(seconds=0)

        timer = Timer(self.time_format)

        if "tqdm" in self.logger: loader = self.__tqdm_loader_wrapper(loader, f"Inference")

        self.model.eval() 
        for step, batch in enumerate(loader, 1):
            self.passed_steps += 1
            with torch.no_grad():
                with autocast(enabled=self.amp):
                    batch_size = len(batch)
                    batch_outputs = self.prediction_step(batch=batch, 
                                                         model=self.model, 
                                                         device=self.device)

                    if "print" in self.logger:
                        if step % self.verbose == 0 or step == steps and self.verbose > 0:
                            elapsed, remain = timer(step/steps)
                            print(f"{step}/{steps} - "
                                f"remain: {remain} - ")
                    
                    batch_outputs = batch_outputs.to("cpu").numpy().astype(self.__numpy_dtype)
                    outputs.extend(batch_outputs)
                    
        if "tqdm" in self.logger and "print" not in self.logger:
            elapsed, remain = timer(1/1)

        elapsed_seconds = timer.elapsed_time.total_seconds()
        total_time += timedelta(seconds=elapsed_seconds)

        if "tqdm" in self.logger: loader.close()

        total_time_string = Timer.format_time(total_time, time_format=self.time_format)
        print(f"Total time: {total_time_string}")

        outputs = torch.tensor(outputs, dtype=self.__torch_dtype)
        outputs = outputs.to("cpu").numpy().astype(self.__numpy_dtype)

        gc.collect()
        
        return outputs