from optparse import Option
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from typing import Optional, Union, Any, Tuple
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import gc

from .utils import SchedulingStrategy, ValidationStrategy
from ..timer import Timer
from ..averager import Averager
from ..import_utils import is_wandb_available, wandb_run_exists, is_torch_xla_available
from ..utils import get_lr


if is_torch_xla_available():
    from torch_xla.amp import GradScaler
    import torch_xla.core.xla_model as xm

if is_wandb_available():
    import wandb


gc.enable()

class Trainer:
    def __init__(self, 
                 model:nn.Module, 
                 optimizer:optim.Optimizer,
                 teacher_model:Optional[nn.Module]=None,
                 scheduler:Optional[lr_scheduler._LRScheduler]=None, 
                 scheduling_strategy:str="step", 
                 gradient_accumulation_steps:int=1, 
                 gradient_scaling:bool=False, 
                 scaler:Optional["GradScaler"]=None,
                 gradient_norm:float=0, 
                 amp:bool=False, 
                 verbose:int=1, 
                 device:Optional[Union[str, torch.device]]="cpu", 
                 validation_strategy:str="epoch",
                 validation_steps:int=1, 
                 decimals:int=4, 
                 logger:Union[str, list]="print", 
                 epochs:int=1, 
                 time_format:str="{hours}:{minutes}:{seconds}") -> None:
        
        """
        
        Inputs:

            model: nn.Module - model to train.
            optimizer: optim.Optimizer - optimizer, which is used for training, i.e optimizing model's parameters.
            scheduler: Optional[lr_scheduler._LRScheduler] - scheduler for scheduling learning rate during training. Default: None.
            scheduling_strategy: str - strategy for calling scheduler's step. Possible values: ["step", "epoch"]. Default: "step".
            gradient_accumulation_steps: int - number of steps for calling optimizer's step. Default: 1.
            gradient_scaling:bool - if True, applies scaling for the loss, in order to prevent low gradient in AMP mode. Default: False.
            scaler:Optional["GradScaler"] - if `gradient_scaling` is True, the provided scaler will do scaling. Default: None.
            gradient_norm:float - max norm of gradients. Default: 0.
            amp:bool - if True, the training will use Auto Mixed Precision training, i.e training with half precision (16 bytes). Default: False.
            verbose:int - number of steps to print the results. Default: 1.
            device: Optional[Union[str, torch.device]] - device for model and batch's data. Default: torch.device("cpu").
            validation_strategy:str - strategy for validating model. Possible values: ["step", "epoch"]. Default: "epoch".
            validation_steps: int - number of steps to validate the model. Default: 1.
            decimals: int - number of decimals to show the numbers, e.g. loss, metrics, etc. Default: 4.
            epochs: int - number of epochs. Default: 1.
            logger: Union[str, list] - logger or loggers for logging training process, it can recieve list or just string of loggers. 
            Possible values: ["wandb", "print", "tqdm"]. Default: "print".
            time_format:str - format for printing the elapsed time. Default: "{hours}:{minutes}:{seconds}".
        

        """



        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduling_strategy = SchedulingStrategy(scheduling_strategy)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_scaling = gradient_scaling
        self.gradient_norm = gradient_norm
        self.amp = amp
        self.device = device
        self.verbose = verbose
        self.validation_strategy = ValidationStrategy(validation_strategy)
        self.validation_steps = validation_steps
        self.scaler = scaler
        self.decimals = decimals
        self.logger = logger
        self.epochs = epochs
        self.time_format = time_format   
        self.is_tpu = is_torch_xla_available()
        self.is_cuda = torch.cuda.is_available()
        self.__numpy_dtype = np.float16 if self.amp else np.float32
        self.__torch_dtype = torch.float16 if self.amp else torch.float32


        if not (0 < self.epochs):
            raise ValueError(f"`epochs` must be greater than 0, but given {self.epochs}.")
        
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"`model` must be subinstance of `torch.nn.Module`, but given `{type(self.model)}`")
        
        if self.teacher_model is not None:
            if not isinstance(self.teacher_model, nn.Module):
                raise TypeError(f"`teacher_model` must be subinstance of `torch.nn.Module`, but given `{type(self.teacher_model)}`")
            
        if not isinstance(self.gradient_accumulation_steps, int):
            raise TypeError(f"`gradient_accumulation_steps` must be integer type, but given `{type(self.gradient_accumulation_steps)}`")
        

        if self.device is None:
            if self.is_cuda:
                self.device = "cuda"
            elif self.is_tpu:
                self.device = xm.xla_device()
            else:
                self.device = "cpu"
        
        if self.gradient_scaling and self.scaler is None and self.amp:
            self.scaler = GradScaler()

        self.best_validation_loss, self.best_validation_metrics, self.best_validation_outputs = None, None, None
        self.lr_key = "lr"
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
    
    
    def train(self, 
              train_loader:DataLoader, 
              validation_loader:Optional[DataLoader]=None, 
              pseudo_loader:Optional[DataLoader]=None, 
              recalculate_metrics_at_end:bool=True, 
              return_validation_outputs:bool=True) -> tuple:
        
        total_time = timedelta(seconds=0)
        is_wandb = wandb_run_exists() and "wandb" in self.logger

        self.model.to(self.device)
        if self.teacher_model is not None: 
            self.teacher_model.to(self.device)
        
        if self.validation_strategy == ValidationStrategy.EPOCH:
            self.validation_steps = len(train_loader) * self.validation_steps
        if is_wandb:
            print(f"Weights & Biases Run: {wandb.run.get_url()}", end="\n"*2)

        train_loss, train_metrics = Averager(), Averager()
        for epoch in range(1, self.epochs+1):
            if "print" in self.logger: print(f"\nEpoch {epoch}/{self.epochs}", end="\n"*2)
            if "tqdm" in self.logger: train_loader = self.__tqdm_loader_wrapper(train_loader, f"Epoch {epoch}/{self.epochs}")

            epoch_train_loss, epoch_train_metrics = Averager(), Averager()
            steps = len(train_loader)    
            timer = Timer(self.time_format)
            
            self.model.zero_grad()
            for step, batch in enumerate(train_loader, 1):
                self.passed_steps += 1
                
                batch_size = len(batch)
                pseudo_batch = None if pseudo_loader is None else next(iter(pseudo_loader))
                
                batch_loss, batch_metrics = self.training_step(batch=batch, 
                                                               overall_loss=epoch_train_loss.average, 
                                                               overall_metrics=epoch_train_metrics.average,
                                                               step=self.passed_steps, 
                                                               epoch=epoch, 
                                                               pseudo_batch=pseudo_batch)

                lr = get_lr(self.optimizer, only_last=True, key=self.lr_key)

                if (step % self.gradient_accumulation_steps == 0) or (step == steps):
                    self.optimization_step()

                    if self.scheduling_strategy == SchedulingStrategy.STEP:
                        self.scheduling_step(loop="training")

                if self.gradient_accumulation_steps > 1:
                    batch_loss = batch_loss * self.gradient_accumulation_steps

                train_loss.update(batch_loss, n=batch_size)
                epoch_train_loss.update(batch_loss, n=batch_size)
                train_metrics.update(batch_metrics, n=batch_size)
                epoch_train_metrics.update(batch_metrics, n=batch_size)

                if is_wandb:
                    logs = {"train/loss": train_loss.average, 
                            "train/loss vs batch": batch_loss, 
                            "train/loss vs epoch": epoch_train_loss.average,
                            "lr": lr}

                    for metric in batch_metrics:
                        logs.update({f"train/{metric}": train_metrics.average[metric], 
                                     f"train/{metric} vs batch": batch_metrics[metric], 
                                     f"train/{metric} vs epoch": epoch_train_metrics.average[metric]})

                    wandb.log(logs, step=self.passed_steps) 

                if "tqdm" in self.logger:
                    train_loader.set_postfix_str(f"loss: {epoch_train_loss.average:.{self.decimals}}"
                                                 f"{self.format_metrics(epoch_train_metrics.average)}")
                if "print" in self.logger:
                     if step % self.verbose == 0 or step == steps and self.verbose > 0:
                        elapsed, remain = timer(step/steps)
                        print(f"{step}/{steps} - "
                              f"remain: {remain} - "
                              f"loss: {epoch_train_loss.average:.{self.decimals}}"
                              f"{self.format_metrics(epoch_train_metrics.average)} - "
                              f"lr: {lr}")


                if validation_loader is not None:
                    if (self.passed_steps % self.validation_steps) == 0:
                        if step > self.validation_steps: print()
                        validation_loss, validation_metrics, validation_outputs = self.validation_loop(loader=validation_loader, 
                                                                                                       return_outputs=return_validation_outputs, 
                                                                                                       recalculate_metrics_at_end=recalculate_metrics_at_end)
                        
                        self.scheduling_step(loss=validation_loss, loop="validation")


                        if is_wandb:
                            logs = {"validation/loss": validation_loss, 
                                    "train/loss vs validation steps": epoch_train_loss.average}

                            for metric, value in validation_metrics.items():
                                logs.update({f"validation/{metric}": value, 
                                             f"train/{metric} vs validation steps": epoch_train_metrics.average[metric]})

                            wandb.log(logs, step=self.passed_steps)

                        is_checkpoint_saved = self.model_checkpointing(loss=validation_loss, 
                                                                       metrics=validation_metrics,
                                                                       model=self.model, 
                                                                       optimizer=self.optimizer, 
                                                                       scheduler=self.scheduler, 
                                                                       step=self.passed_steps, 
                                                                       best_loss=self.best_validation_loss, 
                                                                       best_metrics=self.best_validation_metrics)

                        if is_checkpoint_saved:
                            self.best_validation_loss = validation_loss
                            self.best_validation_metrics = validation_metrics
                            self.best_validation_outputs = validation_outputs

                        del validation_outputs
                        gc.collect()
                            
                        print()

            if self.scheduling_strategy == SchedulingStrategy.EPOCH:
                self.scheduling_step(loop="training")

            if "tqdm" in self.logger and "print" not in self.logger:
                elapsed, remain = timer(1/1)

            epoch_elapsed_seconds = timer.elapsed_time.total_seconds()
            total_time += timedelta(seconds=epoch_elapsed_seconds)

            if is_wandb:
                wandb.log({"epoch": epoch}, step=self.passed_steps)

            if "tqdm" in self.logger: train_loader.close()

            print(f"\nTraining loss: {epoch_train_loss.average:.{self.decimals}}"
                  f"{self.format_metrics(epoch_train_metrics.average)}")

            if validation_loader is not None:
                print(f"Validation loss: {self.best_validation_loss:.{self.decimals}}"
                      f"{self.format_metrics(self.best_validation_metrics)}")

            total_time_string = Timer.format_time(total_time, time_format=self.time_format)
            print(f"Total time: {total_time_string}")

        gc.collect()

        if validation_loader is not None:
            if return_validation_outputs:
                return (epoch_train_loss.average, epoch_train_metrics.average), (self.best_validation_loss, self.best_validation_metrics, self.best_validation_outputs)

            return (epoch_train_loss.average, epoch_train_metrics.average), (self.best_validation_loss, self.best_validation_metrics)

        return (epoch_train_loss.average, epoch_train_metrics.average)

        
    def backward_step(self, loss:torch.Tensor) -> torch.Tensor:
        if self.scaler is not None and self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss
    

    def optimization_step(self) -> None:     
        """
        Applies optimization step.
        """    

        self.clip_gradients()

        if self.scaler is not None and self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.is_tpu:
            xm.optimizer_step(self.optimizer)
        else:
            self.optimizer.step()

        self.model.zero_grad()
        

    def scheduling_step(self, loss:Optional[torch.Tensor]=None, loop:str="training") -> None:
        """
        Applies learning rate scheduling.
        """

        if self.scheduler is not None:
            if loop == "validation":
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
            else:
                if not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

                    
    def training_step(self, 
                      batch:Any, 
                      overall_loss:Optional[float]=None, 
                      overall_metrics:Optional[dict]=None, 
                      step:Optional[int]=None, 
                      epoch:Optional[int]=None, 
                      pseudo_batch:Optional[Any]=None) -> Tuple[torch.Tensor, dict]:

        """
        Applies training step, i.e calculating losses, metrics and returns them for further optimization.
        """
        
        self.model.train()
        with autocast(enabled=self.amp):
            loss, outputs = self.calculate_loss(batch=batch, model=self.model, return_outputs=True, device=self.device)
            targets = self.get_targets(batch)
            metrics = self.calculate_metrics(predictions=outputs, targets=targets, device=self.device)

            if self.gradient_accumulation_steps > 1:
                loss /= self.gradient_accumulation_steps
            
            loss = self.backward_step(loss=loss)

            adversarial_loss = self.adversarial_step(batch=batch, 
                                                     model=self.model, 
                                                     device=self.device, 
                                                     loss=overall_loss, 
                                                     metrics=overall_metrics, 
                                                     step=step, 
                                                     epoch=epoch)

            if adversarial_loss is not None:
                adversarial_loss = self.backward_step(loss=adversarial_loss)

            if pseudo_batch is not None and self.teacher_model is not None:
                pseudo_loss = self.pseudo_labeling_step(batch=batch,
                                                        pseudo_batch=pseudo_batch,
                                                        model=self.model, 
                                                        teacher_model=self.teacher_model, 
                                                        loss=loss, 
                                                        metrics=metrics,
                                                        step=step, 
                                                        epoch=epoch, 
                                                        device=self.device)

                if pseudo_loss is not None:
                    pseudo_loss = self.backward_step(loss=pseudo_loss)

        del batch, targets, outputs
        gc.collect()
        
        return loss.detach(), metrics
                
    def clip_gradients(self) -> None:
        """
        Applies gradient clipping for model's parameters.
        """
        if self.gradient_norm > 0:
            if self.gradient_scaling:
                self.scaler.unscale_(self.optimizer)
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_norm)
        

    def calculate_loss(self, 
                      batch:Any, 
                      model:nn.Module, 
                      return_outputs:bool=True, 
                      device:Union[str, torch.device]="cpu") -> torch.Tensor:
        """
        Calculates loss.        
        """

        raise NotImplementedError(f"`calculate_loss` function is not implemented.")

    def get_targets(self, batch:Any) -> Any:
        """
        Returns targets from batch.

        Inputs:
            batch: Any - batch of data.
        
        Returns:
            targets: Any - batch's targets. Default: [].

        """

        return []
    
    def calculate_metrics(self, predictions:Any, targets:Any, device:Union[str, torch.device]="cpu") -> dict:
        """
        Calculates metrics

        Inputs:
            predictions: Any - outputs of model from `calculate_loss`.
            targets: Any - outputs of `get_targets` function.
            device: Union[str, torch.device] - device. Default: "cpu".
        
        Returns:
            metrics: dict - calculated metrics. Default: {}.

        """

        return {}
    
    def model_checkpointing(self, 
                            loss:float, 
                            metrics:dict, 
                            model:nn.Module, 
                            optimizer:Optional[optim.Optimizer]=None, 
                            scheduler:Optional[lr_scheduler._LRScheduler]=None, 
                            step:Optional[int]=None, 
                            best_loss:Optional[int]=None, 
                            best_metrics:Optional[dict]=None) -> bool:

        """
        Saves model checkpoints.
        """

        return True

    def pseudo_labeling_step(self, 
                             batch:Any, 
                             pseudo_batch:Any, 
                             model:nn.Module, 
                             teacher_model:nn.Module, 
                             loss:Optional[float]=None, 
                             metrics:Optional[dict]=None, 
                             step:Optional[int]=None, 
                             epoch:Optional[int]=None, 
                             device:Optional[Union[str, torch.device]]="cpu") -> torch.Tensor:
        pass
    
    def adversarial_step(self, 
                         batch:Any, 
                         model:nn.Module, 
                         device:Optional[Union[str, torch.device]]="cpu", 
                         loss:Optional[float]=None, 
                         metrics:Optional[dict]=None, 
                         step:Optional[int]=None, 
                         epoch:Optional[int]=None) -> torch.Tensor:
        """
        Applies Adversarial Training.
        """

        pass
    
    def validation_loop(self, 
                        loader:DataLoader, 
                        return_outputs:bool=True, 
                        recalculate_metrics_at_end:bool=False) -> Tuple[Any, dict]:

        """
        Runs validation loop.
        """
        
        self.model.eval()
        loss, metrics = Averager(), Averager()
        timer = Timer(self.time_format)
        outputs, targets = [], []
        steps = len(loader)
        
        if "tqdm" in self.logger: loader = self.__tqdm_loader_wrapper(loader, f"[Validation]")

        is_targets = False
        for step, batch in enumerate(loader, 1):
            with torch.no_grad():
                with autocast(enabled=self.amp):
                    batch_size = len(batch)
                    batch_loss, batch_outputs = self.calculate_loss(batch=batch, 
                                                                    model=self.model, 
                                                                    return_outputs=True, 
                                                                    device=self.device)
                    
                    batch_targets = self.get_targets(batch)
                    batch_metrics = self.calculate_metrics(predictions=batch_outputs, targets=batch_targets, device=self.device)

                    loss.update(batch_loss.item(), n=batch_size)
                    metrics.update(batch_metrics, n=batch_size)

                    if batch_targets is not None and recalculate_metrics_at_end:
                        if isinstance(batch_targets, dict):
                            targets.append(batch_targets)
                        else:
                            targets.extend(batch_targets.to("cpu").numpy().astype(self.__numpy_dtype))

                        is_targets = True

                    if return_outputs or recalculate_metrics_at_end:
                        outputs.extend(batch_outputs.to("cpu").numpy().astype(self.__numpy_dtype))

                    if step == steps and recalculate_metrics_at_end and is_targets:
                        outputs = torch.tensor(outputs, dtype=self.__torch_dtype)
                        targets = torch.tensor(targets, dtype=self.__torch_dtype)

                        metrics = Averager(self.calculate_metrics(predictions=outputs, targets=targets))

                    if "tqdm" in self.logger:
                        loader.set_postfix_str(f"loss: {loss.average:.{self.decimals}}"
                                               f"{self.format_metrics(metrics.average)}")

                    if "print" in self.logger:
                        if step % self.verbose == 0 or step == steps and self.verbose > 0:
                            elapsed, remain = timer(step/steps)

                            print(f"[Validation] "
                                  f"{step}/{steps} - "
                                  f"remain: {remain} - "
                                  f"loss: {loss.average:.{self.decimals}}"
                                  f"{self.format_metrics(metrics.average)}")

                    del batch, batch_outputs, batch_targets
                    gc.collect()

        if not recalculate_metrics_at_end: 
            outputs = torch.tensor(outputs, dtype=self.__torch_dtype)

        if "tqdm" in self.logger:
            loader.close()

        if return_outputs:
            outputs = outputs.to("cpu").numpy().astype(self.__numpy_dtype)
        else:
            outputs = None


        return (loss.average, metrics.average, outputs) if return_outputs else (loss.average, metrics.average)


    def format_metrics(self, metrics:dict, sep:str=" - ", add_sep_to_start:bool=True) -> str:
        """
        Formats the given dictionary of metrics and retuns it as string.
        """

        if metrics != {}:
            string = sep.join([f"{k}: {v:.{self.decimals}}" for k, v in metrics.items()])
            return sep + string if add_sep_to_start else string 

        return ""