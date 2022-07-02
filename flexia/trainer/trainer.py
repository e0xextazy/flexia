import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from typing import Optional, Union, Any, Tuple
from torch.utils.data import DataLoader
from datetime import timedelta
import numpy as np


from ..third_party.addict import Dict
from .trainer_enums import SchedulingStrategy, ValidationStrategy
from ..timer import Timer
from ..averager import Averager
from ..utils import get_lr, initialize_device


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
                 loggers:Union[str, list]="print", 
                 epochs:int=1, 
                 time_format:str="{hours}:{minutes}:{seconds}", 
                 logging_filename:str="training_logs.log", 
                 logging_format:str="%(message)s", 
                 callbacks=[]) -> None:
        
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
        self.loggers = loggers
        self.epochs = epochs
        self.time_format = time_format   
        self.logging_filename = logging_filename
        self.logging_format = logging_format
        self.callbacks = callbacks


        if not (0 < self.epochs):
            raise ValueError(f"`epochs` must be greater than 0, but given {self.epochs}.")
        
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"`model` must be subinstance of `torch.nn.Module`, but given `{type(self.model)}`")
        
        if self.teacher_model is not None:
            if not isinstance(self.teacher_model, nn.Module):
                raise TypeError(f"`teacher_model` must be subinstance of `torch.nn.Module`, but given `{type(self.teacher_model)}`")
            
        if not isinstance(self.gradient_accumulation_steps, int):
            raise TypeError(f"`gradient_accumulation_steps` must be integer type, but given `{type(self.gradient_accumulation_steps)}`")

        self.device = initialize_device(self.device)

        if self.gradient_scaling and self.scaler is None and self.amp:
            self.scaler = GradScaler()

        self.best_validation_loss, self.best_validation_metrics, self.best_validation_outputs = None, None, None
        self.lr_key = "lr"
        self.current_step = 0
        self.current_epoch = 0
        self.history = Dict()
        self.train_loader, self.validation_loader = None, None
    
    
    def train(self, 
              train_loader:DataLoader, 
              validation_loader:Optional[DataLoader]=None, 
              pseudo_loader:Optional[DataLoader]=None, 
              return_validation_outputs:bool=True) -> tuple:
        
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        total_time = timedelta(seconds=0)

        self.model.to(self.device)
        if self.teacher_model is not None: 
            self.teacher_model.to(self.device)
        
        if self.validation_strategy == ValidationStrategy.EPOCH:
            self.validation_steps = len(self.train_loader) * self.validation_steps
        else:
            # validation model after N training steps!
            self.validation_steps = int(self.validation_steps * self.gradient_accumulation_steps)

        self.loggers.on_training_start(self)
        self.history.update({"epochs": self.epochs})

        train_loss, train_metrics = Averager(), Averager()
        for epoch in range(1, self.epochs+1):
            self.current_epoch = epoch

            self.history.update({"epoch": self.current_epoch})

            log_message = f"\nEpoch {epoch}/{self.epochs}"
            self.log(log_message, end="\n"*2)

            epoch_train_loss, epoch_train_metrics = Averager(), Averager()
            steps = len(self.train_loader)    
            timer = Timer(self.time_format)
            
            self.model.zero_grad()
            for step, batch in enumerate(self.train_loader, 1):
                self.current_step += 1
                
                batch_size = len(batch)
                pseudo_batch = None if pseudo_loader is None else next(iter(pseudo_loader))
                
                batch_loss, batch_metrics = self.training_step(batch=batch,
                                                               pseudo_batch=pseudo_batch,
                                                               overall_loss=epoch_train_loss.average, 
                                                               overall_metrics=epoch_train_metrics.average,
                                                               step=self.current_step, 
                                                               epoch=epoch)

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

                elapsed, remain = timer(step/steps)

                self.history.update({
                    "train_loss": train_loss.average,
                    "train_loss_batch": batch_loss,
                    "train_loss_epoch": epoch_train_loss.average,
                    "lr": lr,
                    "elapsed_epoch": elapsed,
                    "remain_epoch": remain,
                    "step_epoch": step,
                    "step": self.current_step,
                    "train_metrics_list": list(batch_metrics.keys())
                })

                for metric in batch_metrics:
                    self.history.update({
                        f"train_{metric}": train_metrics.average[metric],
                        f"train_{metric}_batch": batch_metrics[metric],
                        f"train{metric}_epoch": epoch_train_metrics.average[metric]
                    })

                self.loggers.on_training_step_end(self)


                if self.validation_loader is not None:
                    if (self.current_step % self.validation_steps) == 0:

                        validation_loss, validation_metrics, validation_outputs = self.validation_loop(loader=self.validation_loader, 
                                                                                                       return_outputs=return_validation_outputs)
                        
                        self.scheduling_step(loss=validation_loss, loop="validation")

                        self.history.update({
                            "validation_loss": validation_loss, 
                            "train_loss_validation_steps": epoch_train_loss.average,
                            "validation_metrics_list": list(validation_metrics.keys()),
                        })

                        for metric in validation_metrics:
                            self.history.update({
                                f"validation_{metric}": validation_metrics[metric],
                                f"train_{metric}_validation steps": epoch_train_metrics.average[metric],
                            })

                        self.loggers.on_validation_end(self)

                        is_checkpoint_saved = self.model_checkpointing(loss=validation_loss, 
                                                                       metrics=validation_metrics,
                                                                       step=self.current_step, 
                                                                       best_loss=self.best_validation_loss, 
                                                                       best_metrics=self.best_validation_metrics)

                        if is_checkpoint_saved:
                            self.best_validation_loss = validation_loss
                            self.best_validation_metrics = validation_metrics
                            self.best_validation_outputs = validation_outputs

                        del validation_outputs
                            
                        print()

            if self.scheduling_strategy == SchedulingStrategy.EPOCH:
                self.scheduling_step(loop="training")

            epoch_elapsed_seconds = timer.elapsed_time.total_seconds()
            total_time += timedelta(seconds=epoch_elapsed_seconds)

            self.loggers.on_training_end(self)

        if self.validation_loader is not None:
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
        self.clip_gradients()

        if self.scaler is not None and self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.model.zero_grad()
        

    def scheduling_step(self, loss:Optional[torch.Tensor]=None, loop:str="training") -> None:
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

        self.model.train()
        with autocast(enabled=self.amp):
            loss, outputs = self.compute_loss(batch=batch, return_outputs=True)
            metrics = self.compute_metrics(predictions=outputs, batch=batch)

            if self.gradient_accumulation_steps > 1:
                loss /= self.gradient_accumulation_steps
            
            loss = self.backward_step(loss=loss)

            if pseudo_batch is not None and self.teacher_model is not None:
                pseudo_loss = self.pseudo_labeling_step(batch=batch,
                                                        pseudo_batch=pseudo_batch,
                                                        loss=loss, 
                                                        metrics=metrics,
                                                        step=step, 
                                                        epoch=epoch)

                if pseudo_loss is not None:
                    pseudo_loss = self.backward_step(loss=pseudo_loss)

        return loss.detach(), metrics
                
    def clip_gradients(self) -> None:
        if self.gradient_norm > 0:
            if self.gradient_scaling:
                self.scaler.unscale_(self.optimizer)
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_norm)
        

    def compute_loss(self, 
                      batch:Any, 
                      return_outputs:bool=True) -> torch.Tensor:
        raise NotImplementedError(f"`compute_loss` function is not implemented.")
    
    def compute_metrics(self, predictions:Any, batch:Any) -> dict:
        return {}
    
    def model_checkpointing(self, 
                            loss:float, 
                            metrics:dict, 
                            step:Optional[int]=None, 
                            best_loss:Optional[int]=None, 
                            best_metrics:Optional[dict]=None) -> bool:
        return True


    def pseudo_labeling_step(self, 
                             batch:Any, 
                             pseudo_batch:Any, 
                             loss:Optional[float]=None, 
                             metrics:Optional[dict]=None, 
                             step:int=0, 
                             epoch:int=0) -> torch.Tensor:
        pass
    
    def validation_loop(self, loader:DataLoader, return_outputs:bool=True) -> Tuple[Any, dict]:
        self.model.to(self.device)
        self.model.eval()
        loss, metrics = Averager(), Averager()
        timer = Timer(self.time_format)
        outputs, targets = [], []
        steps = len(loader)
        
        self.loggers.on_validation_start(self)

        for step, batch in enumerate(loader, 1):
            with torch.no_grad():
                with autocast(enabled=self.amp):
                    batch_size = len(batch)
                    batch_loss, batch_outputs = self.compute_loss(batch=batch, return_outputs=True)
                    batch_metrics = self.compute_metrics(predictions=batch_outputs, batch=batch)

                    loss.update(batch_loss.item(), n=batch_size)
                    metrics.update(batch_metrics, n=batch_size)

                    self.history.update({
                        "validation_loss": loss.average,
                        "validation_loss_batch": batch_loss,
                        "validation_step": step,
                        "validation_metrics_list": list(metrics.keys())
                    })


                    if return_outputs:
                        outputs.extend(batch_outputs.to("cpu").numpy())

                    self.loggers.on_validation_step_end(trainer)

        if return_outputs:
            outputs = np.asarray(outputs)
        else:
            outputs = None


        return (loss.average, metrics.average, outputs)