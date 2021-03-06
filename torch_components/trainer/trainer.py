import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from typing import Optional, Union, Any, Tuple
from torch.utils.data import DataLoader
from datetime import timedelta
import gc

from .utils import SchedulingStrategy, ValidationStrategy
from ..timer import Timer
from ..averager import Averager
from ..import_utils import is_wandb_available, wandb_run_exists
from ..utils import get_lr, tqdm_loader_wrapper, get_logger

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
                 time_format:str="{hours}:{minutes}:{seconds}", 
                 logging_filename:str="training_logs.log", 
                 logging_format:str="%(message)s") -> None:
        
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
            device: Optional[Union[str, torch.device]] - device for model and batch's data. Default: "cpu".
            validation_strategy:str - strategy for validating model. Possible values: ["step", "epoch"]. Default: "epoch".
            validation_steps: int - number of steps to validate the model. Default: 1.
            decimals: int - number of decimals to show the numbers, e.g. loss, metrics, etc. Default: 4.
            epochs: int - number of epochs. Default: 1.
            logger: Union[str, list] - logger or loggers for logging training process, it can recieve list or just string of loggers. 
            Possible values: ["wandb", "print", "tqdm", "logging]. Default: "print".
            time_format:str - format for printing the elapsed time. Default: "{hours}:{minutes}:{seconds}".
            logging_filename: str - Default: "training_logs.log".
            logging_format: str - Default: "%(message)s".

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
        self.logging_filename = logging_filename
        self.logging_format = logging_format
        self.is_cuda = torch.cuda.is_available()


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
            else:
                self.device = "cpu"
        
        if self.gradient_scaling and self.scaler is None and self.amp:
            self.scaler = GradScaler()

        if "logging" in self.logger:
            self.logging_logger = get_logger(name="trainer", 
                                             format=self.logging_format,  
                                             filename=self.logging_filename)

        self.best_validation_loss, self.best_validation_metrics, self.best_validation_outputs = None, None, None
        self.lr_key = "lr"
        self.passed_steps = 0
    
    
    def train(self, 
              train_loader:DataLoader, 
              validation_loader:Optional[DataLoader]=None, 
              pseudo_loader:Optional[DataLoader]=None, 
              recompute_metrics_at_end:bool=True, 
              return_validation_outputs:bool=True) -> tuple:
        
        total_time = timedelta(seconds=0)
        is_wandb = wandb_run_exists() and "wandb" in self.logger

        self.model.to(self.device)
        if self.teacher_model is not None: 
            self.teacher_model.to(self.device)
        
        if self.validation_strategy == ValidationStrategy.EPOCH:
            self.validation_steps = len(train_loader) * self.validation_steps
        else:
            # validation model after N training steps!
            self.validation_steps = int(self.validation_steps * self.gradient_accumulation_steps)

        if is_wandb:
            print(f"Weights & Biases Run: {wandb.run.get_url()}", end="\n"*2)

        train_loss, train_metrics = Averager(), Averager()
        for epoch in range(1, self.epochs+1):
            log_message = f"\nEpoch {epoch}/{self.epochs}"
            self.log(log_message, end="\n"*2)
            if "tqdm" in self.logger: train_loader = tqdm_loader_wrapper(train_loader, f"Epoch {epoch}/{self.epochs}")

            epoch_train_loss, epoch_train_metrics = Averager(), Averager()
            steps = len(train_loader)    
            timer = Timer(self.time_format)
            
            self.model.zero_grad()
            for step, batch in enumerate(train_loader, 1):
                self.passed_steps += 1
                
                batch_size = len(batch)
                pseudo_batch = None if pseudo_loader is None else next(iter(pseudo_loader))
                
                batch_loss, batch_metrics = self.training_step(batch=batch,
                                                               pseudo_batch=pseudo_batch,
                                                               overall_loss=epoch_train_loss.average, 
                                                               overall_metrics=epoch_train_metrics.average,
                                                               step=self.passed_steps, 
                                                               epoch=epoch)

                lr = get_lr(self.optimizer, only_last=True, key=self.lr_key)

                if (step % self.gradient_accumulation_steps == 0) or (step == steps):
                    self.optimization_step()

                    if self.scheduling_strategy == SchedulingStrategy.STEP:
                        self.scheduling_step(loop="training")

                self.on_training_step_end(step=self.passed_steps, epoch=epoch)

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

                if "print" in self.logger or "logging" in self.logger:
                     if step % self.verbose == 0 or step == steps and self.verbose > 0:
                        elapsed, remain = timer(step/steps)
                    
                        log_message = f"{step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {epoch_train_loss.average:.{self.decimals}} {self.format_metrics(epoch_train_metrics.average)} - lr: {lr}"
                        self.log(log_message)


                if validation_loader is not None:
                    if (self.passed_steps % self.validation_steps) == 0:
                        if step > self.validation_steps: print()
                        validation_loss, validation_metrics, validation_outputs = self.validation_loop(loader=validation_loader, 
                                                                                                       return_outputs=return_validation_outputs, 
                                                                                                       recompute_metrics_at_end=recompute_metrics_at_end)
                        
                        self.scheduling_step(loss=validation_loss, loop="validation")

                        self.on_validation_end(step=self.passed_steps, epoch=epoch)

                        if is_wandb:
                            logs = {"validation/loss": validation_loss, 
                                    "train/loss vs validation steps": epoch_train_loss.average}

                            for metric, value in validation_metrics.items():
                                logs.update({f"validation/{metric}": value, 
                                             f"train/{metric} vs validation steps": epoch_train_metrics.average[metric]})

                            wandb.log(logs, step=self.passed_steps)

                        is_checkpoint_saved = self.model_checkpointing(loss=validation_loss, 
                                                                       metrics=validation_metrics,
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

            self.on_epoch_end(step=self.passed_steps, epoch=epoch)

            if "tqdm" in self.logger and ("print" not in self.logger and "logging" not in self.logger):
                elapsed, remain = timer(1/1)

            epoch_elapsed_seconds = timer.elapsed_time.total_seconds()
            total_time += timedelta(seconds=epoch_elapsed_seconds)

            if is_wandb:
                wandb.log({"epoch": epoch}, step=self.passed_steps)

            if "tqdm" in self.logger: train_loader.close()

            if "print" in self.logger or "logging" in self.logger:
                log_message = f"\nTraining loss: {epoch_train_loss.average:.{self.decimals}}{self.format_metrics(epoch_train_metrics.average)}"
                self.log(log_message)

                if validation_loader is not None:
                    log_message = f"Validation loss: {self.best_validation_loss:.{self.decimals}}{self.format_metrics(self.best_validation_metrics)}"
                    self.log(log_message)

                total_time_string = Timer.format_time(total_time, time_format=self.time_format)
                log_message = f"Total time: {total_time_string}"
                self.log(log_message)

        gc.collect()

        if validation_loader is not None:
            if return_validation_outputs:
                return (epoch_train_loss.average, epoch_train_metrics.average), (self.best_validation_loss, self.best_validation_metrics, self.best_validation_outputs)

            return (epoch_train_loss.average, epoch_train_metrics.average), (self.best_validation_loss, self.best_validation_metrics)

        return (epoch_train_loss.average, epoch_train_metrics.average)

        
    def log(self, message:str, end:str="\n") -> None:
        if "print" in self.logger:
            print(message, end=end)

        if "logging" in self.logger:
            self.logging_logger.debug(message)

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
            loss, outputs = self.compute_loss(batch=batch, return_outputs=True)
            targets = self.get_targets(batch)
            metrics = self.compute_metrics(predictions=outputs, targets=targets)

            if self.gradient_accumulation_steps > 1:
                loss /= self.gradient_accumulation_steps
            
            loss = self.backward_step(loss=loss)

            adversarial_loss = self.adversarial_step(batch=batch, 
                                                     loss=overall_loss, 
                                                     metrics=overall_metrics, 
                                                     step=step, 
                                                     epoch=epoch)

            if adversarial_loss is not None:
                adversarial_loss = self.backward_step(loss=adversarial_loss)

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
        """
        Applies gradient clipping for model's parameters.
        """
        if self.gradient_norm > 0:
            if self.gradient_scaling:
                self.scaler.unscale_(self.optimizer)
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_norm)
        

    def compute_loss(self, 
                      batch:Any, 
                      return_outputs:bool=True) -> torch.Tensor:
        """
        Computes loss.        
        """

        raise NotImplementedError(f"`compute_loss` function is not implemented.")

    def get_targets(self, batch:Any) -> Any:
        """
        Returns targets from batch.

        Inputs:
            batch: Any - batch of data.
        
        Returns:
            targets: Any - batch's targets. Default: [].

        """

        return []
    
    def compute_metrics(self, predictions:Any, targets:Any) -> dict:
        """
        Computes metrics

        Inputs:
            predictions: Any - outputs of model from `compute_loss`.
            targets: Any - outputs of `get_targets` function.
        
        Returns:
            metrics: dict - computed metrics. Default: {}.

        """

        return {}
    
    def model_checkpointing(self, 
                            loss:float, 
                            metrics:dict, 
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
                             loss:Optional[float]=None, 
                             metrics:Optional[dict]=None, 
                             step:int=0, 
                             epoch:int=0) -> torch.Tensor:
        pass
    
    def adversarial_step(self, 
                         batch:Any, 
                         loss:Optional[float]=None, 
                         metrics:Optional[dict]=None, 
                         step:int=0, 
                         epoch:int=0) -> torch.Tensor:
        """
        Applies Adversarial Training.
        """

        pass
    
    def validation_loop(self, 
                        loader:DataLoader, 
                        return_outputs:bool=True, 
                        recompute_metrics_at_end:bool=False) -> Tuple[Any, dict]:

        """
        Runs validation loop.
        """
        
        self.model.to(self.device)
        self.model.eval()
        loss, metrics = Averager(), Averager()
        timer = Timer(self.time_format)
        outputs, targets = [], []
        steps = len(loader)
        
        if "tqdm" in self.logger: loader = tqdm_loader_wrapper(loader, f"[Validation]")

        is_targets = False
        for step, batch in enumerate(loader, 1):
            with torch.no_grad():
                with autocast(enabled=self.amp):
                    batch_size = len(batch)
                    batch_loss, batch_outputs = self.compute_loss(batch=batch, return_outputs=True)
                    
                    batch_targets = self.get_targets(batch)
                    batch_metrics = self.compute_metrics(predictions=batch_outputs, targets=batch_targets)

                    loss.update(batch_loss.item(), n=batch_size)
                    metrics.update(batch_metrics, n=batch_size)

                    if batch_targets is not None and recompute_metrics_at_end:
                        if isinstance(batch_targets, dict):
                            targets.append(batch_targets)
                        else:
                            targets.extend(batch_targets.to("cpu").numpy())

                        is_targets = True

                    if return_outputs or recompute_metrics_at_end:
                        outputs.extend(batch_outputs.to("cpu").numpy())

                    if step == steps and recompute_metrics_at_end and is_targets:
                        outputs = torch.tensor(outputs)
                        targets = torch.tensor(targets)

                        metrics = Averager(self.compute_metrics(predictions=outputs, targets=targets))

                    if "tqdm" in self.logger:
                        loader.set_postfix_str(f"loss: {loss.average:.{self.decimals}}"
                                               f"{self.format_metrics(metrics.average)}")

                    if "print" in self.logger or "logging" in self.logger:
                        if step % self.verbose == 0 or step == steps and self.verbose > 0:
                            elapsed, remain = timer(step/steps)
                            log_message = f"[Validation] {step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {loss.average:.{self.decimals}}{self.format_metrics(metrics.average)}"
                            self.log(log_message)

        if not recompute_metrics_at_end: 
            outputs = torch.tensor(outputs)

        if "tqdm" in self.logger:
            loader.close()

        if return_outputs:
            outputs = outputs.to("cpu").numpy()
        else:
            outputs = None


        return (loss.average, metrics.average, outputs)


    def format_metrics(self, metrics:dict, sep:str=" - ", add_sep_to_start:bool=True) -> str:
        """
        Formats the given dictionary of metrics and retuns it as string.
        """

        if metrics != {}:
            string = sep.join([f"{k}: {v:.{self.decimals}}" for k, v in metrics.items()])
            return sep + string if add_sep_to_start else string 

        return ""


    def on_epoch_end(self, epoch=0, step=0):
        pass

    def on_training_step_end(self, epoch=0, step=0):
        pass

    def on_validation_end(self, epoch=0, step=0):
        pass

    def __str__(self):
        return f"""
               Trainer(scheduling_strategy={self.scheduling_strategy}, 
               gradient_accumulation_steps={self.gradient_accumulation_steps}, 
               gradient_scaling={self.gradient_scaling}, 
               gradient_norm={self.gradient_norm}, 
               amp={self.amp}, 
               scaler={self.scaler},
               verbose={self.verbose}, 
               epochs={self.epochs},
               validation_strategy={self.validation_strategy}, 
               validation_steps={self.validation_steps}, 
               device={self.device}, 
               decimals={self.decimals}, 
               logger={self.logger}, 
               time_format={self.time_format})
               """

    __repr__ = __str__