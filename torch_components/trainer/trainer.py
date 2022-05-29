import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from typing import Optional, Union, Any, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import timedelta
from .utils import SchedulingStrategy, ValidationStrategy, wandb_run_exists, is_wandb_available, get_lr
from ..timer import Timer
from ..averager import Averager


if is_wandb_available():
    import wandb


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
                 device:Optional[Union[str, torch.device]]=None, 
                 validation_strategy:str="epoch",
                 validation_steps:int=1, 
                 decimals:int=4, 
                 logger:Union[str, list]="print", 
                 epochs:int=1, 
                 time_format:str="{hours}:{minutes}:{seconds}"):
        
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
            if torch.cuda.is_availabel():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device)
                
        if self.gradient_scaling and self.scaler is None:
            self.scaler = GradScaler()
    
        self.best_validation_loss, self.best_validation_metrics, self.best_validation_outputs = None, None, None
        self.lr_key = "lr"
        self.passed_steps = 0
        self.__is_scaler_called = False
    
    
    def __tqdm_loader_wrapper(self, loader:DataLoader, description:str="") -> Any:
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
        self.model.to(self.device)
        if self.teacher_model is not None: 
            self.teacher_model.to(self.device)
        
        if self.validation_strategy == ValidationStrategy.EPOCH:
            self.validation_steps = len(train_loader) * self.validation_steps

        if wandb_run_exists() and "wandb" in self.logger:
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
                step_timer = Timer(self.time_format)
                pseudo_batch = None if pseudo_loader is None else next(iter(pseudo_loader))
                
                batch_loss, batch_metrics = self.training_step(batch=batch, 
                                                               overall_loss=epoch_train_loss.average, 
                                                               overall_metrics=epoch_train_metrics.average,
                                                               step=self.passed_steps, 
                                                               epoch=epoch, 
                                                               pseudo_batch=pseudo_batch)

                lr = get_lr(self.optimizer, only_last=True, key=self.lr_key)

                if step % self.gradient_accumulation_steps == 0:
                    self.optimization_step()
                    self.__is_scaler_called = False

                    if self.scheduling_strategy == SchedulingStrategy.STEP:
                        self.scheduling_step(loop="training")

                elapsed, remain = step_timer(1/1)
                step_seconds = step_timer.elapsed_time.total_seconds()
                sample_seconds = step_seconds / batch_size

                if wandb_run_exists() and "wandb" in self.logger:
                    logs = {"train/seconds vs step": step_seconds, 
                            "train/seconds vs sample": sample_seconds}

                    wandb.log(logs, step=self.passed_steps)

                train_loss.update(batch_loss, n=batch_size)
                epoch_train_loss.update(batch_loss, n=batch_size)
                train_metrics.update(batch_metrics, n=batch_size)
                epoch_train_metrics.update(batch_metrics, n=batch_size)

                if wandb_run_exists() and "wandb" in self.logger:
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
                        validation_steps = len(validation_loader)
                        validation_batch_size = validation_loader.batch_size
                        validation_timer =  Timer(self.time_format)
                        validation_loss, validation_metrics, validation_outputs = self.validation_loop(loader=validation_loader, 
                                                                                                       return_outputs=True, 
                                                                                                       recalculate_metrics_at_end=recalculate_metrics_at_end)
                        
                        self.scheduling_step(loss=validation_loss, loop="validation")

                        elapsed, remain = validation_timer(1/1)
                        validation_seconds = validation_timer.elapsed_time.total_seconds()
                        validation_step_seconds = validation_seconds / validation_steps
                        validation_sample_seconds = validation_step_seconds / validation_batch_size

                        if wandb_run_exists() and "wandb" in self.logger:
                            logs = {"validation/seconds vs step": validation_step_seconds, 
                                    "validation/seconds vs sample": validation_sample_seconds}

                            wandb.log(logs, step=self.passed_steps)

                        if wandb_run_exists() and "wandb" in self.logger:
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
                            
                        print()

            if self.scheduling_strategy == SchedulingStrategy.EPOCH:
                self.scheduling_step(loop="training")

            if "tqdm" in self.logger and "print" not in self.logger:
                elapsed, remain = timer(1/1)

            epoch_elapsed_seconds = timer.elapsed_time.total_seconds()
            total_time += timedelta(seconds=epoch_elapsed_seconds)

            if wandb_run_exists() and "wandb" in self.logger:
                wandb.log({"epoch": epoch}, step=self.passed_steps)

            if "tqdm" in self.logger: train_loader.close()

            print(f"\nTraining loss: {epoch_train_loss.average:.{self.decimals}}"
                  f"{self.format_metrics(epoch_train_metrics.average)}")

            if validation_loader is not None:
                print(f"Validation loss: {self.best_validation_loss:.{self.decimals}}"
                      f"{self.format_metrics(self.best_validation_metrics)}")

            total_time_string = Timer.format_time(total_time, time_format=self.time_format)
            print(f"Total time: {total_time_string}")

        if validation_loader is not None:
            if return_validation_outputs:
                return (epoch_train_loss.average, epoch_train_metrics.average), (self.best_validation_loss, self.best_validation_metrics, self.best_validation_outputs)

            return (epoch_train_loss.average, epoch_train_metrics.average), (self.best_validation_loss, self.best_validation_metrics)

        return (epoch_train_loss.average, epoch_train_metrics.average)

        
    def backward_step(self, loss:torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.__is_scaler_called = True
        else:
            loss.backward()
        
        return loss
    
    def optimization_step(self) -> None:                        
        if self.scaler is not None:
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
            loss, outputs = self.calculate_loss(batch=batch, model=self.model, return_outputs=True, device=self.device)
            targets = self.get_targets(batch)
            metrics = self.calculate_metrics(predictions=outputs, targets=targets, device=self.device)

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

        self.clip_gradients()
        
        return loss.detach(), metrics
                
    def clip_gradients(self) -> None:
        if self.gradient_norm > 0:
            if self.scaler is not None and not self.__is_scaler_called:
                self.scaler.unscale_(self.optimizer)

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_norm)
        

    def calculate_loss(self, 
                      batch:Any, 
                      model:nn.Module, 
                      return_outputs:bool=True, 
                      device:Union[str, torch.device]="cpu") -> torch.Tensor:
        raise NotImplementedError(f"`calculate_loss` function is not implemented.")

    def get_targets(self, batch:Any) -> Any:
        return []
    
    def calculate_metrics(self, predictions:Any, targets:Any, device:Union[str, torch.device]="cpu") -> dict:
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
        pass
    
    def validation_loop(self, 
                        loader:DataLoader, 
                        return_outputs:bool=True, 
                        recalculate_metrics_at_end:bool=True) -> Tuple[Any, dict]:
        
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
                    
                    batch_loss /= self.gradient_accumulation_steps
                    batch_targets = self.get_targets(batch)
                    batch_metrics = self.calculate_metrics(predictions=batch_outputs, targets=batch_targets, device=self.device)

                    loss.update(batch_loss.item(), n=batch_size)
                    metrics.update(batch_metrics, n=batch_size)

                    if batch_targets is not None:
                        if isinstance(batch_targets, dict):
                            targets.append(batch_targets)
                        else:
                            targets.extend(batch_targets.to("cpu").tolist())

                        is_targets = True

                    outputs.extend(batch_outputs.to("cpu").tolist())

                    if step == steps and recalculate_metrics_at_end and is_targets:
                        outputs = torch.tensor(outputs)
                        targets = torch.tensor(targets)

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

        if not recalculate_metrics_at_end: 
            outputs = torch.tensor(outputs)

        if "tqdm" in self.logger:
            loader.close()

        return (loss.average, metrics.average, outputs) if return_outputs else (loss.average, metrics.average)


    def format_metrics(self, metrics:dict, sep:str=" - ", add_sep_to_start:bool=True) -> str:
        if metrics != {}:
            string = sep.join([f"{k}: {v:.{self.decimals}}" for k, v in metrics.items()])
            return sep + string if add_sep_to_start else string 

        return ""