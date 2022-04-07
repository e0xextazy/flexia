import torch
from trainer.arguments import Arguments
from utils import seed_everything, get_optimizer, get_scheduler
from callbacks import Callbacks
import random
import warnings
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import logging


class Trainer:    
    def __init__(self, model, optimizer, criterion, metric=None, scheduler=None, model_parameters=None, arguments=Arguments(), callbacks=Callbacks(), logger=logging.getLogger(__name__)):
        if isinstance(optimizer, dict):
            if model_parameters is None:
                model_parameters = model.parameters()

            optimizer = get_optimizer(model_parameters=model_parameters, **optimizer)

        if scheduler is not None:
            if isinstance(scheduler, dict):
                scheduler = get_scheduler(optimizer=optimizer, **scheduler)

        
        if arguments.seed == "random":
            arguments.seed = random.randint(a=0, b=1000)
            if not arguments.ignore_warnings:
                warnings.warn("Setting the seed as 'random' maybe results in non-reproducible experiments.")
        elif arguments.seed == "none":
            if not arguments.ignore_warnings:
                warnings.warn("Setting the seed as 'none' maybe results in non-reproducible experiments, so you should care that you setted it before calling 'fit()' method.")

        if arguments.save_scheduler_state and (scheduler is None):
            if not arguments.ignore_warnings:
                warnings.warn("The 'save_scheduler_state' parameter in Arguments was setted to the 'True', but the scheduler isn't setted, thereby  'save_scheduler_state' will be ignored.")
        
        if not isinstance(callbacks, Callbacks):
            callbacks = Callbacks(callbacks)
            
        
            
            
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.scheduler = scheduler
        self.arguments = arguments
        self.callbacks = callbacks
        self.logger = logger
        self.scaler = None
        
        
        if self.arguments.gradient_scaling:
            if isinstance(self.arguments.gradient_scaling, dict):
                self.scaler = GradScaler(**self.arguments.gradient_scaling)
            else:
                self.scaler = GradScaler()    
                    
    def validate(self, loader):
        loss, metrics = 0, None
        iterations = len(loader)
        for batch in loader:
            with torch.no_grad():
                with self.autocast_context_manager():
                    batch_loss, batch_outputs = self.calculate_loss(batch, return_outputs=True)
                    batch_metrics = self.calculate_metrics(batch, batch_outputs)
                    
                loss += batch_loss.item()
                
        loss /= iterations
        
        return loss, metrics
    
    def calculate_loss(self, batch, return_outputs=True):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        return (loss, outputs) if return_outputs else loss
           
    
    def calculate_metrics(self, batch, outputs):
        pass
    
    
    def format_metrics(self, metrics, sep=" "):
        string = ""
        if metrics is not None:
            string = sep.join([f"{k}: {v}" for k, v in metrics.items()])
        
        return string
    
    
    def training_step(self, batch):
        self.model.train()
        
        with self.autocast_context_manager():
            loss, outputs = self.calculate_loss(batch, return_outputs=True)
            metrics = self.calculate_metrics(batch, outputs)
        
        loss /= self.arguments.gradient_accumulation_steps
    
        if self.arguments.gradient_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        
        return loss.detach(), metrics
    
    def autocast_context_manager(self):
        if self.arguments.amp:
            context = autocast()
        else:
            context = nullcontext()
            
        return context
    

    def fit(self, train_loader, validation_loader=None):
        if self.arguments.seed is not None:
            seed = seed_everything(self.arguments.seed)
        
        steps_per_epoch = len(train_loader) 
        iterations = int(steps_per_epoch * self.arguments.epochs)
        passed_iterations = 1
        
        if self.arguments.debug:
            self.logger.info(f"Epochs: {self.arguments.epochs}")
            self.logger.info(f"Steps per epoch: {steps_per_epoch}")
            self.logger.info(f"Gradient accumulation steps: {self.arguments.gradient_accumulation_steps}")
            self.logger.info(f"Gradient norm: {self.arguments.gradient_norm}")
            self.logger.info(f"Auto Mixed Precision: {self.arguments.amp}")
            self.logger.info(f"Seed: {self.arguments.seed}")
        
        if self.arguments.validation_strategy == "epoch":
            self.arguments.validation_steps = steps_per_epoch
            if not self.arguments.ignore_warnings:
                warnings.warn("When setting 'validation_strategy' equals to 'epoch', the 'validation_steps' parameter will be ignored.")
        
        if self.arguments.debug:
            self.logger.info(f"Validation steps: {self.arguments.validation_steps}")
        
        print()
        
        self.logger.info("Start of training.")
        self.model.zero_grad()
        self.model.to(self.arguments.device)
        for epoch in range(1, self.arguments.epochs+1):
            epoch_loss = 0
            for batch in train_loader:
                train_loss, train_metrics = self.training_step(batch)
                train_metrics_string = self.format_metrics(train_metrics)
                
                if passed_iterations % self.arguments.gradient_accumulation_steps == 0:
                    if self.arguments.gradient_norm > 0:
                        if self.arguments.gradient_scaling:
                            self.scaler.unscale_(self.optimizer)
                            
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.arguments.gradient_norm)
                        
                    if self.arguments.gradient_scaling:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.model.zero_grad()
                
                if self.arguments.verbose != "epoch":
                    if (passed_iterations % self.arguments.verbose) == 0:
                        self.logger.info(f"Epoch [{epoch}/{self.arguments.epochs}] Iteration [{passed_iterations}/{iterations}] Train Loss: {train_loss} {train_metrics_string}")
                
                if validation_loader is not None:
                    if (passed_iterations % self.arguments.validation_steps) == 0:
                        validation_loss, validation_metrics = self.validate(validation_loader)
                        validation_metrics_string = self.format_metrics(validation_metrics)
                        self.logger.info(f"Epoch [{epoch}/{self.arguments.epochs}] Iteration [{passed_iterations}/{iterations}] Validation Loss: {validation_loss} {validation_metrics_string}")
                    

                epoch_loss += train_loss
                passed_iterations += 1

            epoch_loss /= steps_per_epoch

            self.logger.info(f"Epoch [{epoch}/{self.arguments.epochs}] Train Loss: {epoch_loss}")
            print()
        
        if self.arguments.save_model:
            torch.save(self.model.state_dict(), self.arguments.model_path)
            self.logger.info(f"Model's weights saved to '{self.arguments.model_path}'.")
    
        
        self.logger.info("End of training")
        
    def __str__(self):
        return f"Trainer(arguments={str(self.arguments)})"

    
    __repr__ = __str__
