from distutils.log import warn
from logging import warning
import torch
from torch import nn
from arguments import Arguments
import os
from utils import seed_everything, get_optimizer, get_scheduler
from torch import optim
from torch.optim import Optimizer
import random
import warnings


class Trainer:
    def __init__(self, model, optimizer, criterion,  scheduler=None, model_parameters=None, arguments=Arguments(), logger=print):
        if isinstance(optimizer, dict):
            if model_parameters is None:
                model_parameters = model.parameters()

            optimizer = get_optimizer(model_parameters=model_parameters, **optimizer)

        if scheduler is not None:
            if isinstance(scheduler, dict):
                scheduler = get_scheduler(optimizer=optimizer, **scheduler)

        
        if self.arguments.seed == "random":
            self.arguments.seed = random.randint(a=0, b=1000)
            warnings.warn("Setting the seed as 'random' maybe results in non-reproducible experiments.", category=warnings.WarningMessage)

        if self.arguments.save_scheduler_state and (scheduler is None):
            if not self.arguments.ignore_warnings:
                warnings.warn("The 'save_scheduler_state' parameter in Arguments was setted to the 'True', but the scheduler isn't setted, thereby  'save_scheduler_state' will be ignored.")
            
        

            
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.arguments = arguments
        self._criterion_name = self.criterion.__class__.__name__

        seed = seed_everything(self.arguments.seed)
        print(f"Seed: {seed}")



    def __debug(self, message):
        if self.arguments.debug:
            print(message)


    def collate_batch(self, batch, inputs_device="cpu", targets_device="cpu"):
        if len(batch) == 2:
            inputs, targets = batch
            inputs = inputs.to(inputs_device)
            targets = targets_device.to(targets_device)

            return inputs, targets
        else:
            inputs = batch
            inputs = inputs.to(inputs_device)

            return inputs


    def calculate_loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return loss


    def save_checkpoint(self, path, loss=None, iteration=None, epoch=None):
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss,
            "iteration": iteration,
            "epoch": epoch,
        }

        if (self.scheduler is not None) and self.arguments.save_scheduler_state:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        
        return checkpoint


    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        model_state = checkpoint["model_state"]
        self.model.load_state_dict(model_state)

        optimizer_state = checkpoint["optimizer_state"]
        self.optimizer.load_state_dict(optimizer_state)

        if ("scheduler" in checkpoint) and (self.scheduler is not None):
            scheduler_state = checkpoint["scheduler_state"]
            self.scheduler.load_state_dict(scheduler_state)

        return checkpoint


    def validate(self, loader):
        loss = 0
        iterations = len(loader)
        for batch in loader:
            with torch.no_grad():
                inputs, targets = batch
                outputs = self.model(inputs)
                batch_loss = self.calculate_loss(outputs, targets)
                loss += batch_loss.item()

        loss /= iterations
        
        return loss


    def apply_gradient_clipping(self, parameters):
        pass


    def train_one_batch(self, batch):
        self.model.zero_grad()
        self.model.train()
        
        inputs, targets = self.collate_batch(batch, inputs_device=self.arguments.device, targets_device="cpu")
        outputs = self.model(inputs)
        batch_loss = self.calculate_loss(outputs, targets)


        batch_loss.backward()
        self.optimizer.step()

        return batch_loss.detach()


    def fit(self, train_loader, validation_loader=None):
        iterations_per_epoch = len(train_loader) 
        iterations = int(iterations_per_epoch * self.arguments.epochs)
        passed_iterations = 1

        if self.arguments.validation_strategy == "epoch":
            self.arguments.validation_steps = iterations_per_epoch
            if not self.arguments.ignore_warnings:
                warnings.warn("When setting 'validation_strategy' equal to 'epoch', the 'validation_steps' parameter wiil be ignored.", category=warnings.WarningMessage)

        self.model.train()
        self.model.to(self.arguments.device)
        for epoch in range(1, self.arguments.epochs+1):
            epoch_loss = 0
            for batch in train_loader:
                train_loss = self.train_one_batch(batch)
                
                if (passed_iterations % self.arguments.validation_steps) == 0:
                    validation_loss = self.validate(validation_loader)
                    self.__debug(f"Iteration [{passed_iterations}/{iterations}] Validation Loss: {validation_loss}")

                epoch_loss += train_loss
                passed_iterations += 1

            epoch_loss /= iterations_per_epoch


        self.__debug(f"Epoch [{epoch}/{self.arguments.epochs}] Train Loss: {epoch_loss}")
