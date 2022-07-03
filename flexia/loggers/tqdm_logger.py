from tqdm import tqdm

from .logger import Logger



class TQDMLogger(Logger):
    def __init__(self, 
                 bar_format="{l_bar} {bar} {n_fmt}/{total_fmt} - elapsed: {elapsed} - remain: {remaining}{postfix}", 
                 description="",
                 color="#000", 
                 decimals=4, notebook=False):
        
        self.bar_format = bar_format
        self.description = description
        self.color = color
        self.decimals = decimals
        self.notebook = notebook

    def on_epoch_start(self, trainer):
        epoch = trainer.history["epoch"]
        epochs = trainer.history["epochs"]

        description = f"Epoch {epoch}/{epochs}"
        trainer.train_loader = self.__loader_wrapper(loader=trainer.train_loader, description=description)

    def on_validation_start(self, trainer):
        description = "Validation"
        trainer.validation_loader = self.__loader_wrapper(loader=trainer.validation_loader, description=description)

    def on_training_step_end(self, trainer):
        train_loss_epoch = trainer.history["train_loss_epoch"] 
        train_metric_epoch = trainer.history["train_metrics_epoch"]

        string = f"loss: {train_loss_epoch:.{self.decimals}}{self.format_metrics(train_metric_epoch)}"
        trainer.train_loader.set_postfix_str(string)

    def on_validation_step_end(self, trainer):
        validation_loss = trainer.history["validation_loss"]
        validation_metrics = trainer.history["validation_metrics"]

        string = f"loss: {validation_loss:.{self.decimals}}{self.format_metrics(validation_metrics)}"
        trainer.validation_loader.set_postfix_str(string)

    def on_training_end(self, trainer):
        trainer.train_loader.close()

    def on_validation_end(self, trainer):
        trainer.validation_loader.close()

    def __loader_wrapper(self, loader):
        if self.notebook:
            from tqdm.notebook import tqdm

    
        steps = len(loader)
        loader = tqdm(iterable=loader, 
                      total=steps,
                      colour=self.color,
                      bar_format=self.bar_format)

        loader.set_description_str(self.description)

        return loader