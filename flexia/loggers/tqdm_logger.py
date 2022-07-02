from tqdm import tqdm

from .logger import Logger



class TQDMLogger(Logger):
    def __init__(self, 
                 bar_format="{l_bar} {bar} {n_fmt}/{total_fmt} - elapsed: {elapsed} - remain: {remaining}{postfix}", 
                 description="",
                 color="#000", 
                 decimals=4):
        
        self.bar_format = bar_format
        self.description = description
        self.color = color
        self.decimals = decimals

    def on_epoch_start(self, trainer):
        epoch = trainer.history["epoch"]
        epochs = trainer.history["epochs"]

        trainer.train_loader = self.__loader_wrapper(loader=trainer.train_loader, description=f"Epoch {epoch}/{epochs}")

    def on_validation_start(self, trainer):
        trainer.validation_loader = self.__loader_wrapper(loader=trainer.validation_loader, description="Validation")

    def on_training_step_end(self, trainer):
        epoch_train_loss = trainer.history["epoch_train_loss"] 
        epoch_train_metrics = trainer.history["train_metrics_list"]

    
        string = f"loss: {epoch_train_loss:.{self.decimals}}{self.format_metrics(epoch_train_metrics)}"
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
        steps = len(loader)
        loader = tqdm(iterable=loader, 
                      total=steps,
                      colour=self.color,
                      bar_format=self.bar_format)

        loader.set_description_str(self.description)

        return loader