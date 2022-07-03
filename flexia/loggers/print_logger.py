from .logger import Logger


class PrintLogger(Logger):
    def __init__(self, verbose:int=1, sep=" - ", decimals=4) -> None:
        super().__init__()

        self.verbose = verbose
        self.sep = sep
        self.decimals = decimals

    def on_training_step_end(self, trainer):
        step = trainer.history["epoch_step"]
        steps = trainer.history["epoch_steps"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            elapsed = trainer.history["epoch_elapsed"]
            remain = trainer.history["epoch_remain"]
            train_loss_epoch = trainer.history["train_loss_epoch"]
            train_metrics_epoch = trainer.history["train_metrics"]
            lr = trainer.history["lr"]
            
            log_message = f"{step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {train_loss_epoch:.{self.decimals}} {self.format_metrics(train_metrics_epoch)} - lr: {lr}"
            print(log_message)


    def on_validation_step_end(self, trainer):
        step = trainer.history["epoch_step"]
        steps = trainer.history["validation_steps"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            loss = trainer.history["validation_loss"]
            metrics = trainer.history["validation_metrics"]
            elapsed = trainer.history["epoch_elapsed"]
            remain = trainer.history["epoch_remain"]
            
            log_message = f"[Validation] {step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {loss:.{self.decimals}}{self.format_metrics(metrics.average)}"
            print(log_message)


    def on_epoch_start(self, trainer):
        epoch = trainer.history["epoch"]
        epochs = trainer.history["epochs"]

        print(f"Epoch {epoch}/{epochs}")

    def format_metrics(self, metrics:dict) -> str:
        if metrics != {}:
            string = self.sep.join([f"{k}: {v:.{self.decimals}}" for k, v in metrics.items()])
            return string
            
        return ""

