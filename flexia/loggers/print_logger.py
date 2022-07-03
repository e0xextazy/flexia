from .logger import Logger


class PrintLogger(Logger):
    def __init__(self, verbose:int=1, decimals=4) -> None:
        super().__init__()

        self.verbose = verbose
        self.decimals = decimals

    def on_training_step_end(self, trainer):
        step = trainer.history["step_epoch"]
        steps = trainer.history["steps_epoch"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            elapsed = trainer.history["elapsed_epoch"]
            remain = trainer.history["remain_epoch"]
            train_loss_epoch = trainer.history["train_loss_epoch"]
            train_metrics_epoch = trainer.history["train_metrics"]
            lr = trainer.history["lr"]
            
            log_message = f"{step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {train_loss_epoch:.{self.decimals}} {self.format_metrics(train_metrics_epoch)} - lr: {lr}"
            print(log_message)


    def on_validation_step_end(self, trainer):
        step = trainer.history["epoch_step"]
        steps = trainer.history["steps_validation"]

        if step % self.verbose == 0 or step == steps and self.verbose > 0:
            loss = trainer.history["validation_loss"]
            metrics = trainer.history["validation_metrics"]
            elapsed = trainer.history["elapsed_epoch"]
            remain = trainer.history["remain_epoch"]
            
            log_message = f"[Validation] {step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {loss:.{self.decimals}}{self.format_metrics(metrics.average)}"
            print(log_message)


    def on_epoch_start(self, trainer):
        epoch = trainer.history["epoch"]
        epochs = trainer.history["epochs"]
        log_message = f"Epoch {epoch}/{epochs}"
        print(log_message)

    def format_metrics(self, metrics:dict) -> str:
        if metrics != {}:
            string = self.sep.join([f"{k}: {v:.{self.decimals}}" for k, v in metrics.items()])
            return string
            
        return ""