from .logger import Logger


class PrintLogger(Logger):
    def __init__(self, verbose:int=1, sep=" - ", decimals=4) -> None:
        super().__init__()

        self.verbose = verbose
        self.sep = sep
        self.decimals = decimals

    def on_training_step_end(self, trainer):
        if "print" in self.loggers or "logging" in self.loggers:
            if step % self.verbose == 0 or step == steps and self.verbose > 0:
                log_message = f"{step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {epoch_train_loss.average:.{self.decimals}} {self.format_metrics(epoch_train_metrics.average)} - lr: {lr}"
                print(log_message)


    def on_validation_step_end(self, trainer):
        if "print" in self.loggers or "logging" in self.loggers:
            if step % self.verbose == 0 or step == steps and self.verbose > 0:
                elapsed, remain = timer(step/steps)
                log_message = f"[Validation] {step}/{steps} - elapsed: {elapsed} - remain: {remain} - loss: {loss.average:.{self.decimals}}{self.format_metrics(metrics.average)}"
                self.log(log_message)



    def on_epoch_start(self, trainer):
        pass

    def on_training_end(self, trainer):
        if "print" in self.loggers or "logging" in self.loggers:
                log_message = f"\nTraining loss: {epoch_train_loss.average:.{self.decimals}}{self.format_metrics(epoch_train_metrics.average)}"
                self.log(log_message)

                if self.validation_loader is not None:
                    log_message = f"Validation loss: {self.best_validation_loss:.{self.decimals}}{self.format_metrics(self.best_validation_metrics)}"
                    self.log(log_message)

                total_time_string = Timer.format_time(total_time, time_format=self.time_format)
                log_message = f"Total time: {total_time_string}"
                self.log(log_message)

    
    def format_metrics(self, metrics:dict) -> str:
        if metrics != {}:
            string = self.sep.join([f"{k}: {v:.{self.decimals}}" for k, v in metrics.items()])
            return string
            
        return ""

