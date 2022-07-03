from .logger import Logger
import wandb


class WANDBLogger(Logger):
    def __init__(self, **kwargs):
        wandb.init(**kwargs)

    def on_training_step_end(self, trainer):
        logs = {
            "train/loss": trainer.history["train_loss"], 
            "train/loss vs batch": trainer.history["train_loss_batch"], 
            "train/loss vs epoch": trainer.history["train_loss_epoch"],
            "lr": trainer.history["lr"]
        }

        train_metrics = trainer.history["train_metrics"]
        train_metrics_batch = trainer.history["train_metrics_batch"]
        train_metrics_epoch = trainer.history["train_metrics_epoch"]
        step = trainer.history["step"]

        for metric in train_metrics.keys():
            logs.update({
                f"train/{metric}": train_metrics[metric], 
                f"train/{metric} vs batch": train_metrics_batch[metric], 
                f"train/{metric} vs epoch": train_metrics_epoch[metric],
            })

        wandb.log(logs, step=step) 

    def on_validation_end(self, trainer):
        logs = {
            "validation/loss": trainer.history["validation_loss"], 
            "train/loss vs validation steps": trainer.history["train_loss_validation_steps"]
        }

        validation_metrics = trainer.history["validation_metrics"]
        train_metrics_validation_steps = trainer.history["train_metrics_validation_steps"]
        step = trainer.history["step"]

        for metric in validation_metrics.keys():
            logs.update({
                f"validation/{metric}": validation_metrics[metric], 
                f"train/{metric} vs validation steps": train_metrics_validation_steps[metric],
            })

        wandb.log(logs, step=step)

    def on_epoch_end(self, trainer):
        logs = {
            "epoch": trainer.history["epoch"]
        }
        
        step = trainer.history["step"]

        wandb.log(logs, step=step)

    def on_training_end(self, trainer):
        wandb.finish()

    def __wandb_run_exists() -> bool:
        return wandb.run is not None