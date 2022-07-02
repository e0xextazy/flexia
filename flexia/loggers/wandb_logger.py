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

        for metric in trainer.history["train_metrics_list"]:
            logs.update({
                f"train/{metric}": trainer.history[f"train_{metric}"], 
                f"train/{metric} vs batch": trainer.history[f"train_{metric}_batch"], 
                f"train/{metric} vs epoch": trainer.history[f"train_{metric}_epoch"]
            })

        wandb.log(logs, step=trainer.history["step"]) 

    def on_validation_end(self, trainer):
        logs = {
            "validation/loss": trainer.history["validation_loss"], 
            "train/loss vs validation steps": trainer.history["train_loss_validation_steps"]
        }

        for metric in trainer.history["validation_metrics_list"]:
            logs.update({
                f"validation/{metric}": trainer.history[f"validation_{metric}"], 
                f"train/{metric} vs validation steps": trainer.history[f"train_{metric}_epoch"],
            })

        wandb.log(logs, step=trainer.history["step"])

    def on_epoch_end(self, trainer):
        logs = {
            "epoch": trainer.history["epoch"]
            }
        wandb.log(logs, step=trainer.history["step"])

    def __wandb_run_exists() -> bool:
        return wandb.run is not None
