class Callback:    
    def state_dict(self) -> dict:
        pass
    
    def load_state_dict(self, state_dict:dict):
        pass

    def on_training_step_start(self, trainer):
        pass

    def on_training_step_end(self, trainer):
        pass

    def on_validation_step_start(self, trainer):
        pass

    def on_validation_step_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_validation_start(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_training_start(self, trainer):
        pass

    def on_training_end(self, trainer):
        pass

    def on_prediction_step_start(self, inferencer):
        pass

    def on_prediction_step_end(self, inferencer):
        pass

    def on_prediction_start(self, inferencer):
        pass

    def on_prediction_end(self, inferencer):
        pass


    def get_monitor_value(self, trainer):
        monitor_value = trainer.history.get(self.monitor_value)
        if monitor_value is None:
            possible_monitor_values = list(trainer.history.keys())
            raise KeyError(f"{self.monitor_value} is not in History of Trainer. Please choose one of {possible_monitor_values}")

        return monitor_value