from .print_logger import PrintLogger
import logging


class LoggingLogger(PrintLogger):
    def on_training_step_end(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_training_end(self, trainer):
        pass