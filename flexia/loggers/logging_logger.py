from .print_logger import PrintLogger
import logging


def get_logger(name:str=__name__, 
               format:str="[%(asctime)s][%(levelname)s]: %(message)s", 
               filename:Optional[str]=None) -> logging.Logger:
               
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(format)

    if filename is not None:
        file_handler = logging.FileHandler(name)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    
    logger.propagate = False
    
    return logger

class LoggingLogger(PrintLogger):
    def on_training_step_end(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_training_end(self, trainer):
        pass