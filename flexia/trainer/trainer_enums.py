from ..enums import ExplicitEnum
from enum import Enum


class ValidationStrategy(ExplicitEnum):
    EPOCH = "epoch"
    STEP = "step"
    
    
class SchedulingStrategy(ExplicitEnum):
    EPOCH = "epoch"
    STEP = "step"

class TrainingStates(Enum):
    INIT = "on_init"
    TRAINING_START = "on_training_start"
    TRAINING_END = "on_training_end"
    TRAINING_STEP_START = "on_training_step_start"
    TRAINING_STEP_END = "on_training_step_end"
    VALIDATION_START = "on_validation_start"
    VALIDATION_END = "on_validation_end"
    EPOCH_START = "on_epoch_start"
    EPOCH_END = "on_epoch_end"