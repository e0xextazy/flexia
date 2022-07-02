from ..enums import ExplicitEnum
from enum import Enum

class ValidationStrategy(ExplicitEnum):
    EPOCH = "epoch"
    STEP = "step"
    
    
class SchedulingStrategy(ExplicitEnum):
    EPOCH = "epoch"
    STEP = "step"

class TrainingStates(Enum):
    STOP = "stop"
    CONTINUE = "continue"