from ..enums import ExplicitEnum


class ValidationStrategy(ExplicitEnum):
    EPOCH = "epoch"
    STEP = "step"
    
    
class SchedulingStrategy(ExplicitEnum):
    EPOCH = "epoch"
    STEP = "step"