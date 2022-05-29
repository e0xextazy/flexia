from enum import Enum


class ExplicitEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        keys = list(cls._value2member_map_.keys())
        raise ValueError(f"`{value}` is not a valid `{cls.__name__}`, select one of `{keys}`.")
        

class ValidationStrategy(ExplicitEnum):
    EPOCH = "epoch"
    STEP = "step"
    
    
class SchedulingStrategy(ExplicitEnum):
    EPOCH = "epoch"
    STEP = "step"