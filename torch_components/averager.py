from typing import Union, Optional


class Averager:
    def __init__(self, average:Union[int, float]=0, sum_:Union[int, float]=0, count:int=0, value:Optional[Union[float, int]]=None):
        """
        Computes statistics (sum, average, and count) for given values. 
        
        average: Union[int, float] - average across all input values. Default: 0.
        sum: Union[int, float] - sum across all input values. Default 0.
        count: int - count of input values. Default: 0.
        value: Optional[Union[float, int]] - previous value. Default: None.
        
        """
        
        self.average = average
        self.sum = sum_
        self.count = count
        self.value = value
        
    def state_dict(self) -> dict:
        state = {
            "average": self.average,
            "sum": self.sum,
            "count": self.count,
            "value": self.value,
        }
        
        return state
    
    def load_state_dict(self, state_dict) -> "Averager":
        self.average = state_dict["average"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]
        self.value = state_dict["value"]
        
        return self
        
    
    def reset(self) -> None:
        """
        Resets all stored values.
        """
        self.average = 0
        self.sum = 0
        self.count = 0
        self.value = None
        
    
    def update(self, value:Union[int, float], n:int=1) -> None:
        """
        Updates statistics (average, count and sum).
        """
        
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
        
    
    def __str__(self) -> str:
        return f"Averager(average={self.average}, sum={self.sum}, count={self.count}, value={self.value})"
    
    __repr__ = __str__