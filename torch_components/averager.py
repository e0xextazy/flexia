from typing import Union, Optional
from copy import deepcopy
from .third_party.addict import Dict


class Averager:
    def __init__(self, 
                average:Union[int, float, dict]=0, 
                sum_:Union[int, float, dict]=0, 
                count:int=0, 
                value:Optional[Union[float, int, dict]]=None):
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
        self.__calls = 0
        
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
        

    def sum_over_dictionary(self, input:dict, other:dict, n:int=1) -> dict:
        input = deepcopy(input)
        for k, v in other.items():
            input[k] += v * n
        
        return input

    def average_over_dictionary(self, input:dict, n:int=1) -> dict:
        input = deepcopy(input)
        for k in input:
            input[k] /= n

        return input

    
    def update(self, value:Union[int, float, dict], n:int=1) -> None:
        """
        Updates statistics (average, count and sum).
        """

        if isinstance(value, dict):
            self.count += n
            if self.__calls == 0:
                self.sum = self.sum_over_dictionary(value, value, n=n-1)
                self.average = self.average_over_dictionary(self.sum, n=self.count)
            else:
                self.sum = self.sum_over_dictionary(self.sum, value, n=n)
                self.average = self.average_over_dictionary(self.sum, n=self.count)

            self.sum = Dict(self.sum)
            self.average = Dict(self.average)
            self.value = Dict(value)

        else:
            self.value = value
            self.sum += value * n
            self.count += n
            self.average = self.sum / self.count

        self.__calls += 1
        
    
    def __str__(self) -> str:
        return f"Averager(average={self.average}, sum={self.sum}, count={self.count}, value={self.value})"
    
    __repr__ = __str__