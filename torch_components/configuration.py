from collections import UserDict
import json
from typing import Any
from addict import Dict
from .import_utils import is_addict_available

if is_addict_available():
    from addict import Dict


class Configuration:
    """
    Configuration provides you comfortable interface of saving your variables.
    """
    
    def __init__(self, *args, **kwargs:dict) -> None:
        if len(args) != 0:
            raise ValueError(f"You must put attributes in format key=value.")
        
        self.__attributes = Dict(kwargs) if is_addict_available() else kwargs 


    def get_attributes_names(self) -> list:
        return list(self.__attributes.keys())
                
    def __getattr__(self, attribute:str) -> Any:
        if attribute not in self.__attributes:
            attributes_names = self.get_attributes_names()
            raise ValueError(f"Given attribute `{attribute}` is not setted in Configuration. Choose one of {attributes_names}.")
        
        return self.__attributes[attribute]
    
    def __setitem__(self, attribute:str, value:Any) -> None:
        self.__attributes[attribute] = value
    
    def to_json_string(self) -> str:
        return json.dumps(self.__attributes)
    
    def to_json(self, path:str) -> str:
        with open(path, "w", encoding="utf-8") as file:
            data = self.to_json_string()
            file.write(data)
        
        return path
    
    def from_json_string(self, string:str) -> dict:
        return json.loads(string)
    
    def from_json(self, path:str) -> "Configuration":
        with open(path, "r", encoding="utf-8") as file:
            data = self.from_json_string(file.read())
        
        for k, v in data.items():
            self.__attributes[k] = v
            
        return self

    def __str__(self) -> str:
        attributes_string = ", ".join([f"{k}={v}" for k, v in self.__attributes.items()])
        return f"Configuration({attributes_string})"
    
    def __contains__(self, attribute:str):
        return attribute in self.__attributes
        
    def get(self, attribute:str, default:Any) -> Any:
        return self.__attributes.get(attribute, default)

    def to_dict(self):
        return self.__attributes


    __repr__ = __str__
    __getitem__ = __getattr__