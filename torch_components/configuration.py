import json
from .third_party.addict import Dict


class Configuration(Dict):
    """
    Configuration provides you comfortable interface of saving your variables.
    """
    
    def to_json_string(self) -> str:
        return json.dumps(self)
    
    def to_json(self, path:str) -> str:
        with open(path, "w", encoding="utf-8") as file:
            data = self.to_json_string()
            file.write(data)
        
        return path
    
    def from_json_string(self, string:str) -> dict:
        return json.loads(string)
    
    def from_json(self, path:str) -> "Configuration":
        with open(path, "r", encoding="utf-8") as file:
            self = Configuration(self.from_json_string(file.read()))     
        
        return self

    def __str__(self) -> str:
        attributes_string = ", ".join([f"{k}={v}" for k, v in self.items()])
        return f"Configuration({attributes_string})"

    __repr__ = __str__