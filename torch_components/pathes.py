import warnings
import os
from .configuration import Configuration
from .import_utils import is_addict_available


if is_addict_available():
    from addict import Dict


class Pathes(Configuration):
    """
    Comfortable interface to keep your pathes in one place.
    Also it will send you warnings if some pathes are not found.
    """

    def __init__(self, *args, **kwargs:dict) -> None:
        if len(args) != 0:
            raise ValueError(f"You must put attributes in format key=value.")

        self.__attributes = Dict(kwargs) if is_addict_available() else kwargs 
        
        for k, v in self.__attributes.items():
            if not os.path.exists(v):
                warnings.warn(f"`{k}` path is not found. Correct it to avoid further exceptions.")

    def __str__(self) -> str:
        attributes_string = ", ".join([f"{k}={v}" for k, v in self.__attributes.items()])
        return f"Pathes({attributes_string})"

    __repr__ = __str__