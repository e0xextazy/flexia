from .configuration import Configuration
import warnings
import os


class Pathes(Configuration):
    """
    Comfortable interface to keep your pathes in one place.
    Also it will send you warnings if some pathes are not found.
    """

    def __post_init__(self):
        for k, v in self.attributes.items():
            if not os.path.exists(v):
                warnings.warn(f"Attribute `{k}` path is not found. Correct it to avoid further exceptions.")