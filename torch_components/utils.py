import torch
import numpy as np
import transformers as T
from typing import Any



def to_tensor(input:Any):
    if not isinstance(input, torch.Tensor):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        else:
            input = torch.tensor(input)
        
    return input


def to_tensors(*inputs:Any):
    return tuple(map(lambda input: to_tensor(input), inputs))