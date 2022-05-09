from transformers import PreTrainedTokenizer
from typing import Any
import numpy as np


def convert_ids_to_string(ids:Any, tokenizer:PreTrainedTokenizer) -> str:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    string = tokenizer.convert_tokens_to_string(tokens)
    
    return string


def create_probability_mask(inputs:Any, probability:float=0.5) -> np.ndarray:
    probability_matrix = np.random.uniform(low=0, high=1, size=inputs.shape)
    return (probability_matrix < probability).astype(bool)