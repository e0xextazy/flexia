def is_transformers_available() -> bool:
    """
    Checks the availablity of `transformers` library.
    """

    try:
        import transformers
        return True
    except ModuleNotFoundError:
        return False


def is_wandb_available() -> bool:
    """
    Checks the availablity of `wandb` library.
    """

    try:
        import wandb
        return True
    except ModuleNotFoundError:
        return False
    
def wandb_run_exists() -> bool:
    """
    Checks the availablity of Weighs & Biases run.
    """
    if is_wandb_available(): 
        import wandb
        return wandb.run is not None
    else:
        return False


def is_torch_xla_available() -> bool:
    """
    Checks the availablity of `torch_xla` library.    
    """
    
    try:
        import torch_xla
        return True
    except ModuleNotFoundError:
        return False


def is_bitsandbytes_available() -> bool:
    """
    Checks the availablity of `bitsandbytes` library.    
    """
    
    try:
        import bitsandbytes
        return True
    except ModuleNotFoundError:
        return False