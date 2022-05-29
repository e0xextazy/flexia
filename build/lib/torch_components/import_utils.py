def is_transformers_available():
    try:
        import transformers
        return True
    except ModuleNotFoundError:
        return False


def is_wandb_available():
    try:
        import wandb
        return True
    except ModuleNotFoundError:
        return False
    
def wandb_run_exists():
    if is_wandb_available(): 
        import wandb
        return wandb.run is not None
    else:
        return False