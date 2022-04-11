def is_transformers_available():
    try:
        import transformers
        return True
    except ModuleNotFoundError:
        return False


def is_addict_available():
    try:
        import addict
        return True
    except ModuleNotFoundError:
        return False