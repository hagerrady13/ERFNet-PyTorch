import torch

def load_my_state_dict(model, state_dict):
    """
    # custom function to load model when not all dict keys are there
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    return model