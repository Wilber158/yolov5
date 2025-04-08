import torch
import os

def loadWeights(model, restore_path, map_location)
    state = torch.load(os.path.join(restore_path),
                       map_location=map_location)

    from collections import OrderedDict

    new_state_dict = OrderedDict()

    for k, v in state['net'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model