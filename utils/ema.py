# utils/ema.py
import copy
import torch

class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Keeps EMA parameters in same device as model by default.
    decay: float in (0,1), e.g., 0.9999
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        # copy model for ema
        self.ema = copy.deepcopy(model).eval()
        # disable grad for ema
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # model: the current training model
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(self.decay).add_(msd[k].to(esd[k].device), alpha=1.0 - self.decay)
            else:
                esd[k] = msd[k].to(esd[k].device)
        self.ema.load_state_dict(esd)

    def state_dict(self):
        return self.ema.state_dict()

    def to(self, device):
        self.ema.to(device)
        self.device = device
