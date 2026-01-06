# utils/ema.py
import copy
import torch

class ModelEMA:
    """
    Maintain EMA of model parameters.
    state_dict() returns a dict containing ema parameters for saving.
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        # copy model
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.to(device)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if v.dtype.is_floating_point and k in msd:
                # perform in-place update on ema parameter
                v.mul_(self.decay).add_(msd[k].to(v.device), alpha=1.0 - self.decay)
            else:
                # copy exact (e.g., buffers / ints)
                esd[k] = msd[k].to(v.device)
        self.ema.load_state_dict(esd)

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "decay": self.decay}

    def load_state_dict(self, state):
        if "ema" in state:
            self.ema.load_state_dict(state["ema"])
        if "decay" in state:
            self.decay = state["decay"]

    def to(self, device):
        self.ema.to(device)
        self.device = device
