import torch
from torch.optim import Optimizer


class LARSWrapper(object):
    def __init__(
        self, optimizer, eta=0.02, clip=True, eps=1e-8, exclude_bias_n_norm=True
    ):
        """
        Wrapper that adds LARS scheduling to any optimizer. This helps stability with huge batch sizes.
        Copied from: https://github.com/Lightning-AI/lightning-bolts/blob/27855d002405967e8f31d4b33fec523d290e2a3a/pl_bolts/optimizers/lars_scheduling.py#L47

        Args:
            optimizer: torch optimizer
            eta: LARS coefficient (trust)
            clip: True to clip LR
            eps: adaptive_lr stability coefficient
            exclude_bias_n_norm : exclude bias and normalization layers from lars.
        """
        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip
        self.exclude_bias_n_norm = exclude_bias_n_norm

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group
        self.__setstate__ = self.optim.__setstate__
        self.__getstate__ = self.optim.__getstate__
        self.__repr__ = self.optim.__repr__

    @property
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            [
                self.update_p(p, group, weight_decay)
                for p in group["params"]
                if p.grad is not None and (p.ndim != 1 or not self.exclude_bias_n_norm)
            ]

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (g_norm + p_norm * weight_decay + self.eps)

            # clip lr
            if self.clip:
                new_lr = min(new_lr / group["lr"], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr
