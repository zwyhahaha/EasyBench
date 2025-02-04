import torch
from torch.optim.optimizer import Optimizer

import numpy as np

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("HDM",)

class HDM(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: OptFloat = None,
        beta_lr: OptFloat = 1.0,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        relax_coef: float = 1.0,
        dampening: float = 0.0,
        P_version: str = "diag",
        beta_version: str = "scalar",
        monotone_epoch: int = -1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr,eps=eps,
                        weight_decay=weight_decay,
                        beta_lr=beta_lr, relax_coef=relax_coef,
                        dampening=dampening, P_version=P_version,
                        beta_version=beta_version, monotone_epoch=monotone_epoch)
        super(HDM,self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None, epoch = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                weight_decay = group["weight_decay"]
                if weight_decay != 0:
                    grad = p.grad.data.add(p.data, alpha=weight_decay)
                else:
                    grad = p.grad.data

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(p)

                    if group["P_version"] == 'diag':
                        state["P"] = torch.zeros_like(p)
                        state["G"] = torch.zeros_like(p)
                    elif group["P_version"] == 'scalar':
                        state["P"] = 0.0
                        state["G"] = 0.0
                    else:
                        raise ValueError(f"Invalid P_version: {group['P_version']}")
                    
                    if group["beta_version"] == 'diag':
                        state["beta"] = torch.zeros_like(p) + 0.95
                        state["Gm"] = torch.zeros_like(p)
                    elif group["beta_version"] == 'scalar':
                        state["beta"] = 0.95
                        state["Gm"] = 0.0
                    else:
                        raise ValueError(f"Invalid beta_version: {group['beta_version']}")
                    
                else:
                    state["step"] += 1
                    prev_grad = state["prev_grad"]
                    m = state["m"]

                    eps = group["eps"]
                    dampening = group["dampening"]
                    lr = group["lr"]
                    beta_lr = group["beta_lr"]
                    monotone_epoch = group["monotone_epoch"]
                    
                    momentum_normalizer = m.norm()**2/(lr)
                    if group["P_version"] == "diag":
                        gr = -(1-dampening) * prev_grad.mul(grad) / (prev_grad.norm()**2 + momentum_normalizer)
                        state["G"].addcmul_(gr, gr, value=1)
                        state["P"].addcdiv_(gr, state["G"].add(eps).sqrt(), value=-lr)
                    elif group["P_version"] == "scalar":
                        gr = -(1-dampening) * (prev_grad * grad).sum() / (prev_grad.norm()**2 + momentum_normalizer)
                        state["G"] += gr ** 2
                        state["P"] -= lr * gr / (state["G"].add(eps).sqrt())
                    
                    if group["beta_version"] == "diag":
                        gm = -(1-dampening) * prev_grad.mul(m) / (prev_grad.norm()**2 + momentum_normalizer)
                        state["Gm"].addcmul_(gm, gm, value=1)
                        state["beta"].addcdiv_(gm, state["Gm"].add(eps).sqrt(), value=-beta_lr * lr)
                        state["beta"].clamp_(0.0,0.9995)
                    elif group["beta_version"] == "scalar":
                        gm = (1-dampening) * (grad * m).sum() / (prev_grad.norm()**2 + momentum_normalizer)
                        state["Gm"] += gm ** 2
                        state["beta"] -= beta_lr * lr * gm / (state["Gm"].add(eps).sqrt())
                        state["beta"].clamp_(-1.0,1.0)

                    if closure is not None and epoch <= monotone_epoch:
                        loss = closure()
                        pcopy = p.data.clone()
                        p.add_(-(1-dampening)*state["P"]*grad).add_((1-dampening)*state["beta"] * m)
                        loss_new = closure()
                        if loss_new > group["relax_coef"] * loss: # monotone oracle
                            p.data = pcopy
                        state["m"] = p.data - pcopy
                        del pcopy
                    else:
                        pcopy = p.data.clone()
                        p.add_(-(1-dampening)*state["P"]*grad).add_((1-dampening)*state["beta"] * m)
                        state["m"] = p.data - pcopy
                        del pcopy

                state["prev_grad"] = grad.clone()

        return loss