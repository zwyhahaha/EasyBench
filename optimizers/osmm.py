import torch
from torch.optim.optimizer import Optimizer

import numpy as np

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("OSMM",)

class OSMM(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: OptFloat = None,
        beta_lr: OptFloat = 1.0,
        beta: float = 0.9,
        eps: float = 1e-08,
        gr_eps: float = 1e-20,
        weight_decay: float = 0.0,
        stop_step: OptFloat = None,
        relax_coef: float = 1.0,
        min_beta: float = -0.9995,
        dampening: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr,eps=eps,beta=beta,
                        weight_decay=weight_decay,stop_step=stop_step,
                        beta_lr=beta_lr, relax_coef=relax_coef,
                        gr_eps=gr_eps, min_beta=min_beta,
                        dampening=dampening)
        super(OSMM,self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None, restart=False, epoch = None, monotone_step=-1) -> OptFloat:
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
                    state["beta_avg"] = group["beta"]
                    state["Gm"] = 0 
                    state["m"] = torch.zeros_like(p)
                    state["Q"] = torch.zeros_like(p)
                    state["Q_avg"] = torch.zeros_like(p)
                    state["G"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(grad)
                else:
                    state["step"] += 1

                    prev_grad = state["prev_grad"]
                    m = state["m"]
                    eps = group["eps"]
                    gr_eps = group["gr_eps"]
                    dampening = group["dampening"]
                    lr = group["lr"]
                    beta_lr = group["beta_lr"]
                    min_beta = group["min_beta"]
                    stop_step = group["stop_step"]
                    step = state["step"]
                    if stop_step is None:
                        stop_step = np.inf
                    beta_lr = beta_lr * lr

                    # if epoch is not None and epoch > 9:
                    #     frac = (epoch-9) / 40
                    #     dampening = 1.0 * frac + 0.0 * (1 - frac)
                    
                    if restart:
                        state["Q"] = state["Q_avg"]
                        group["beta"] = state["beta_avg"]
                        state["Q_avg"] = torch.zeros_like(p)
                        state["beta_avg"] = 0.0
                        state["Gm"] = 0
                        state["G"] = torch.zeros_like(p)
                    else:
                        momentum_normalizer = m.norm()**2/(lr)
                        gr = -(1-dampening)*prev_grad.mul(grad) / (prev_grad.norm()**2 + momentum_normalizer)
                        state["G"].addcmul_(gr, gr, value=1)
                        state["Q"].addcdiv_(gr, state["G"].add(eps).sqrt(), value=-lr)
                        state["Q_avg"] = state["Q_avg"]*(step-1)/step + state["Q"]/step
                        
                        gm = (1-dampening)*(grad * m).sum() / (prev_grad.norm()**2 + momentum_normalizer)
                        state["Gm"] += gm ** 2
                        group["beta"] = group["beta"] - beta_lr * gm / (state["Gm"].add(eps).sqrt())
                        group["beta"].clamp_(min_beta,0.9995)
                        state["beta_avg"] = state["beta_avg"]*(step-1)/step + group["beta"]/step

                    if closure is not None and epoch <= monotone_step:
                        loss = closure()
                        pcopy = p.data.clone()
                        p.add_(-(1-dampening)*state["Q"]*grad).add_((1-dampening)*group["beta"] * m)
                        loss_new = closure()
                        if loss_new > group["relax_coef"] * loss: # monotone oracle
                            p.data = pcopy
                        state["m"] = p.data - pcopy
                        del pcopy
                    else:
                        pcopy = p.data.clone()
                        p.add_(-(1-dampening)*state["Q"]*grad).add_((1-dampening)*group["beta"] * m)
                        state["m"] = p.data - pcopy
                        del pcopy

                state["prev_grad"] = grad.clone()

        return loss