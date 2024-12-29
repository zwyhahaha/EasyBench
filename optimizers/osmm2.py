import torch
from torch.optim.optimizer import Optimizer

import numpy as np

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("OSMM2",)

class OSMM2(Optimizer):
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
        min_beta: float = -0.0005,
        dampening: float = 0.0,
        adagrad: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr,eps=eps,beta=torch.tensor(beta),
                        weight_decay=weight_decay,stop_step=stop_step,
                        beta_lr=beta_lr, relax_coef=relax_coef,
                        gr_eps=gr_eps, min_beta=min_beta,
                        dampening=dampening, adagrad=adagrad)
        super(OSMM2,self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None, restart=False) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                weight_decay = group["weight_decay"]
                adagrad = group["adagrad"]
                if weight_decay != 0:
                    grad = p.grad.data.add(p.data, alpha=weight_decay)
                else:
                    grad = p.grad.data

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["beta_avg"] = torch.tensor(group["beta"])
                    state["m"] = torch.zeros_like(p)
                    state["Q"] = 0
                    state["Q_avg"] = 0
                    state["prev_grad"] = torch.zeros_like(grad)
                    if adagrad:
                        state["G"] = 0
                        state["Gm"] = 0
                else:
                    state["step"] += 1

                    prev_grad = state["prev_grad"]
                    m = state["m"]
                    eps = group["eps"]
                    gr_eps = group["gr_eps"]
                    lr = group["lr"]
                    beta_lr = group["beta_lr"]
                    min_beta = group["min_beta"]
                    stop_step = group["stop_step"]
                    step = state["step"]
                    if stop_step is None:
                        stop_step = np.inf
                    
                    if restart:
                        # state["Q"] = torch.zeros_like(p)
                        state["Q"] = state["Q_avg"]
                        # group["beta"] = torch.tensor(0)
                        group["beta"] = state["beta_avg"]
                        state["Q_avg"] = 0
                        state["beta_avg"] = torch.tensor(0)
                        if adagrad:
                            state["Gm"] = 0
                            state["G"] = 0
                    else:
                        gr = - prev_grad.mul(grad).sum() / (prev_grad.norm() ** 2 + gr_eps) # gradient of preconditioner
                        if adagrad:
                            state["G"] += gr ** 2 # Adagrad normalizer
                            state["Q"] = state["Q"] - lr * gr / (state["G"].add(eps).sqrt()) # adagrad preconditioner update
                        else:
                            state["Q"] = state["Q"] - lr * gr / step
                        # state["Q"].clamp_(0,1)
                        state["Q_avg"] = state["Q_avg"]*(step-1)/step + state["Q"]/step
                        
                        gm = (grad * m).sum() / (prev_grad.norm() ** 2 + gr_eps) # gradient of momentum coef
                        if adagrad:
                            state["Gm"] += gm ** 2 # Adagrad normalizer for momentum coef
                            group["beta"] = group["beta"] - beta_lr * lr * gm / (state["Gm"].add(eps).sqrt()) # adagrad preconditioner update
                        else:
                            group["beta"] = group["beta"] - beta_lr * lr * gm / step
                        group["beta"].clamp_(min_beta,0.9995)
                        state["beta_avg"] = state["beta_avg"]*(step-1)/step + group["beta"]/step

                    pcopy = p.data.clone()
                    p.add_(-(1-group["dampening"])*state["Q"]*grad).add_(group["beta"] * m)

                    loss_new = closure()

                    if loss_new > group["relax_coef"] * loss:
                        p.data = pcopy

                    state["m"] = p - pcopy

                    del pcopy

                state["prev_grad"] = grad.clone()

        return loss