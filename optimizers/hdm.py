import torch
from torch.optim.optimizer import Optimizer
import numpy as np
from .types import OptFloat, OptLossClosure, Params

__all__ = ("HDM",)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HDM(Optimizer):
    def __init__(
        self,
        params: Params,
        P_lr: OptFloat = None,
        beta_lr: OptFloat = 1.0,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        relax_coef: float = 1.0,
        P_version: str = "diag",
        beta_version: str = "scalar",
        monotone_epoch: int = -1,
        normalize: bool = True,
    ):
        if not 0.0 <= P_lr:
            raise ValueError("Invalid learning rate: {}".format(P_lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(eps=eps,
                        weight_decay=weight_decay,
                        relax_coef=relax_coef,
                        monotone_epoch=monotone_epoch)
        
        self.P_version = P_version
        self.beta_version = beta_version
        self.normalize = normalize
        if self.normalize:
            self.prev_grad_norm_sq = torch.tensor(0.0).to(DEVICE)
            self.prev_m_norm_sq = torch.tensor(0.0).to(DEVICE)
        self.beta_lr = beta_lr
        self.P_lr = P_lr

        if beta_version == 'scalar':
            self.beta = torch.tensor(0.95).to(DEVICE)
            self.Gm = torch.tensor(0.0).to(DEVICE)
            
        if P_version == 'scalar':
            self.P = torch.tensor(self.P_lr).to(DEVICE)
            self.G = torch.tensor(0.0).to(DEVICE)
            
        super(HDM,self).__init__(params, defaults)

    @torch.no_grad()
    def get_grad_norm(self):
        grad_norm = 0.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    weight_decay = group["weight_decay"]
                    if weight_decay != 0:
                        g = p.grad.data.add(p.data, alpha=weight_decay)
                    else:
                        g = p.grad.data
                    grad_norm += torch.sum(torch.mul(g, g))
        grad_norm = torch.sqrt(grad_norm)
        return grad_norm
    
    def sgdm_update(self, p, grad, m):
        if self.P_version == "diag":
            p.data.addcmul_(self.state[p]['P'], -grad)
        elif self.P_version == "scalar":
            p.data.add_(grad, alpha=-self.P)

        if self.beta_version == "diag":
            p.data.addcmul_(self.state[p]['beta'], m)
        elif self.beta_version == "scalar":
            p.data.add_(m, alpha=self.beta)
        return p
        
    @torch.no_grad()
    def step(self, loss=None, closure: OptLossClosure = None, epoch = None) -> OptFloat:
        if self.normalize:
            grad_norm_sq = torch.tensor(0.0).to(DEVICE)
            m_norm_sq = torch.tensor(0.0).to(DEVICE)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                weight_decay = group["weight_decay"]
                if weight_decay != 0:
                    grad = p.grad.data.add(p.data, alpha=weight_decay)
                else:
                    grad = p.grad.data
                
                if self.normalize:
                    grad_norm_sq += grad.norm()**2

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["prev_m"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(p)
                    state["step"] = 0
                    state["L_est"] = torch.tensor(1/self.P_lr).sqrt().to(DEVICE)
                    state["P_lr"] = self.P_lr

                    if self.P_version == 'diag':
                        state["P"] = torch.zeros_like(p) + self.P_lr
                        state["G"] = torch.zeros_like(p)
                    
                    if self.beta_version == 'diag':
                        state["beta"] = torch.zeros_like(p) + 0.95
                        state["Gm"] = torch.zeros_like(p)
                else: # no update for the first iteration
                    state["step"] += 1
                    step = state["step"]
                    prev_grad = state["prev_grad"]
                    prev_m = state["prev_m"]
                    m = state["m"]
                    if self.normalize:
                        m_norm_sq += m.norm()**2
                    eps = group["eps"]
                    monotone_epoch = group["monotone_epoch"]

                    L_guess = torch.norm(grad - prev_grad)**2 / m.norm()**2
                    state["L_est"] = (step * state["L_est"] + L_guess) / (step + 1)
                    if step % 100 == 0:
                        state["P_lr"] = 1 / state["L_est"]
                        # state["L_est"] = torch.tensor(0.0).to(DEVICE)
                    
                    normalizer = self.prev_grad_norm_sq + self.prev_m_norm_sq + eps if self.normalize else 1.0
                    # Hyperparameter Update
                    if self.P_version == "diag":
                        gr = -prev_grad.mul(grad) / normalizer
                        state["G"].addcmul_(gr, gr, value=1)
                        state["P"].addcdiv_(gr, state["G"].add(eps).sqrt(), value=-state["P_lr"])
                        state["P"].clamp_(0.0)
                    elif self.P_version == "scalar":
                        gr = -(prev_grad * grad).sum() / normalizer
                        self.G += gr ** 2
                        self.P -= state["P_lr"] * gr / (self.G.add(eps).sqrt())
                        self.P.clamp_(0.0)
                    
                    if self.beta_version == "diag":
                        gm = grad.mul(prev_m) / normalizer
                        state["Gm"].addcmul_(gm, gm, value=1)
                        state["beta"].addcdiv_(gm, state["Gm"].add(eps).sqrt(), value=-self.beta_lr * self.P_lr)
                        state["beta"].clamp_(0.0,0.9995)
                    elif self.beta_version == "scalar":
                        gm = (grad * prev_m).sum() / normalizer
                        self.Gm += gm ** 2
                        self.beta -= self.beta_lr * self.P_lr * gm / (self.Gm.add(eps).sqrt())
                        self.beta.clamp_(0.0,0.9995)

                    state["prev_m"] = m.clone()
                    state["prev_grad"] = grad.clone()

                    # Update Parameters
                    if closure is not None and epoch <= monotone_epoch:
                        # loss = closure()
                        assert loss is not None
                        pcopy = p.data.clone()
                        p = self.sgdm_update(p, grad, m)
                        loss_new = closure()
                        if loss_new > group["relax_coef"] * loss: # monotone oracle
                            p.data = pcopy
                        state["m"] = p.data - pcopy
                        del pcopy
                    else:
                        pcopy = p.data.clone()
                        p = self.sgdm_update(p, grad, m)
                        state["m"] = p.data - pcopy
                        del pcopy
                        
        if self.normalize:
            self.prev_grad_norm_sq = grad_norm_sq
            self.prev_m_norm_sq = m_norm_sq # / (self.P_lr)
        return loss
    