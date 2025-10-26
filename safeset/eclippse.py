import torch
from torch.nn.functional import softplus

# safeset, eclippse based g_ell(a) = (v/uv)^2 + (w/uw)^2 - 1 <= 0
rho = 1.0 ## appropriately

def eclippse(a,uv=1.0,uw=1.0):
    v,w = a[...,0], a[...,1]

    return (v/uv)**2 + (w/uw)**2 - 1.0

# for loss, use softplus instead of max(,) as differentiable
def violation_loss(a,uv=1.0,uw=1.0,kappa=10.0):
    
    violation = eclippse(a,uv,uw)
    loss = ((softplus(kappa * violation) / kappa) ** 2)
    return loss

## ---- for loss -----

def task_loss(a_hat, a_star):
    return 0.5 * ((a_hat - a_star) ** 2).sum(dim=-1)

def rhotheta_violation_loss(a_raw):
    return rho * violation_loss(a_raw)
