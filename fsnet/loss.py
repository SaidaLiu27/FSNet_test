import torch
from safeset.eclippse import *

def first_loss(a_hat: torch.Tensor, a_star: torch.Tensor) -> torch.Tensor:
    # loss = MSE, image: PCPO style

    return 0.5 * ((a_hat - a_star) ** 2).sum(dim=-1)

# def fsnet_total_loss(a_raw: torch.Tensor, a_hat: torch.Tensor,
#                      a_star: torch.Tensor, rho: float) -> torch.Tensor:
#     f = first_loss(a_hat, a_star) ## the loss is for task()
#     reg = 0.5* rho * ((a_raw - a_hat) ** 2).sum(dim=-1).mean()
#     return f + reg, f, reg





def fsnet_total_loss(a_raw, a_hat, a_star, rho, threshold=0.1):
    f = first_loss(a_hat, a_star)
    reg = 0.5 * rho * ((a_raw - a_hat) ** 2).sum(dim=-1)
    total = f + reg

    phi_raw = violation_loss(a_raw)

    # add violation at first steps (section 5.1 idea)
    with torch.no_grad():
        is_violating = (phi_raw >= threshold)

    violation_term = torch.where(
        is_violating,
        rhotheta_violation_loss(a_raw),
        torch.zeros_like(phi_raw)
    )

    total = total + violation_term
    return total.mean(), f.mean(), reg.mean(), violation_term.mean()

                     
