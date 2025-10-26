import torch

@torch.enable_grad()
def fs_unroll(a0, x, phi_fn, K, Kp, eta):
    a = a0
    # first k steps
    for _ in range(min(K, Kp)):
        phi = phi_fn(a).mean()
        grad_a, = torch.autograd.grad(phi, a, create_graph=False)
        a = a - eta * grad_a

    a_diff = a
    a_nd = a.detach()


    ## last K-Kp steps: no grad (save memory, section 5.1 idea)
    for _ in range(Kp, K):
        a_nd = a_nd.detach().requires_grad_(True)
        phi_batch = phi_fn(a_nd)
        phi = phi_batch.mean()
        grad_a, = torch.autograd.grad(phi, a_nd, create_graph=False)
        with torch.no_grad():
            a_nd = a_nd - eta * grad_a
        if phi_batch.detach().max() < 1e-3:
            break

    a_hat = a_diff + (a_nd - a_diff).detach()
    return a_hat
