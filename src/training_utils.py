import torch

def compute_ode_residual(model, z, Xi):
    z.requires_grad = True
    x = model(z, Xi)

    x_z = torch.autograd.grad(x, z, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    x_zz = torch.autograd.grad(x_z, z, grad_outputs=torch.ones_like(x_z), create_graph=True)[0]

    residual = x_zz + 2*Xi*x_z + x
    return residual

def generate_collocation_points(n_points, z_max=20):
    z =  torch.rand(n_points, 1) * z_max
    Xi = 0.1 + torch.rand((n_points, 1)) * 0.3

    return z, Xi