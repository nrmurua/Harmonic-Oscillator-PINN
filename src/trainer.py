import torch
import torch.optim as optim
from tqdm import tqdm

class PINNTrainer:
  def __init__(self, model, sampler, residual_fn, device='cpu'):
    self.model = model.to(device)
    self.sampler = sampler
    self.residual_fn = residual_fn
    self.device = device
    self.loss_history = []

  def fit(self, adam_epochs=5000, lbfgs_iter=100, n_points=1024):
    optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer_adam, 
      mode='min',    
      factor=0.5,    
      patience=500
    )

    milestones = {
        adam_epochs // 4: (0, 10),
        (2 * adam_epochs) // 4: (0, 15),
        (3 * adam_epochs) // 4: (0, 20)
    }

    pbar = tqdm(range(adam_epochs), desc="Training")

    for epoch in pbar:
      if hasattr(self.sampler, 'update_ranges'):
        new_range = milestones[epoch]
        self.sampler.update_ranges(new_z_range=new_range)
        for param_group in optimizer_adam.param_groups:
          param_group['lr'] = max(param_group['lr'], 1e-4)

      optimizer_adam.zero_grad()

      z, Xi = self.sampler.sample(n_points)
      res = self.residual_fn(self.model, z.to(self.device), Xi.to(self.device))
      
      loss = torch.mean(res**2)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
      optimizer_adam.step()
      scheduler.step(loss.detach())

      self.loss_history.append(loss.item())
      current_lr = optimizer_adam.param_groups[0]['lr']
      pbar.set_postfix({"loss": f"{loss.item():.6e}", "lr": f"{current_lr:.6e}"})
    
    optimizer_lbfgs = torch.optim.LBFGS(
      self.model.parameters(),
      max_iter = lbfgs_iter,
      tolerance_grad = 1e-9,
      tolerance_change = 1e-11,
      history_size = 50,
      line_search_fn = "strong_wolfe"
    )

    z_fixed, Xi_fixed = self.sampler.sample(n_points * 2)
    z_fixed, Xi_fixed = z_fixed.to(self.device), Xi_fixed.to(self.device)

    def closure():
      optimizer_lbfgs.zero_grad()
      res = self.residual_fn(self.model, z_fixed, Xi_fixed)
      loss = torch.mean(res**2)
      loss.backward()
      return loss
    
    optimizer_lbfgs.step(closure)

    final_loss = closure().item()
    self.loss_history.append(final_loss)

    return self.loss_history
  


def compute_ode_residual(model, z, Xi):
  z.requires_grad = True
  x = model(z, Xi)

  x_z = torch.autograd.grad(x, z, grad_outputs=torch.ones_like(x), create_graph=True)[0]
  x_zz = torch.autograd.grad(x_z, z, grad_outputs=torch.ones_like(x_z), create_graph=True)[0]

  residual = x_zz + 2*Xi*x_z + x
  return residual