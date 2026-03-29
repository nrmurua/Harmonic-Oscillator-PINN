import torch

class PointSampler:
  def __init__(self, z_range=(0, 20), Xi_range=(0.1, 0.4)):
    self.z_min, self.z_max = z_range
    self.Xi_min, self.Xi_max = Xi_range

  def sample(self):
    raise NotImplementedError("Subclasses must implement the sample method")
    


class RandomSampler(PointSampler):
  def sample(self, n_points):
    z = torch.rand(n_points, 1) * (self.z_max - self.z_min) + self.z_min
    Xi = torch.rand(n_points, 1) * (self.Xi_max - self.Xi_min) + self.Xi_min
    return z, Xi
  


class SobolSampler(PointSampler):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.sampler = torch.quasirandom.SobolEngine(dimension=2, scramble=True)

  def sample(self, n_points):
    points = self.sampler.draw(n_points)
    z = self.z_min + points[:, 0:1] * (self.z_max - self.z_min)
    Xi = self.Xi_min + points[:, 1:2] * (self.Xi_max - self.Xi_min)
    
    return z, Xi
  


class CurriculumSampler(PointSampler):
  def __init__(self, initial_z_range=(0, 5), initial_Xi_range=(0.1, 0.2), **kwargs):
    super().__init__(**kwargs)
    self.current_z_range = initial_z_range
    self.current_Xi_range = initial_Xi_range

  def sample(self, n_points):
    z_min, z_max = self.current_z_range
    Xi_min, Xi_max = self.current_Xi_range

    z = torch.rand(n_points, 1) * (z_max - z_min) + z_min
    Xi = torch.rand(n_points, 1) * (Xi_max - Xi_min) + Xi_min
    
    return z, Xi

  def update_ranges(self, new_z_range=None, new_Xi_range=None):
    if new_z_range:
      self.current_z_range = new_z_range
    if new_Xi_range:
      self.current_Xi_range = new_Xi_range



class AdaptiveSampler(PointSampler):
  def __init__(self, model, residual_fn, pool_factor=10 ,**kwargs):
    super().__init__(**kwargs)
    self.model = model
    self.residual_fn = residual_fn
    self.pool_factor = pool_factor

  def sample(self, n_points):
    n_candidates = n_points * self.pool_factor

    z = torch.rand(n_candidates, 1) * (self.z_max - self.z_min) + self.z_min
    Xi = torch.rand(n_candidates, 1) * (self.Xi_max - self.Xi_min) + self.Xi_min
    
    z.requires_grad = True
    res = self.residual_fn(self.model, z, Xi)

    with torch.no_grad():  
      abs_res = torch.abs(res).squeeze() + 1e-6
      probabilities = abs_res / torch.sum(abs_res)
      indices = torch.multinomial(probabilities.squeeze(), n_points, replacement=True)
    
    return z[indices].detach(), Xi[indices].detach()