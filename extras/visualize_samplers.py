import torch
import matplotlib.pyplot as plt
import numpy as np
from src.samplers import RandomSampler, SobolSampler, CurriculumSampler, AdaptiveSampler
from src.trainer import compute_ode_residual
from src.dho_pinn import DhoPINN

def main():
  z_range = (0, 20)
  xi_range = (0.1, 0.4)
  n_points = 800
  device = "cpu" 
    
  model = DhoPINN(hidden_layers=3, hidden_dim=32).to(device)

  samplers = {
      "Random": RandomSampler(z_range, xi_range),
      "Sobol": SobolSampler(z_range=z_range, Xi_range=xi_range),
      "Curriculum": CurriculumSampler(initial_z_range=(0, 5), Xi_range=(0.1, 0.4)),
      "Adaptive": AdaptiveSampler(model, compute_ode_residual, pool_factor=10, z_range=z_range, Xi_range=xi_range)
  }

  fig, axes = plt.subplots(2, 2, figsize=(14, 11))
  axes = axes.flatten()

  for i, (name, sampler) in enumerate(samplers.items()):
    ax = axes[i]
        
    if name == "Curriculum":
      steps = 4
      colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
            
      for step in range(steps):
        z_max_step = 4 + step * (z_range[1] - 4) / (steps - 1)
        
        sampler.update_ranges(new_z_range=(0, z_max_step))
        z_step, xi_step = sampler.sample(n_points // steps)
                
        ax.scatter(z_step.detach().numpy(), xi_step.detach().numpy(), 
          alpha=0.6, s=12, color=colors[step], 
          label=f"Fase {step+1} (z_max={z_max_step:.1f})")
          
      ax.legend(fontsize='x-small', loc='upper left') 
    else:
      z, xi = sampler.sample(n_points)
      ax.scatter(z.detach().numpy(), xi.detach().numpy(), alpha=0.5, s=10, color='teal')
        
    ax.set_title(f"Sampler: {name}", fontweight='bold')
    ax.set_xlim(z_range)
    ax.set_ylim(xi_range)
    ax.set_xlabel("Time (z)")
    ax.set_ylabel("Damping (Xi)")
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig("extras/sampler_comparison.png", dpi=300)

if __name__ == "__main__":
    main()