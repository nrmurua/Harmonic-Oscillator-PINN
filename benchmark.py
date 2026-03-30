import torch
import itertools
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from src.visualizer import save_triple_comparison

from src.dho_pinn import DhoPINN
from src.samplers import RandomSampler, SobolSampler, CurriculumSampler, AdaptiveSampler
from src.trainer import PINNTrainer, compute_ode_residual
from src.dho_solvers import runge_kutta_4, analytical_solution
from src.pinn_utils import orthogonal_init, xavier_init

def save_loss_history(exp_dir, exp_id, history, adam_epochs):
    history_df = pd.DataFrame({"epoch": range(len(history)), "loss": history})
    history_df.to_csv(os.path.join(exp_dir, f"loss_history_{exp_id}.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.semilogy(history, label='Total Loss', color='#2c3e50')

    if len(history) > adam_epochs:
        plt.axvline(x=adam_epochs, color='#e74c3c', linestyle='--', alpha=0.8, 
                    label=f'Switch to L-BFGS (Epoch {adam_epochs})')
        
        plt.axvspan(adam_epochs, len(history), color='#e74c3c', alpha=0.05)

    plt.title(f"Convergence: {exp_id}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(exp_dir, f"loss_curve_{exp_id}.png"))
    plt.close()


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  results_base_dir = "results/benchmark"
  os.makedirs(results_base_dir, exist_ok=True)

  samplers_config = ['Random', 'Sobol', 'Curriculum', 'Adaptive']
  activations = ['Tanh', 'SiLU']
  initializations = ['Xavier', 'Orthogonal']

  Xis = [0.1, 0.25, 0.4]

  combinations = list(itertools.product(samplers_config, activations, initializations))
  summary_results = []

  print(f"Running benchmark with {len(combinations)} configurations on {device.upper()}...")
  pbar = tqdm(combinations, desc="Benchmarking Configurations")

  for s_name, a_name, i_name in pbar:
    exp_id = f"{s_name}_{a_name}_{i_name}"
    exp_dir = os.path.join(results_base_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    pbar.set_description(f"Running: {exp_id}")

    init_fn = xavier_init if i_name == 'Xavier' else orthogonal_init
    act_fn = torch.nn.Tanh() if a_name == 'Tanh' else torch.nn.SiLU()
    model = DhoPINN(activation=act_fn, init_factory=init_fn, hidden_layers=4, hidden_dim=128).to(device)

    if s_name == 'Random':
      sampler = RandomSampler(z_range=(0, 20), Xi_range=(0.1, 0.4))
    elif s_name == 'Sobol':
      sampler = SobolSampler(z_range=(0, 20), Xi_range=(0.1, 0.4))
    elif s_name == 'Curriculum':
      sampler = CurriculumSampler(initial_z_range=(0, 5), Xi_range=(0.1, 0.4))
    elif s_name == 'Adaptive':
      sampler = AdaptiveSampler(model, compute_ode_residual, pool_factor=10, device=device, z_range=(0, 20), Xi_range=(0.1, 0.4))

    trainer = PINNTrainer(model=model, sampler=sampler, residual_fn=compute_ode_residual, device=device)

    start_time = time.time()
    history = trainer.fit(adam_epochs=100, lbfgs_iter=100, n_points=2048)
    duration = time.time() - start_time

    save_loss_history(exp_dir, exp_id, history, adam_epochs=100) 

    model_path = os.path.join(exp_dir, f"model_{exp_id}.pth")
    torch.save({
      'model_state_dict': model.state_dict(),
      'config': {
        'sampler': s_name,
        'activation': a_name,
        'initialization': i_name,
        'hidden_dim': 128
      },
      'final_loss': history[-1]
    }, model_path)

    for Xi in Xis:
      z_test = torch.linspace(0, 20, 1000).view(-1,1).to(device)

      with torch.no_grad():
        u_pred = model(z_test, torch.full_like(z_test, Xi)).cpu().numpy().flatten()

      z_rk4,  x_rk4 = runge_kutta_4(Xi)
      u_analytical = analytical_solution(z_test.cpu().numpy(), Xi)

      l2_error_pinn = np.linalg.norm(u_pred - u_analytical) / np.linalg.norm(u_analytical)
      l2_error_rk4 = np.linalg.norm(x_rk4 - u_analytical) / np.linalg.norm(u_analytical)

      summary_results.append({
        "Sampler": s_name,
        "Activation": a_name,
        "Initialization": i_name,
        "Xi": Xi,
        "Final Loss": history[-1],
        "L2 Error PINN": l2_error_pinn,
        "L2 Error RK4": l2_error_rk4,
        "Training Time (s)": duration
      })

      val_id = f"{s_name}_{a_name}_{i_name}_Xi{Xi}"
      save_triple_comparison(exp_dir, val_id, z_rk4, u_analytical, x_rk4, z_test.cpu().numpy(), u_pred, Xi, s_name, a_name)

  df = pd.DataFrame(summary_results)
  df.to_csv(os.path.join(results_base_dir, "benchmark_summary.csv"), index=False)
  print(f"Benchmark completed. Summary saved to {os.path.join(results_base_dir, 'benchmark_summary.csv')}")

if __name__ == "__main__":
    main()