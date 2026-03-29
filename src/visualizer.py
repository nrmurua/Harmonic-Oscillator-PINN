import matplotlib.pyplot as plt
import os
import numpy as np
import torch

def plot_comparison_AN(z_ana, x_ana, z_rk, x_rk, xi, output_dir="plots"):
    """
      Create and save the graph comparing the analytical and numerical solutions of the damped harmonic oscillator.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    
    plt.plot(z_ana, x_ana, label="Analytical", color="black", lw=2, alpha=0.8)
    plt.plot(z_rk, x_rk, '--', label="Numerical (RK4)", color="red", lw=1.5)
    
    plt.title(f"Benchmark: Damped Harmonic Oscillator (xi = {xi})", fontsize=14)
    plt.xlabel("Time (z)", fontsize=12)
    plt.ylabel("Position x(z)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    file_name = f"benchmark_AN_xi_{str(xi).replace('.', '_')}.png"
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()

def save_triple_comparison(exp_dir, val_id, z_rk4, u_exact, x_rk4, z_test, u_pred, xi, s_name, a_name):
    """
      Create and save the graph comparing the analytical solution, RK4 baseline, and PINN predictions for a specific scenario.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(z_rk4, u_exact, 'k-', linewidth=2, label='Analítica', alpha=0.6)    
    plt.plot(z_rk4, x_rk4, 'b--', label='RK4 (Baseline)')
    
    if torch.is_tensor(u_pred):
        u_pred = u_pred.detach().cpu().numpy()
    if torch.is_tensor(z_test):
        z_test = z_test.detach().cpu().numpy()
        
    plt.plot(z_test, u_pred, 'r:', label='PINN (Ours)')
    
    plt.title(f"Scenario $\\xi={xi}$ | {s_name} + {a_name}")
    plt.xlabel("Time (z)")
    plt.ylabel("Position (x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(exp_dir, f"{val_id}.png"))
    plt.close()