import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.dho_pinn import DhoPINN
from src.dho_solvers import analytical_solution 

def plot_advanced_diagnostics(model, z_range, xi, title, save_path):
    device = next(model.parameters()).device
    z = torch.linspace(z_range[0], z_range[1], 1000).view(-1, 1).to(device).requires_grad_(True)
    xi_tensor = torch.full_like(z, xi)
    
    u_pred = model(z, xi_tensor)
    
    u_dot_pred = torch.autograd.grad(u_pred, z, torch.ones_like(u_pred), create_graph=True)[0]
    
    u_true_all = analytical_solution(z.detach().cpu().numpy(), xi)
    
    if isinstance(u_true_all, tuple):
        u_true, u_dot_true = u_true_all
    else:
        u_true = u_true_all
        u_dot_true = np.gradient(u_true.flatten(), (z_range[1]-z_range[0])/1000)
    
    u_pred_np = u_pred.detach().cpu().numpy().flatten()
    u_dot_pred_np = u_dot_pred.detach().cpu().numpy().flatten()
    u_true = u_true.flatten()
    u_dot_true = u_dot_true.flatten()
    
    abs_error = np.abs(u_true - u_pred_np)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(u_true, u_dot_true, 'k-', alpha=0.3, label='Ground Truth (Sink)')
    ax1.plot(u_pred_np, u_dot_pred_np, 'r--', label='PINN Trajectory')
    ax1.set_title(r"Phase Space Consistency ($\Xi=" + f"{xi}" + r"$)")
    ax1.set_xlabel("Position $u(z)$")
    ax1.set_ylabel("Velocity $u'(z)$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    z_np = z.detach().cpu().numpy().flatten()
    ax2.fill_between(z_np, abs_error, color='crimson', alpha=0.2)
    ax2.plot(z_np, abs_error, color='crimson', lw=1.5)
    ax2.set_yscale('log') 
    ax2.set_title("Pointwise Absolute Error (Log Scale)")
    ax2.set_xlabel("Time (z)")
    ax2.set_ylabel("Absolute Error")
    ax2.set_ylim([1e-7, 1e0])
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def load_model_from_benchmark(exp_id, device="cpu"):
    path = f"results/benchmark/{exp_id}/model_{exp_id}.pth"
    checkpoint = torch.load(path, map_location=device)
    
    config = checkpoint['config']
    act_fn = torch.nn.Tanh() if config['activation'] == 'Tanh' else torch.nn.SiLU()
    
    model = DhoPINN(
        activation=act_fn, 
        hidden_layers=4, 
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        best_id = "Adaptive_Tanh_Xavier"
        best_model = load_model_from_benchmark(best_id, device)
        plot_advanced_diagnostics(
            best_model, (0, 20), 0.1, 
            "Best Case Diagnostics: Adaptive + Tanh", 
            "results/plots/diagnostics_best.png"
        )

        worst_id = "Random_SiLU_Orthogonal"
        worst_model = load_model_from_benchmark(worst_id, device)
        plot_advanced_diagnostics(
            worst_model, (0, 20), 0.1, 
            "Worst Case Diagnostics: Random + SiLU", 
            "results/plots/diagnostics_worst.png"
        )
        
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontraron los archivos del benchmark. {e}")