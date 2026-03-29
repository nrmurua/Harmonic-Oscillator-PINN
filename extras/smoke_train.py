import torch
import os
from src.dho_pinn import DhoPINN
from src.samplers import SobolSampler
from src.trainer import compute_ode_residual
from src.trainer import PINNTrainer

def smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "results/smoke_test"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Testing on {device.upper()}...")

    model = DhoPINN(activation=torch.nn.SiLU()).to(device)
    
    sampler = SobolSampler(z_range=(0, 20), Xi_range=(0.1, 0.4))
    
    trainer = PINNTrainer(
        model=model,
        sampler=sampler,
        residual_fn=compute_ode_residual,
        device=device
    )

    print("Training (100 epochs)...")
    history = trainer.fit(adam_epochs=90, lbfgs_iter=10, n_points=512)

    # 4. Verificación de salida
    if len(history) > 0:
        print(f"Initial Loss: {history[0]:.4e}")
        print(f"Final Loss: {history[-1]:.4e}")
    else:
        print("[ERROR] Empty Loss History.")

if __name__ == "__main__":
    smoke_test()