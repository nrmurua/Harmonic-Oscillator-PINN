import matplotlib.pyplot as plt
import os

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