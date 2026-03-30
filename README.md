# DHO-PINN: Parametric Physics-Informed Neural Networks
> **Research and Implementation of Damped Harmonic Oscillators using PINNs for GSoC 2026 (ML4SCI).**

This repository provides a comprehensive benchmarking framework for solving the Damped Harmonic Oscillator (DHO) Ordinary Differential Equation (ODE) using Physics-Informed Neural Networks (PINNs). The model is parametrically conditioned on the damping ratio $\xi \in [0.1, 0.4]$ over an extended temporal domain $z \in [0, 20]$.

---

## Mathematical Formulation

### 1. The Governing Equation
The system models a **Damped Harmonic Oscillator (DHO)**, a fundamental second-order linear ordinary differential equation (ODE). The governing physics, conditioned on the damping ratio $\xi$, is defined as:

$$\frac{d^2x}{dz^2} + 2\xi\frac{dx}{dz} + x = 0$$

Given the initial conditions (ICs) for this benchmark:
* $x(0) = x_0 = 0.7$
* $\frac{dx}{dz}(0) = v_0 = 1.2$
* Domain: $z \in [0, 20]$
* Parameter range: $\xi \in [0.1, 0.4]$ (Underdamped regime)

### 2. Analytical Solution (Reference)
To evaluate the PINN's accuracy, we compare the predictions against the exact analytical solution for the underdamped case ($0 \le \xi < 1$):

$$x_{exact}(z) = e^{-\xi z} \left[ x_0 \cos(\omega_d z) + \frac{v_0 + \xi x_0}{\omega_d} \sin(\omega_d z) \right]$$

Where the damped natural frequency $\omega_d$ is defined as:
$$\omega_d = \sqrt{1 - \xi^2}$$

---

## Technical Architecture

The core implementation focuses on a **Parametric PINN** architecture optimized for long-term numerical stability and gradient flow efficiency.

### 1. Hard Constraints (Exponential Ansatz)
Instead of penalizing initial conditions (ICs) within the loss function (Soft-PINN), this project utilizes an **Exponential Ansatz**. This mathematical formulation guarantees that $x(0)$ and $x'(0)$ are satisfied by construction:

$$x(z, \xi) = x_0 + (1 - e^{-z})v_0 + (1 - e^{-z})^2 \cdot \text{NN}(z, \xi)$$

* **Advantage:** Eliminates numerical instability in the early stages of training and significantly narrows the optimizer's search space by satisfying the Cauchy problem inherently.

### 2. Domain Regularization (Input Scaling)
To mitigate gradient vanishing/explosion issues over $z \in [0, 20]$, the temporal domain is internally regularized to $\hat{z} \in [0, 1]$. Physical consistency is maintained through the chain rule, ensuring the network operates within the high-sensitivity zones of the activation functions.

### 3. Activation Function Analysis (Tanh vs. SiLU)
The benchmark evaluates two distinct activation regimes to determine their impact on gradient propagation and spectral bias:
* **Tanh (Hyperbolic Tangent):** A classical choice for PINNs due to its $C^{\infty}$ continuity, providing strong saturation and smooth second derivatives, which are essential for calculating the ODE residual.
* **SiLU (Sigmoid Linear Unit / Swish):** A self-gated activation function ($x \cdot \sigma(x)$) that avoids the vanishing gradient problem more effectively than Tanh in deeper architectures. Its non-monotonicity often leads to better generalization in high-frequency oscillatory systems.

### 4. Sampling Strategies (Engineered Samplers)
Four distinct sampling engines were implemented to evaluate training efficiency:
* **Random:** Standard uniform distribution sampling.
* **Sobol:** Quasi-Monte Carlo (QMC) sequences for optimal low-discrepancy coverage of the 2D parameter space $(z, \xi)$.
* **Curriculum:** Dynamic domain expansion ($0 \to 5, 0 \to 10, 0 \to 15, 0 \to 20$) based on training epochs to guide the network from local to global dynamics.
* **Adaptive:** Importance sampling based on the magnitude of the physical residual, prioritizing regions with higher local errors (Refined via a `pool_factor` of 10).

---

## Hybrid Optimization Pipeline

The training process is bifurcated into two critical phases to achieve high-fidelity convergence:
1.  **Adam Phase:** 10,000 epochs for global exploration and escaping local minima.
2.  **L-BFGS Phase (Strong-Wolfe):** A second-order refinement stage using a fixed batch and line search to minimize the remaining residual error.

---

## Repository Structure

The project is organized to separate research logic from benchmarking and validation:

```bash
├── src/                 # Core logic and physics-informed components
│   ├── dho_pinn.py      # Architecture definition and Exponential Ansatz
│   ├── samplers.py      # Random, Sobol, Curriculum, and Adaptive logic
│   ├── trainer.py       # Hybrid training loop (Adam + L-BFGS)
│   ├── pinn_utils.py    # Weight initializations (Xavier, Orthogonal)
│   ├── dho_solvers.py   # Analytical and RK4 reference solvers
│   └── visualizer.py    # Plotting and comparison utilities
├── results/             # Benchmarking artifacts
│   └── benchmark/       # Subfolders for each config (CSV, PNG, .pth)
├── test/                # Unit testing for model and samplers
│   └── test_dho_pinn.py
├── extras/              # Auxiliary and experimental scripts
│   └── smoke_train.py   # Fast training script for debugging
├── benchmark.py         # Main execution script for the 16-config study
├── run.sh               # Shell script to automate the entire pipeline
└── requirements.txt     # Python dependencies
```

## Results and Discussion

### 1. Comparative Performance Matrix
The following table summarizes the performance across all 16 architectural combinations. Precision is measured using the **Mean Relative L2 Error** across the tested damping ratios $\xi \in \{0.1, 0.25, 0.4\}$.

| ID | Sampler | Activation | Init | Adam Loss | Final Loss | Mean L2 Error | Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | **Random** | Tanh | Xavier | | | | |
| 02 | **Random** | Tanh | Ortho | | | | |
| 03 | **Random** | SiLU | Xavier | | | | |
| 04 | **Random** | SiLU | Ortho | | | | |
| 05 | **Sobol** | Tanh | Xavier | | | | |
| 06 | **Sobol** | Tanh | Ortho | | | | |
| 07 | **Sobol** | SiLU | Xavier | | | | |
| 08 | **Sobol** | SiLU | Ortho | | | | |
| 09 | **Curriculum** | Tanh | Xavier | | | | |
| 10 | **Curriculum** | Tanh | Ortho | | | | |
| 11 | **Curriculum** | SiLU | Xavier | | | | |
| 12 | **Curriculum** | SiLU | Ortho | | | | |
| 13 | **Adaptive** | Tanh | Xavier | | | | |
| 14 | **Adaptive** | Tanh | Ortho | | | | |
| 15 | **Adaptive** | SiLU | Xavier | | | | |
| 16 | **Adaptive** | SiLU | Ortho | | | | |