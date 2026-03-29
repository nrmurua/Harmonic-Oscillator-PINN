import numpy as np
import argparse
import os
from src.visualizer import plot_comparison_AN

def analytical_solution(z, Xi, x0=0.7, v0=1.2):
	"""
		Analytical solution for the damped harmonic oscillator with given initial conditions.
			
			x(z) = e^(-Xi*z) * (A*cos(wd*z) + B*sin(wd*z))

		Parameters:
			z: input array
			Xi: damping coefficient
			x0: initial position
			v0: initial velocity
		
		Returns:
			x(z): analytical solution at input z
	"""
	wd = np.sqrt(1 - Xi**2)
	exp = np.exp(-Xi*z)
	A = x0
	B = (v0 + Xi*x0) / wd
	
	return exp * (A*np.cos(wd*z) + B*np.sin(wd*z))

def runge_kutta_4(Xi, z_max=20, n_steps=1000, x0=0.7, v0=1.2):
	"""
		Runge-Kutta 4th order method to solve the damped harmonic oscillator ODE.
		
		Parameters:
			Xi: damping coefficient
			z_max: maximum time to simulate
			n_steps: number of time steps
			x0: initial position
			v0: initial velocity
		Returns:
			z: time array
			x: position array at each time step
	"""
	dt = z_max / n_steps
	z = np.linspace(0, z_max, n_steps)
	x = np.zeros(n_steps)
	v = np.zeros(n_steps)

	x[0], v[0] = x0, v0

	def f(x_val, v_val):
		return v_val, -2*Xi*v_val - x_val 

	for i in range(1, n_steps):
		k1_x, k1_v = f(x[i-1], v[i-1])
		k2_x, k2_v = f(x[i-1] + 0.5*dt*k1_x, v[i-1] + 0.5*dt*k1_v)
		k3_x, k3_v = f(x[i-1] + 0.5*dt*k2_x, v[i-1] + 0.5*dt*k2_v)
		k4_x, k4_v = f(x[i-1] + dt*k3_x, v[i-1] + dt*k3_v)

		x[i] = x[i-1] + (dt/6)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
		v[i] = v[i-1] + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

	return z, x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark: Damped Harmonic Oscillator")
    parser.add_argument("--xi", type=float, default=0.1, help="Damping coefficient")
    parser.add_argument("--z_max", type=float, default=20.0, help="Maximum time")
    args = parser.parse_args()

    # Data generation
    z_space = np.linspace(0, args.z_max, 500)
    x_ana = analytical_solution(z_space, args.xi)
    z_rk4, x_rk4 = runge_kutta_4(args.xi, z_max=args.z_max)

    # Visualization
    plot_comparison_AN(z_space, x_ana, z_rk4, x_rk4, args.xi)
    print(f"Simulation completed for xi={args.xi}. Graphs generated in /plots")