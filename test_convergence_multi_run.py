#!/usr/bin/env python3
# -*- coding: utf-8 -*-

test_convergence_multi_run.py

"""
TET–CVTL - Test Robustezza Convergenza Nash Attractor
=======================================================

Esegue N simulazioni RK45 con y0 casuali per confermare convergenza universale.

Autore: Simon Soliman (TET Collective)
Data: Febbraio 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def braiding_ode(t, y, golden_phi=(1 + np.sqrt(5)) / 2):
    x, y, z = y
    dx = np.cos(t) + 4 * np.cos(2 * t)
    dy = -np.sin(t) + 4 * np.sin(2 * t)
    dz = -3 * np.cos(3 * t)
    torque_factor = 0.05 * np.sin(2 * np.pi * t / (golden_phi ** 2))
    dx += torque_factor * (y * z - 0.5 * x)
    dy += torque_factor * (z * x - 0.5 * y)
    dz += torque_factor * (x * y - 0.5 * z)
    speed = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    return [dx / speed * 2, dy / speed * 2, dz / speed * 2]


def run_one(y0):
    sol = solve_ivp(braiding_ode, (0, 100), y0, method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y.T  # (n_steps, 3)


def main(n_runs=20):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n_runs):
        y0 = np.random.uniform(-1, 1, 3)
        print(f"Run {i+1}/{n_runs}  y0 = {y0}")
        traj = run_one(y0)
        ax.plot(traj[:,0], traj[:,1], traj[:,2], color='cyan', alpha=0.4, linewidth=1)
    
    # Trifoglio ideale
    t_ideal = np.linspace(0, 12*np.pi, 800)
    ax.plot(np.sin(t_ideal)+2*np.sin(2*t_ideal), 
            np.cos(t_ideal)-2*np.cos(2*t_ideal), 
            -np.sin(3*t_ideal), color='gold', linewidth=3, linestyle='--',
            label='Nash attractor')
    
    ax.set_title(f'Test robustezza: {n_runs} run con y0 casuali\nTET–CVTL')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('convergence_multi_run_test.png', dpi=300, bbox_inches='tight')
    print("Figura salvata: convergence_multi_run_test.png")
    plt.show()


if __name__ == "__main__":
    main(n_runs=20)  # aumenta a 50–100 per test più estesi.