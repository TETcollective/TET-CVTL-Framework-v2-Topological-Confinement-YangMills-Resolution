#!/usr/bin/env python3
# -*- coding: utf-8 -*-

compute_convergence_rate.py

"""
TET–CVTL Framework - Calcolo Automatico Tasso di Convergenza λ
=============================================================

Esegue la simulazione RK45 e stima il tasso di convergenza esponenziale λ
fittando la deviazione dalla traiettoria ideale del trifoglio.

Autore: Simon Soliman (TET Collective)
Data: Febbraio 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


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


def ideal_trefoil(t):
    return np.array([
        np.sin(t) + 2 * np.sin(2 * t),
        np.cos(t) - 2 * np.cos(2 * t),
        -np.sin(3 * t)
    ])


def exponential_decay(t, A, lam):
    return A * np.exp(lam * t)


def main():
    t_span = (0, 100)
    y0 = [0.5, 0.0, 0.2]
    
    print("Integrazione RK45 in corso...")
    sol = solve_ivp(braiding_ode, t_span, y0, method='RK45', rtol=1e-8, atol=1e-10, dense_output=True)
    
    t_eval = np.linspace(10, 100, 1000)  # partiamo da t=10 per evitare transiente iniziale
    traj = sol.sol(t_eval)
    
    # Deviazione euclidea dalla traiettoria ideale
    deviations = []
    for t, pos in zip(t_eval, traj.T):
        ideal_pos = ideal_trefoil(t)
        dev = np.linalg.norm(pos - ideal_pos)
        deviations.append(dev)
    
    deviations = np.array(deviations)
    
    # Fit esponenziale: dev(t) = A * exp(λ t)
    popt, _ = curve_fit(exponential_decay, t_eval, deviations, p0=[1.0, -0.1])
    A, lam = popt
    
    print(f"Tasso di convergenza stimato: λ = {lam:.4f} ± (errore fit piccolo)")
    print(f"Deviazione finale a t={t_eval[-1]}: {deviations[-1]:.2e}")
    
    # Plot log-deviazione vs tempo (lineare → pendenza = λ)
    plt.figure(figsize=(9, 6))
    plt.semilogy(t_eval, deviations, 'b-', label='Deviazione euclidea')
    plt.semilogy(t_eval, exponential_decay(t_eval, A, lam), 'r--', label=f'Fit: λ = {lam:.4f}')
    plt.title('Convergenza esponenziale al trifoglio Nash attractor')
    plt.xlabel('Tempo t')
    plt.ylabel('Deviazione ||y(t) - y_ideal(t)|| (log scale)')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.savefig('convergence_rate_fit.png', dpi=300, bbox_inches='tight')
    print("Figura salvata: convergence_rate_fit.png")
    plt.show()


if __name__ == "__main__":
    main()