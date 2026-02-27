#!/usr/bin/env python3
# -*- coding: utf-8 -*-

eternal_braider_simulation.py

"""
TET–CVTL Framework - Eternal Anyon Braiding Simulation (RK45)
=============================================================

Simulazione numerica ad alta precisione del braiding eterno di anyon 
lungo la traiettoria del trifoglio primordiale (3₁, Lk=6).
Mostra convergenza al Nash attractor scalato con rapporto aureo ϕ ≈ 1.618.
Calcola il proxy del vacuum torque τ_vac come media di |r × v|.
Salva la traiettoria in formato .npy per riuso futuro.

Autore: Simon Soliman (TET Collective)
Data: Febbraio 2026
Versione: TET-CVTL 4.0-lite – Simulation Module
Licenza: Uso interno TET Collective

Requisiti: numpy, scipy, matplotlib
Esecuzione: python code/eternal_braider_simulation.py
Output: eternal_braider_trajectory.png (300 dpi) + traj.npy
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def braiding_ode(t, y, golden_phi=(1 + np.sqrt(5)) / 2):
    """
    Sistema di equazioni differenziali per il braiding anyon con perturbazione chirale.
    
    Parametri:
        t          : tempo
        y          : [x, y, z]
        golden_phi : rapporto aureo ϕ ≈ 1.618
    
    Ritorna:
        [dx/dt, dy/dt, dz/dt] normalizzati + perturbazione torque chirale
    """
    x, y, z = y
    
    # Componenti tangenti al trifoglio primordiale parametrico
    dx = np.cos(t) + 4 * np.cos(2 * t)
    dy = -np.sin(t) + 4 * np.sin(2 * t)
    dz = -3 * np.cos(3 * t)
    
    # Perturbazione chirale scalata con ϕ² (dinamica Nash attractor)
    torque_factor = 0.05 * np.sin(2 * np.pi * t / (golden_phi ** 2))
    dx += torque_factor * (y * z - 0.5 * x)
    dy += torque_factor * (z * x - 0.5 * y)
    dz += torque_factor * (x * y - 0.5 * z)
    
    # Normalizzazione per mantenere "velocità eterna" costante
    speed = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    return [dx / speed * 2, dy / speed * 2, dz / speed * 2]


def main():
    # Parametri simulazione
    t_span = (0, 100)                # intervallo temporale
    y0 = [0.5, 0.0, 0.2]             # condizioni iniziali (leggermente fuori dal ciclo)
    n_points = 2000                  # punti per l'interpolazione densa
    
    print("Integrazione RK45 in corso (tolleranza rtol=1e-8, atol=1e-10)...")
    
    sol = solve_ivp(
        braiding_ode,
        t_span,
        y0,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
        max_step=0.1,
        dense_output=True
    )
    
    # Valutazione traiettoria
    t_eval = np.linspace(0, 100, n_points)
    traj = sol.sol(t_eval)           # shape: (3, n_points)
    
    # Salva la traiettoria in formato .npy (shape: n_points × 3)
    np.save('traj.npy', traj.T)
    print("Traiettoria salvata: traj.npy (shape:", traj.T.shape, ")")
    
    # Calcolo proxy vacuum torque τ_vac = media |r × v|
    r = np.stack((traj[0], traj[1], traj[2]), axis=0).T   # (n_points, 3)
    v = np.gradient(traj, t_eval, axis=1).T               # (n_points, 3)
    torque_vectors = np.cross(r, v, axis=1)               # (n_points, 3)
    tau_vac_proxy = np.mean(np.linalg.norm(torque_vectors, axis=1)) * 1e-3
    
    print(f"τ_vac proxy calcolato: {tau_vac_proxy:.6f}")
    
    # Plot 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Traiettoria simulata
    ax.plot(traj[0], traj[1], traj[2],
            color='cyan', linewidth=2.5,
            label='Traiettoria anyon (RK45)')
    
    # Trifoglio ideale (Nash attractor)
    t_ideal = np.linspace(0, 12 * np.pi, 800)
    ax.plot(np.sin(t_ideal) + 2 * np.sin(2 * t_ideal),
            np.cos(t_ideal) - 2 * np.cos(2 * t_ideal),
            -np.sin(3 * t_ideal),
            color='gold', linestyle='--', linewidth=1.2,
            label='Nash attractor: Trefoil primordiale (Lk=6)')
    
    # Frecce torque (ogni 150 punti per chiarezza)
    step = 150
    ax.quiver(traj[0][::step], traj[1][::step], traj[2][::step],
              torque_vectors[::step, 0] * 0.3,
              torque_vectors[::step, 1] * 0.3,
              torque_vectors[::step, 2] * 0.3,
              color='magenta', length=0.8, normalize=True, alpha=0.7,
              label='Frecce τ_vac (proxy torque)')
    
    ax.set_title(f'RK45 Eternal Braiding Simulation TET–CVTL\n'
                 f'Convergenza al trifoglio Nash (ϕ-scaled) — τ_vac proxy = {tau_vac_proxy:.6f}',
                 fontsize=14)
    ax.set_xlabel('X (topological coord)')
    ax.set_ylabel('Y (topological coord)')
    ax.set_zlabel('Z (topological coord)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Salva figura ad alta risoluzione
    plt.savefig('eternal_braider_trajectory.png', dpi=300, bbox_inches='tight')
    print("Figura salvata come 'eternal_braider_trajectory.png'")
    
    plt.show()


if __name__ == "__main__":
    main()