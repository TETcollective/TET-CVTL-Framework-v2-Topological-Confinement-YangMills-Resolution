#!/usr/bin/env python3
# -*- coding: utf-8 -*-

compute_tau_vac_proxy.py

"""
TET–CVTL - Calcolo Proxy Vacuum Torque da Traiettoria
=======================================================

Carica una traiettoria salvata (es. da eternal_braider_simulation.py) 
e calcola il proxy τ_vac = media |r × v|.

Autore: Simon Soliman
"""

import numpy as np


def compute_tau_proxy(traj_file='traj.npy'):
    """
    Calcola proxy τ_vac da file .npy contenente traiettoria (shape: (n_points, 3))
    """
    try:
        traj = np.load(traj_file)
    except FileNotFoundError:
        print(f"File {traj_file} non trovato.")
        return None
    
    t_eval = np.linspace(0, 100, len(traj))
    r = traj
    v = np.gradient(traj, t_eval, axis=0)
    torque_vectors = np.cross(r, v, axis=1)
    tau_proxy = np.mean(np.linalg.norm(torque_vectors, axis=1)) * 1e-3
    
    print(f"τ_vac proxy: {tau_proxy:.6f}")
    return tau_proxy


if __name__ == "__main__":
    compute_tau_proxy()