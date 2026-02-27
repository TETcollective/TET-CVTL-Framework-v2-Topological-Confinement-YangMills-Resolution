#!/usr/bin/env python3
# -*- coding: utf-8 -*-

tau_vax_proxy_from_trajectory.py

"""
TET–CVTL - Calcolo Proxy τ_vac da Traiettoria Salvata
=======================================================

Carica una traiettoria .npy e calcola il proxy τ_vac = media |r × v|.

Autore: Simon Soliman (TET Collective)
Data: Febbraio 2026
"""

import numpy as np


def main(traj_file='traj.npy'):
    try:
        traj = np.load(traj_file)
        print(f"Caricata traiettoria: {traj.shape}")
    except Exception as e:
        print(f"Errore: {e}")
        return
    
    if traj.shape[1] != 3:
        print("Errore: traiettoria deve essere (n_points, 3)")
        return
    
    t_eval = np.linspace(0, 100, len(traj))
    r = traj
    v = np.gradient(traj, t_eval, axis=0)
    torque = np.cross(r, v, axis=1)
    tau_proxy = np.mean(np.linalg.norm(torque, axis=1)) * 1e-3
    
    print(f"τ_vac proxy: {tau_proxy:.6f}")


if __name__ == "__main__":
    main()  # modifica traj_file se necessario