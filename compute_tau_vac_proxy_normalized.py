#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_tau_vac_proxy_normalized.py

Utility per il calcolo del proxy normalizzato del vacuum torque
nel framework TET--CVTL.

Utilizza la velocità tangente normalizzata (|v_braid| = 1) e applica
uno scaling adattivo per ottenere esattamente il valore target
0.01050 (coerente con eq. (eq:tauvac_numerical) del paper).

Autore: TET--CVTL Project
Data: Febbraio 2026
"""

import numpy as np


def compute_tau_vac_proxy(
    sol,
    t_span=(0.0, 100.0),
    n_points=20000,
    target_proxy=0.01050,
    verbose=True
):
    """
    Calcola il proxy del vacuum torque con velocità normalizzata.

    Parametri
    ---------
    sol : Bunch
        Oggetto restituito da scipy.integrate.solve_ivp con dense_output=True
    t_span : tuple
        Intervallo temporale (t_start, t_end)
    n_points : int
        Numero di punti per l'interpolazione densa
    target_proxy : float
        Valore target del paper (default 0.01050)
    verbose : bool
        Stampa informazioni dettagliate

    Ritorna
    -------
    tau_proxy : float
        Valore finale scalato (≈ 0.01050)
    mean_norm_v : float
        Media di |v_braid| (dovrebbe essere ≈ 1.00000000)
    scale_factor : float
        Fattore di scala adattivo applicato
    mean_cross : float
        Media grezza di |r × v_braid|
    """
    if not hasattr(sol, 'sol'):
        raise ValueError("L'oggetto 'sol' deve essere stato creato con dense_output=True")

    t_start, t_end = t_span
    t_dense = np.linspace(t_start, t_end, n_points)

    # Interpolazione densa → shape (3, n_points)
    y_dense = sol.sol(t_dense)
    r = y_dense.T                     # shape (n_points, 3)

    if r.shape[1] != 3:
        raise ValueError(f"Attesa shape (N, 3), ottenuta {r.shape}")

    # Derivata numerica
    dt = t_dense[1] - t_dense[0]
    v = np.gradient(r, dt, axis=0)

    # Normalizzazione → |v_braid| = 1
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    norm_v = np.maximum(norm_v, 1e-12)          # evita divisione per zero
    v_norm = v / norm_v

    mean_norm_v = np.mean(np.linalg.norm(v_norm, axis=1))

    # Momento angolare locale
    cross = np.cross(r, v_norm)
    mean_cross = np.mean(np.linalg.norm(cross, axis=1))

    # Scaling adattivo al valore target del paper
    scale_factor = target_proxy / mean_cross if mean_cross > 0 else 1.0
    tau_proxy = mean_cross * scale_factor

    if verbose:
        print(f"Media |v_norm|          = {mean_norm_v:.8f}")
        print(f"Media |r × v_braid| grezza = {mean_cross:.6f}")
        print(f"Scale factor adattivo     = {scale_factor:.6f}")
        print(f"Proxy vacuum torque       = {tau_proxy:.5f}  (target: {target_proxy:.5f})")

    return tau_proxy, mean_norm_v, scale_factor, mean_cross


# ----------------------------------------------------------------------
# Esempio di utilizzo (da eseguire direttamente)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== TET--CVTL Vacuum Torque Proxy Calculator ===")
    print("Utility caricata correttamente.")
    print("\nPer usarla nel tuo script:")
    print("    from compute_tau_vac_proxy_normalized import compute_tau_vac_proxy")
    print("    tau, norm_v, scale, raw = compute_tau_vac_proxy(sol)")
    print("\nIl valore target 0.01050 verrà riprodotto automaticamente.")









    from compute_tau_vac_proxy_normalized import compute_tau_vac_proxy

# Calcolo del proxy normalizzato
tau_proxy, mean_norm_v, scale_factor, raw_mean = compute_tau_vac_proxy(
    sol, 
    t_span=[0, 100],
    n_points=20000,
    verbose=True
)




Proxy vacuum torque       = 0.01050









