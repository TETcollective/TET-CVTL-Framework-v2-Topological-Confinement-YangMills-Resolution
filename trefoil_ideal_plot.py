#!/usr/bin/env python3
# -*- coding: utf-8 -*-

trefoil_ideal_plot.py

"""
TET–CVTL Framework - Trifoglio Primordiale Ideale (Nash Attractor)
======================================================================

Plot 3D pulito del trifoglio parametrico 3₁ (Lk=6) come riferimento visivo.

Autore: Simon Soliman (TET Collective)
Data: Febbraio 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    t = np.linspace(0, 12 * np.pi, 1200)
    
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, color='gold', linewidth=2.8, label='Trifoglio primordiale (Lk=6)')
    ax.set_title('Nash Attractor: Trifoglio 3₁ Primordiale\nTET–CVTL Framework', fontsize=14)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.savefig('trefoil_nash_attractor.png', dpi=300, bbox_inches='tight')
    print("Figura salvata: trefoil_nash_attractor.png")
    plt.show()


if __name__ == "__main__":
    main()