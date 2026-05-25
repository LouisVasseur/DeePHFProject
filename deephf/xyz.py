"""Read an xyz file into (atomic_numbers, coordinates)."""

import numpy as np

from .constants import SYMBOL_TO_Z


def read_xyz(path):
    with open(path) as f:
        n = int(f.readline())
        f.readline()
        Z = np.zeros(n, np.int32)
        R = np.zeros((n, 3), np.float32)
        for i in range(n):
            parts = f.readline().split()
            Z[i] = SYMBOL_TO_Z.get(parts[0], 0)
            R[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return Z, R
