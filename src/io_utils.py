from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional

def load_coords(path: str) -> np.ndarray:
    try:
        coords = np.loadtxt(path, delimiter=",", dtype=float)
    except Exception:
        coords = np.loadtxt(path, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    if coords.shape[1] == 3:
        coords = coords[:, 1:3]
    if coords.shape[1] != 2:
        raise ValueError("El archivo de coordenadas debe tener columnas (x,y) o (id,x,y).")
    return coords

def load_distance_matrix(path: str) -> np.ndarray:
    try:
        M = np.loadtxt(path, delimiter=",", dtype=float)
    except Exception:
        M = np.loadtxt(path, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("La matriz de distancias debe ser cuadrada NxN.")
    if not np.allclose(M, M.T, atol=1e-9):
        raise ValueError("La matriz de distancias debe ser simétrica.")
    np.fill_diagonal(M, 0.0)
    return M

def build_distance_matrix_from_coords(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        diff = coords[i] - coords
        M[i] = np.sqrt((diff**2).sum(axis=1))
    np.fill_diagonal(M, 0.0)
    return M

def load_tsplib_to_matrix(path: str) -> np.ndarray:
    import tsplib95 
    problem = tsplib95.load(path)
    n = problem.dimension
    if getattr(problem, "node_coords", None):
        coords = np.array([problem.node_coords[i] for i in problem.get_nodes()], dtype=float)
        return build_distance_matrix_from_coords(coords)
    M = np.zeros((n, n), dtype=float)
    nodes = list(problem.get_nodes())
    idx = {node: k for k, node in enumerate(nodes)}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            M[idx[i], idx[j]] = problem.get_weight(i, j)
    np.fill_diagonal(M, 0.0)
    if not np.allclose(M, M.T, atol=1e-9):
        M = 0.5 * (M + M.T)
    return M
