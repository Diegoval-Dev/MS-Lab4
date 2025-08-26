from __future__ import annotations
import numpy as np
from typing import List, Tuple

# =========================
# Utilidades y fitness
# =========================
def route_distance(route: np.ndarray, dist: np.ndarray) -> float:
    d = 0.0
    for i in range(len(route)):
        d += dist[route[i], route[(i+1) % len(route)]]
    return d

def evaluate_population(pop: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return np.array([route_distance(ind, dist) for ind in pop], dtype=float)

# =========================
# Inicialización
# =========================
def init_population(n_cities: int, N: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(n_cities, dtype=int)
    pop = np.zeros((N, n_cities), dtype=int)
    pop[0] = base
    for i in range(1, N):
        pop[i] = rng.permutation(base)
    return pop

# =========================
# Selección
# =========================
def tournament_selection(pop: np.ndarray, fitness: np.ndarray, k: int, tsize: int, rng: np.random.Generator) -> np.ndarray:
    N = len(pop)
    selected = np.zeros((k, pop.shape[1]), dtype=int)
    for i in range(k):
        idxs = rng.choice(N, size=tsize, replace=False)
        winner = idxs[np.argmin(fitness[idxs])] 
        selected[i] = pop[winner]
    return selected

# =========================
# Cruces (crossover) para permutaciones TSP
# =========================
def ox_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b+1] = p1[a:b+1]
    p2_list = [g for g in p2 if g not in child]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = p2_list[j]
            j += 1
    return child

def pmx_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b+1] = p1[a:b+1]
    mapping = {}
    for i in range(a, b+1):
        mapping[p2[i]] = p1[i]
    for i in list(range(0, a)) + list(range(b+1, n)):
        x = p2[i]
        while x in mapping:
            x = mapping[x]
        child[i] = x
    return child

# =========================
# Mutaciones
# =========================
def mutate_swap(route: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    r = route.copy()
    if rng.random() < rate:
        i, j = rng.integers(0, len(r), size=2)
        r[i], r[j] = r[j], r[i]
    return r

def mutate_reverse_segment(route: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    r = route.copy()
    if rng.random() < rate:
        i, j = sorted(rng.integers(0, len(r), size=2))
        r[i:j+1] = r[i:j+1][::-1]
    return r

def diversify_random_immigrants(n_new: int, n_cities: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(n_cities, dtype=int)
    return np.array([rng.permutation(base) for _ in range(n_new)], dtype=int)
