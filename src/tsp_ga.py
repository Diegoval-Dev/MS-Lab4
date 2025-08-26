from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List
from .ga_operators import (
    init_population, evaluate_population, tournament_selection,
    ox_crossover, pmx_crossover,
    mutate_swap, mutate_reverse_segment,
    diversify_random_immigrants, route_distance
)

def run_ga(
    dist: np.ndarray,
    N: int = 200,
    maxIter: int = 500,
    survivors_frac: float = 0.2,
    crossover_frac: float = 0.6,
    mutation_frac: float = 0.2,
    mutation_rate: float = 0.25,
    tournament_size: int = 3,
    crossover_op: str = "ox",  
    mutation_op: str = "reverse",
    immigrants_frac: float = 0.05,
    seed: int | None = 42
) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
    """
    Retorna:
      best_route, best_distance, history_routes, history_distances
    """
    rng = np.random.default_rng(seed)
    n_cities = dist.shape[0]
    pop = init_population(n_cities, N, seed=seed)

    history_routes: List[np.ndarray] = []
    history_dist: List[float] = []

    k_surv = max(1, int(round(survivors_frac * N)))
    k_cross = max(0, int(round(crossover_frac * N)))
    k_mut = max(0, int(round(mutation_frac * N)))
    total = k_surv + k_cross + k_mut
    if total < N:
        k_mut += (N - total)
    elif total > N:
        k_mut = max(0, k_mut - (total - N))

    immigrants = max(0, int(round(immigrants_frac * N)))

    for it in range(maxIter):
        fitness = evaluate_population(pop, dist) 
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]

        best_route = pop[0].copy()
        best_dist = fitness[0]
        history_routes.append(best_route)
        history_dist.append(best_dist)

        survivors = pop[:k_surv]

        parents_needed = max(0, k_cross)
        if parents_needed % 2 == 1:  
            parents_needed += 1
        parents = tournament_selection(pop, fitness, parents_needed, tournament_size, rng)

        children = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i+1]
            if crossover_op == "pmx":
                c = pmx_crossover(p1, p2, rng)
            else:
                c = ox_crossover(p1, p2, rng)
            children.append(c)
        children = np.array(children[:k_cross], dtype=int) if len(children) > 0 else np.empty((0, n_cities), dtype=int)

        mut_targets = tournament_selection(pop, fitness, k_mut, tournament_size, rng) if k_mut > 0 else np.empty((0, n_cities), dtype=int)
        mutants = []
        for r in mut_targets:
            if mutation_op == "swap":
                mutants.append(mutate_swap(r, mutation_rate, rng))
            else:
                mutants.append(mutate_reverse_segment(r, mutation_rate, rng))
        mutants = np.array(mutants, dtype=int) if len(mutants) > 0 else np.empty((0, n_cities), dtype=int)

        new_immigrants = diversify_random_immigrants(immigrants, n_cities, rng) if immigrants > 0 else np.empty((0, n_cities), dtype=int)

        pop = np.vstack([survivors, children, mutants, new_immigrants])

        if pop.shape[0] < N:
            faltan = N - pop.shape[0]
            extra = tournament_selection(pop, evaluate_population(pop, dist), faltan, max(2, tournament_size), rng)
            pop = np.vstack([pop, extra])
        elif pop.shape[0] > N:
            f = evaluate_population(pop, dist)
            idx = np.argsort(f)[:N]
            pop = pop[idx]

    final_fit = evaluate_population(pop, dist)
    idx_best = np.argmin(final_fit)
    final_route = pop[idx_best]
    final_dist = final_fit[idx_best]
    return final_route, float(final_dist), history_routes, history_dist
