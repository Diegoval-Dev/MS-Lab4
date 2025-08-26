from __future__ import annotations
import argparse
import numpy as np
from typing import Optional
from .io_utils import load_coords, load_distance_matrix, build_distance_matrix_from_coords, load_tsplib_to_matrix
from .tsp_ga import run_ga
from .visualize import animate_evolution

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA para TSP con entradas flexibles y animación.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--coords", type=str, help="Ruta a archivo de coordenadas (x,y) o (id,x,y).")
    src.add_argument("--matrix", type=str, help="Ruta a archivo con matriz simétrica de distancias NxN.")
    src.add_argument("--tsplib", type=str, help="Ruta a archivo TSPLIB .tsp")

    p.add_argument("--N", type=int, default=200, help="Tamaño de población.")
    p.add_argument("--maxIter", type=int, default=500, help="Número máximo de iteraciones.")
    p.add_argument("--survivors", type=float, default=0.2, help="Fracción de sobrevivientes [0,1].")
    p.add_argument("--crossover", type=float, default=0.6, help="Fracción creada por cruce [0,1].")
    p.add_argument("--mutation", type=float, default=0.2, help="Fracción creada por mutación [0,1].")
    p.add_argument("--mutationRate", type=float, default=0.25, help="Probabilidad de aplicar mutación a un individuo.")
    p.add_argument("--tournamentK", type=int, default=3, help="Tamaño de torneo para selección.")
    p.add_argument("--crossoverOp", choices=["ox", "pmx"], default="ox", help="Operador de cruce.")
    p.add_argument("--mutationOp", choices=["swap", "reverse"], default="reverse", help="Operador de mutación.")
    p.add_argument("--immigrants", type=float, default=0.05, help="Fracción de inmigrantes aleatorios por iteración.")
    p.add_argument("--seed", type=int, default=42, help="Semilla RNG.")
    p.add_argument("--no-anim", action="store_true", help="No mostrar animación.")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    coords = None
    if args.coords:
        coords = load_coords(args.coords)
        D = build_distance_matrix_from_coords(coords)
    elif args.matrix:
        D = load_distance_matrix(args.matrix)
    else:
        D = load_tsplib_to_matrix(args.tsplib)

    for name in ["survivors", "crossover", "mutation", "immigrants"]:
        val = getattr(args, name)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} debe estar en [0,1], recibido={val}")

    best, Dbest, hist_routes, hist_dists = run_ga(
        dist=D,
        N=args.N,
        maxIter=args.maxIter,
        survivors_frac=args.survivors,
        crossover_frac=args.crossover,
        mutation_frac=args.mutation,
        mutation_rate=args.mutationRate,
        tournament_size=args.tournamentK,
        crossover_op=args.crossoverOp,
        mutation_op=args.mutationOp,
        immigrants_frac=args.immigrants,
        seed=args.seed
    )

    print("\n===== RESULTADO FINAL =====")
    print(f"best (ruta en índices 0..n-1): {best.tolist()}")
    print(f"D (distancia total): {Dbest:.6f}")

    if not args.no_anim:
        try:
            animate_evolution(hist_routes, hist_dists, coords=coords, dist=D)
        except Exception as e:
            print(f"[AVISO] No fue posible animar (entorno sin backend gráfico): {e}")

if __name__ == "__main__":
    main()
