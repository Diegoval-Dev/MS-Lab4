from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional
from matplotlib.animation import FuncAnimation, PillowWriter 

def _fake_coords_from_dist(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]; V = V[:, idx]
    w2 = np.maximum(w[:2], 1e-9)
    X = V[:, :2] * np.sqrt(w2)
    return X

def animate_evolution(
    history_routes,
    history_dists,
    coords,
    dist,
    interval_ms: int = 80,
    save_path: str | None = None,   
    fps: int = 12                 
) -> None:
    if coords is None:
        coords = _fake_coords_from_dist(dist)
    x, y = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    scat = ax.scatter(x, y, s=30)
    line, = ax.plot([], [], lw=2)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    ax.set_title("Evolución GA - TSP (mejor ruta por iteracion)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    pad = 0.05 * max(x.max()-x.min(), y.max()-y.min())
    ax.set_xlim(x.min()-pad, x.max()+pad)
    ax.set_ylim(y.min()-pad, y.max()+pad)

    def update(frame: int):
        route = history_routes[frame]
        px = np.append(x[route], x[route[0]])
        py = np.append(y[route], y[route[0]])
        line.set_data(px, py)
        txt.set_text(f"Iter {frame+1}/{len(history_routes)}\nDist: {history_dists[frame]:.3f}")
        return line, scat, txt

    anim = FuncAnimation(fig, update, frames=len(history_routes), interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"[anim] Guardado en {save_path}")
        plt.close(fig)
    else:
        plt.show()
