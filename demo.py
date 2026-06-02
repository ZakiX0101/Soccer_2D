"""
demo.py — Démonstration visuelle du GeneticAgent entraîné.

Charge results/best_genome.npy et joue N_EPISODES épisodes en affichant
la grille ASCII animée dans le terminal (la grille se met à jour en place).

Usage :
    source .venv/bin/activate
    python demo.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from env.football_env     import FootballEnv
from agents.genetic_agent import GeneticAgent

# ── configuration ──────────────────────────────────────────────────────────────
N_EPISODES  = 5
STEP_DELAY  = 0.3
GENOME_PATH = os.path.join(ROOT, "results", "best_genome.npy")
BASE_SEED   = 300_000   # seeds démo distincts de l'entraînement et de l'éval

# ── couleurs ANSI ──────────────────────────────────────────────────────────────
_R = "\033[0m"   # reset
B  = "\033[1m"   # bold
GR = "\033[90m"  # gray   (bordures)
GN = "\033[92m"  # green  (agent A)
YL = "\033[93m"  # yellow (but G)
RD = "\033[91m"  # red    (défenseur D)
CY = "\033[96m"  # cyan   (vide .)


def _colored_board(board: str) -> str:
    """Applique les couleurs ANSI sur chaque caractère de la grille."""
    out = []
    for ch in board:
        if   ch == "A": out.append(f"{B}{GN}A{_R}")
        elif ch == "G": out.append(f"{B}{YL}G{_R}")
        elif ch == "D": out.append(f"{B}{RD}D{_R}")
        elif ch == ".": out.append(f"{GR}.{_R}")
        elif ch in ("+", "-", "|"): out.append(f"{GR}{ch}{_R}")
        else: out.append(ch)
    return "".join(out)


def _clear_lines(n: int) -> None:
    """Remonte le curseur de n lignes et efface jusqu'à la fin de l'écran."""
    sys.stdout.write(f"\033[{n}A\033[J")
    sys.stdout.flush()


ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}


# ── boucle de démo ─────────────────────────────────────────────────────────────

def run_demo() -> None:
    if not os.path.exists(GENOME_PATH):
        print(f"{RD}Génome introuvable :{_R} {GENOME_PATH}")
        print("Lancez d'abord :  python experiments/run_experiment.py")
        sys.exit(1)

    genome = np.load(GENOME_PATH)
    agent  = GeneticAgent(genome)
    env    = FootballEnv(render_mode=None)

    print(f"\n{B}━━━  Football IA — Demo GeneticAgent  ━━━{_R}")
    print(f"Génome chargé : {genome.round(3)}")
    print(
        f"Légende :  {B}{GN}A{_R} agent   "
        f"{B}{YL}G{_R} but   "
        f"{B}{RD}D{_R} défenseur   "
        f"{GR}.{_R} vide\n"
    )

    summary: list[tuple[int, float, bool]] = []

    for ep in range(1, N_EPISODES + 1):
        seed = BASE_SEED + ep
        obs, _ = env.reset(seed=seed)

        total    = 0.0
        step     = 0
        done     = False
        info: dict = {}

        # calcul de la hauteur du bloc à effacer (constant par épisode)
        board_str  = env.render()                   # grille initiale
        # lignes émises par une frame : \n + titre(1) + statut(1) + grille
        frame_h    = 2 + 1 + board_str.count("\n") + 1
        first_frame = True

        while not done:
            action = agent.select_action(obs)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            step  += 1
            done   = term or trunc

            board_str = env.render()
            colored   = _colored_board(board_str)
            dist      = info["dist_to_goal"]

            if not first_frame:
                _clear_lines(frame_h)
            first_frame = False

            print(
                f"\n{B}Épisode {ep}/{N_EPISODES}{_R}  "
                f"seed={seed}  pas={step:3d}  "
                f"action={ARROW[action]}  "
                f"dist={dist:.0f}  "
                f"cumul={total:+.1f}"
            )
            print(colored)

            time.sleep(STEP_DELAY)

        # ── résultat de l'épisode ──────────────────────────────────────────────
        won = bool(info.get("reached_goal", False))
        if won:
            outcome = f"{B}{GN}BUT !{_R}"
        elif term:
            outcome = f"{B}{RD}Capturé{_R}"
        else:
            outcome = f"{B}{YL}Timeout{_R}"

        print(f"\n  ➜  {outcome}   score final : {B}{total:+.1f}{_R}\n")
        print(f"{GR}{'─' * 44}{_R}")
        summary.append((ep, total, won))
        time.sleep(0.8)

    # ── tableau récapitulatif ──────────────────────────────────────────────────
    wins     = sum(w for _, _, w in summary)
    scores   = [s for _, s, _ in summary]
    mean_sc  = float(np.mean(scores))

    print(f"\n{B}{'━' * 44}{_R}")
    print(f"{B}  Récapitulatif — {N_EPISODES} épisodes{_R}")
    print(f"{B}{'━' * 44}{_R}")
    for ep, score, won in summary:
        icon = f"{GN}✓{_R}" if won else f"{RD}✗{_R}"
        print(f"  Épisode {ep} : {icon}   score = {score:+7.1f}")
    print(f"{GR}{'─' * 44}{_R}")
    print(
        f"  Victoires  : {B}{wins}/{N_EPISODES}{_R}  "
        f"({wins / N_EPISODES * 100:.0f}%)"
    )
    print(f"  Score moyen: {B}{mean_sc:+.1f}{_R}")
    print(f"{B}{'━' * 44}{_R}\n")


if __name__ == "__main__":
    run_demo()
