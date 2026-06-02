"""
experiments/run_experiment.py — Protocole expérimental complet.

Étapes :
  1. Entraînement de l'AG (pop=50, gen=100)
  2. Évaluation des 3 agents sur 50 épisodes avec seeds fixes
  3. Calcul des métriques : taux de victoire, score moyen, pas moyen
  4. Génération de 3 figures sauvegardées dans results/
       01_convergence_ag.png       — courbe de convergence de l'AG
       02_winrate_comparison.png   — taux de victoire en barplot
       03_score_distribution.png   — distribution des scores en boxplot
  5. Tableau récapitulatif en console

Reproductibilité : toutes les sources d'aléatoire sont contrôlées
par MASTER_SEED.  Evaluation sur seeds [EVAL_BASE..EVAL_BASE+N_EVAL-1],
disjoints des seeds d'entraînement (tirés dans [0, 100 000)).
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.football_env       import FootballEnv
from agents.random_agent    import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.genetic_agent   import GeneticAgent, GeneticAlgorithm, SEED as DEFAULT_SEED

# ── configuration ──────────────────────────────────────────────────────────────
MASTER_SEED   = DEFAULT_SEED   # 42 — unique source de reproductibilité
RESULTS_DIR   = os.path.join(ROOT, "results")

N_EVAL        = 50             # épisodes par agent pour l'évaluation
EVAL_BASE     = 200_000        # seeds eval disjoint des seeds d'entraînement

GA_POP        = 50
GA_GEN        = 100
GA_EVAL_EPS   = 5              # épisodes par évaluation de fitness interne

# Couleurs par agent (constant pour les 3 figures)
COLORS = {
    "Random":    "tab:gray",
    "Heuristic": "tab:green",
    "Genetic":   "tab:blue",
}


# ── évaluation ─────────────────────────────────────────────────────────────────

def run_episode(agent, env: FootballEnv, seed: int) -> tuple[float, int, bool]:
    """
    Joue un épisode complet.

    Returns
    -------
    total_return  : float — somme des récompenses
    n_steps       : int   — nombre de pas
    reached_goal  : bool  — True si l'agent a marqué (atteint la zone de but)
    """
    obs, _ = env.reset(seed=seed)
    total, steps, done = 0.0, 0, False
    info: dict = {}
    while not done:
        action = agent.select_action(obs)
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        steps += 1
        done = term or trunc
    return total, steps, bool(info.get("reached_goal", False))


def evaluate_agent(
    agent,
    env: FootballEnv,
    n_episodes: int = N_EVAL,
    base_seed: int  = EVAL_BASE,
) -> dict:
    """
    Évalue un agent sur n_episodes épisodes consécutifs.

    Returns
    -------
    dict avec :
      returns    : np.ndarray (n_episodes,)
      steps      : np.ndarray (n_episodes,)
      wins       : np.ndarray bool (n_episodes,)
      win_rate   : float  [0, 1]
      mean_return: float
      std_return : float
      mean_steps : float
    """
    returns, steps_list, wins = [], [], []
    for i in range(n_episodes):
        ret, st, won = run_episode(agent, env, seed=base_seed + i)
        returns.append(ret)
        steps_list.append(st)
        wins.append(won)

    returns_arr = np.array(returns)
    steps_arr   = np.array(steps_list)
    wins_arr    = np.array(wins, dtype=bool)
    return {
        "returns":     returns_arr,
        "steps":       steps_arr,
        "wins":        wins_arr,
        "win_rate":    float(wins_arr.mean()),
        "mean_return": float(returns_arr.mean()),
        "std_return":  float(returns_arr.std()),
        "mean_steps":  float(steps_arr.mean()),
    }


# ── figures ────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(history: dict, path: str) -> None:
    """Figure 1 — Courbe de convergence de l'AG (best + mean ± std)."""
    gens = np.arange(1, len(history["best"]) + 1)
    best = np.array(history["best"])
    mean = np.array(history["mean"])
    std  = np.array(history["std"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(gens, best, label="Meilleur individu", color="tab:blue",   linewidth=2)
    ax.plot(gens, mean, label="Moyenne populat.",  color="tab:orange", linewidth=2)
    ax.fill_between(gens, mean - std, mean + std,
                    color="tab:orange", alpha=0.2, label="±1 écart-type")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", label="Score nul")
    ax.set_xlabel("Génération", fontsize=12)
    ax.set_ylabel("Retour cumulé moyen", fontsize=12)
    ax.set_title(
        f"AG — Courbe de convergence  (pop={GA_POP}, {GA_GEN} générations)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _savefig(fig, path)


def plot_winrate(results: dict, path: str) -> None:
    """Figure 2 — Taux de victoire comparé (barplot)."""
    names    = list(results.keys())
    winrates = [results[n]["win_rate"] * 100 for n in names]
    colors   = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, winrates, color=colors, width=0.45,
                  edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, winrates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Taux de victoire (%)", fontsize=12)
    ax.set_title(
        f"Comparaison des agents — Taux de victoire ({N_EVAL} épisodes)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, path)


def plot_score_dist(results: dict, path: str) -> None:
    """Figure 3 — Distribution des scores par agent (boxplot)."""
    names  = list(results.keys())
    data   = [results[n]["returns"] for n in names]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data, tick_labels=names, patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # mean markers
    for i, (name, color) in enumerate(zip(names, colors), start=1):
        ax.scatter(i, results[name]["mean_return"],
                   marker="D", color=color, s=60, zorder=5,
                   edgecolors="black", linewidths=0.8, label=f"Moy. {name}")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Retour cumulé par épisode", fontsize=12)
    ax.set_title(
        f"Distribution des scores par agent ({N_EVAL} épisodes)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, path)


# ── console ────────────────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    col_w = [14, 16, 14, 12, 12]
    headers = ["Agent", "Taux victoire", "Score moyen", "Std score", "Pas moyen"]
    sep = "─" * (sum(col_w) + len(col_w) - 1)

    def row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print()
    print(sep)
    print(row(headers))
    print(sep)
    for name, r in results.items():
        print(row([
            name,
            f"{r['win_rate']*100:.1f}%",
            f"{r['mean_return']:.2f}",
            f"{r['std_return']:.2f}",
            f"{r['mean_steps']:.1f}",
        ]))
    print(sep)
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = FootballEnv(seed=MASTER_SEED)

    # ── 1. Entraînement AG ─────────────────────────────────────────────────────
    print("=" * 62)
    print("  ÉTAPE 1 — Entraînement de l'algorithme génétique")
    print(f"  Population={GA_POP}  Générations={GA_GEN}  K_éval={GA_EVAL_EPS}")
    print("=" * 62)
    t0 = time.time()

    ga = GeneticAlgorithm(
        pop_size=GA_POP,
        n_gen=GA_GEN,
        eval_episodes=GA_EVAL_EPS,
        seed=MASTER_SEED,
    )
    best_genome, history = ga.run(env, verbose=True)

    elapsed = time.time() - t0
    print(f"\n  Entraînement terminé en {elapsed:.1f}s")
    print(f"  Meilleur score (fitness) : {max(history['best']):.2f}")
    print(f"  Meilleur génome : {best_genome.round(3)}")

    genome_path = os.path.join(RESULTS_DIR, "best_genome.npy")
    np.save(genome_path, best_genome)
    print(f"  Génome sauvegardé → {genome_path}")

    # ── 2. Évaluation des agents ───────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  ÉTAPE 2 — Évaluation sur {N_EVAL} épisodes")
    print(f"  Seeds : {EVAL_BASE} … {EVAL_BASE + N_EVAL - 1}")
    print("=" * 62)

    agents = {
        "Random":    RandomAgent(seed=MASTER_SEED),
        "Heuristic": HeuristicAgent(),
        "Genetic":   GeneticAgent(best_genome),
    }

    results: dict[str, dict] = {}
    for name, agent in agents.items():
        print(f"  Évaluation {name:<10} ...", end=" ", flush=True)
        t1 = time.time()
        results[name] = evaluate_agent(agent, env, N_EVAL, EVAL_BASE)
        r = results[name]
        print(
            f"victoire={r['win_rate']*100:.0f}%  "
            f"score={r['mean_return']:.1f}±{r['std_return']:.1f}  "
            f"({time.time()-t1:.1f}s)"
        )

    # ── 3. Figures ─────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  ÉTAPE 3 — Génération des figures")
    print("=" * 62)

    paths = {
        "conv":    os.path.join(RESULTS_DIR, "01_convergence_ag.png"),
        "winrate": os.path.join(RESULTS_DIR, "02_winrate_comparison.png"),
        "boxplot": os.path.join(RESULTS_DIR, "03_score_distribution.png"),
    }

    plot_convergence(history,  paths["conv"])
    plot_winrate(results,      paths["winrate"])
    plot_score_dist(results,   paths["boxplot"])

    for label, path in paths.items():
        print(f"  [{label:<8}] → {path}")

    # ── 4. Tableau récapitulatif ───────────────────────────────────────────────
    print("=" * 62)
    print("  RÉSULTATS — Tableau récapitulatif")
    print("=" * 62)
    print_table(results)

    # persist raw metrics for later use
    np.save(os.path.join(RESULTS_DIR, "eval_results.npy"), results)
    print(f"  Métriques brutes → {os.path.join(RESULTS_DIR, 'eval_results.npy')}")
    print()


if __name__ == "__main__":
    main()
