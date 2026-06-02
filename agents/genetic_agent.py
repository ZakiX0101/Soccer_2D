"""
GeneticAgent and GeneticAlgorithm for FootballEnv.

Genome: 8 real weights encoding a linear utility function.
  score(action) = genome @ features(obs, action)

The 8 features computed for each candidate action are:
  w1 · goal proximity    — closer to goal is better
  w2 · min-def safety    — farther from nearest defender
  w3 · mean-def safety   — farther from all defenders on average
  w4 · approach delta    — reward moving toward goal
  w5 · valid-move flag   — penalise bumping into wall
  w6 · row progress      — prefer top rows (goal side)
  w7 · danger flag       — penalise being ≤1 cell from a defender
  w8 · bias              — constant term

GA operators:
  Initialisation : uniform random in [-1, 1]
  Fitness        : mean return over K episodes (same seeds per generation)
  Selection      : tournament (k=3)
  Crossover      : arithmetic  child = α·p1 + (1-α)·p2
  Mutation       : Gaussian noise (σ=0.1) with per-gene probability 0.3
  Elitism        : top-2 copied unchanged to next generation

Default hyperparameters: POP=50, GEN=100, K_eval=5
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.football_env import FootballEnv, GRID_SIZE


# ── genome / hyper-parameter constants ────────────────────────────────────────
GENOME_SIZE    = 8
POP_SIZE       = 50
N_GEN          = 100
K_TOURNAMENT   = 3
MUTATION_SIGMA = 0.1
MUTATION_PROB  = 0.3    # per-gene probability of being mutated
ELITE_SIZE     = 2
EVAL_EPISODES  = 5
SEED           = 42

_NORM    = float(GRID_SIZE - 1)          # = 9.0
_MAX_D   = _NORM * 2                     # max Manhattan distance on the grid
_DELTA   = {
    0: np.array([-1,  0]),   # up
    1: np.array([ 1,  0]),   # down
    2: np.array([ 0, -1]),   # left
    3: np.array([ 0,  1]),   # right
}


# ── feature extraction ─────────────────────────────────────────────────────────

def compute_features(obs: np.ndarray, action: int) -> np.ndarray:
    """
    Return 8 features describing the outcome of `action` from state `obs`.

    obs layout (values normalised to [0, 1]):
      [agent_r, agent_c,
       def0_r,  def0_c,
       def1_r,  def1_c,
       def2_r,  def2_c,
       goal_r,  goal_c]
    """
    agent     = obs[0:2] * _NORM
    defenders = [obs[2 + 2*i : 4 + 2*i] * _NORM for i in range(3)]
    goal      = obs[8:10] * _NORM

    new_agent = np.clip(agent + _DELTA[action], 0.0, _NORM)

    prev_dist_goal = float(np.abs(agent     - goal).sum())
    dist_goal      = float(np.abs(new_agent - goal).sum())

    dists_def  = [float(np.abs(new_agent - d).sum()) for d in defenders]
    min_d_def  = min(dists_def)
    mean_d_def = float(np.mean(dists_def))

    return np.array([
        -dist_goal / _MAX_D,                          # w1 goal proximity
        min_d_def  / _MAX_D,                          # w2 min-defender safety
        mean_d_def / _MAX_D,                          # w3 mean-defender safety
        (prev_dist_goal - dist_goal) / _NORM,         # w4 approach delta
        float(not np.array_equal(new_agent, agent)),  # w5 valid-move flag
        1.0 - new_agent[0] / _NORM,                   # w6 row progress
        -float(min_d_def <= 1.0),                     # w7 immediate-danger flag
        1.0,                                           # w8 bias
    ], dtype=np.float64)


# ── agent ──────────────────────────────────────────────────────────────────────

class GeneticAgent:
    """
    Greedy policy defined by a genome:
      action = argmax_a { genome @ features(obs, a) }
    """

    def __init__(self, genome: np.ndarray):
        self.genome = np.asarray(genome, dtype=np.float64)

    def select_action(self, obs: np.ndarray) -> int:
        scores = [float(self.genome @ compute_features(obs, a)) for a in range(4)]
        return int(np.argmax(scores))


# ── fitness evaluation ─────────────────────────────────────────────────────────

def evaluate_fitness(
    genome: np.ndarray,
    env: FootballEnv,
    episode_seeds: list[int],
) -> float:
    """Mean cumulative return over the given episode seeds."""
    agent = GeneticAgent(genome)
    returns = []
    for seed in episode_seeds:
        obs, _ = env.reset(seed=seed)
        ep_return = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, term, trunc, _ = env.step(action)
            ep_return += reward
            done = term or trunc
        returns.append(ep_return)
    return float(np.mean(returns))


# ── genetic algorithm ──────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Classic generational GA for evolving GeneticAgent genomes.

    Parameters
    ----------
    pop_size       : population size (default 50)
    n_gen          : number of generations (default 100)
    k_tournament   : tournament size for parent selection (default 3)
    mutation_sigma : std-dev of Gaussian noise applied per mutated gene
    mutation_prob  : per-gene probability of mutation (default 0.3)
    elite_size     : number of best individuals copied to next gen (default 2)
    eval_episodes  : episodes per fitness evaluation (default 5)
    seed           : master RNG seed for reproducibility (default 42)
    """

    def __init__(
        self,
        pop_size:       int   = POP_SIZE,
        n_gen:          int   = N_GEN,
        k_tournament:   int   = K_TOURNAMENT,
        mutation_sigma: float = MUTATION_SIGMA,
        mutation_prob:  float = MUTATION_PROB,
        elite_size:     int   = ELITE_SIZE,
        eval_episodes:  int   = EVAL_EPISODES,
        seed:           int   = SEED,
    ):
        self.pop_size       = pop_size
        self.n_gen          = n_gen
        self.k_tournament   = k_tournament
        self.mutation_sigma = mutation_sigma
        self.mutation_prob  = mutation_prob
        self.elite_size     = elite_size
        self.eval_episodes  = eval_episodes
        self.rng            = np.random.default_rng(seed)

    # ── operators ──────────────────────────────────────────────────────────────

    def _init_population(self) -> np.ndarray:
        """Uniform random initialisation in [-1, 1]."""
        return self.rng.uniform(-1.0, 1.0, (self.pop_size, GENOME_SIZE))

    def _tournament_select(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> np.ndarray:
        """
        Tournament selection: pick k_tournament candidates at random,
        return a copy of the genome with the highest fitness.
        """
        idx    = self.rng.choice(self.pop_size, size=self.k_tournament, replace=False)
        winner = idx[int(np.argmax(fitness[idx]))]
        return population[winner].copy()

    def _crossover(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Arithmetic crossover with a single random mixing coefficient α.
          child1 = α·p1 + (1-α)·p2
          child2 = (1-α)·p1 + α·p2
        """
        alpha  = float(self.rng.random())
        child1 = alpha * p1 + (1.0 - alpha) * p2
        child2 = (1.0 - alpha) * p1 + alpha * p2
        return child1, child2

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """
        Per-gene Gaussian mutation: each gene is perturbed with probability
        mutation_prob by N(0, mutation_sigma).
        """
        mutated = genome.copy()
        mask    = self.rng.random(GENOME_SIZE) < self.mutation_prob
        if mask.any():
            mutated[mask] += self.rng.normal(0.0, self.mutation_sigma, int(mask.sum()))
        return mutated

    # ── main loop ──────────────────────────────────────────────────────────────

    def run(
        self,
        env: FootballEnv,
        verbose: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """
        Evolve the population for n_gen generations.

        All individuals in a generation are evaluated on the same randomly
        drawn set of episode seeds, ensuring a fair fitness comparison.

        Returns
        -------
        best_genome : np.ndarray, shape (GENOME_SIZE,)
        history     : dict with lists 'best', 'mean', 'std' (length n_gen)
        """
        population   = self._init_population()
        best_genome  = population[0].copy()
        best_fitness = -np.inf
        history      = {"best": [], "mean": [], "std": []}

        for gen in range(self.n_gen):
            # same episode seeds for every individual this generation
            ep_seeds = [
                int(self.rng.integers(0, 100_000))
                for _ in range(self.eval_episodes)
            ]

            fitness = np.array([
                evaluate_fitness(ind, env, ep_seeds)
                for ind in population
            ])

            gen_best_idx = int(np.argmax(fitness))
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_genome  = population[gen_best_idx].copy()

            history["best"].append(float(fitness[gen_best_idx]))
            history["mean"].append(float(np.mean(fitness)))
            history["std"].append(float(np.std(fitness)))

            if verbose:
                print(
                    f"Gen {gen + 1:3d}/{self.n_gen} | "
                    f"best={fitness[gen_best_idx]:8.2f} | "
                    f"mean={np.mean(fitness):8.2f} ± {np.std(fitness):.2f}"
                )

            # ── elitism ───────────────────────────────────────────────────────
            elite_idx = np.argsort(fitness)[-self.elite_size :]
            elites    = population[elite_idx].copy()

            # ── reproduction ──────────────────────────────────────────────────
            next_pop = list(elites)
            while len(next_pop) < self.pop_size:
                p1       = self._tournament_select(population, fitness)
                p2       = self._tournament_select(population, fitness)
                c1, c2   = self._crossover(p1, p2)
                next_pop.append(self._mutate(c1))
                if len(next_pop) < self.pop_size:
                    next_pop.append(self._mutate(c2))

            population = np.array(next_pop[: self.pop_size])

        return best_genome, history

    # ── persistence helpers ────────────────────────────────────────────────────

    @staticmethod
    def save_curve(history: dict, path: str) -> None:
        """Save best/mean ± std fitness per generation as a PNG."""
        gens = np.arange(1, len(history["best"]) + 1)
        best = np.array(history["best"])
        mean = np.array(history["mean"])
        std  = np.array(history["std"])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(gens, best, label="Best fitness",  color="tab:blue",   linewidth=2)
        ax.plot(gens, mean, label="Mean fitness",  color="tab:orange", linewidth=2)
        ax.fill_between(
            gens, mean - std, mean + std,
            color="tab:orange", alpha=0.2, label="±1 std",
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean cumulative return (over evaluation episodes)")
        ax.set_title("Genetic Algorithm — Fitness Convergence Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Fitness curve saved → {path}")

    @staticmethod
    def save_genome(genome: np.ndarray, path: str) -> None:
        """Persist the best genome as a .npy file."""
        np.save(path, genome)
        print(f"Best genome saved   → {path}")


# ── quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    _env = FootballEnv(seed=SEED)
    _ga  = GeneticAlgorithm(pop_size=10, n_gen=5, eval_episodes=3, seed=SEED)
    _best, _history = _ga.run(_env, verbose=True)
    print(f"\nBest genome: {_best.round(3)}")

    _out = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(_out, exist_ok=True)
    GeneticAlgorithm.save_curve(_history, os.path.join(_out, "fitness_curve_demo.png"))
    GeneticAlgorithm.save_genome(_best,   os.path.join(_out, "best_genome_demo.npy"))
