"""
RandomAgent — uniform random action baseline.

Selects one of the 4 discrete actions uniformly at random each step.
Provides the same select_action(obs) interface as GeneticAgent so all
agents are interchangeable in experiment loops.
"""

import numpy as np


class RandomAgent:
    """
    Baseline agent that ignores observations and acts uniformly at random.

    Parameters
    ----------
    seed : int or None
        RNG seed for reproducibility.  None means non-deterministic.
    """

    def __init__(self, seed: int | None = 42):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> int:  # noqa: ARG002
        return int(self.rng.integers(0, 4))

    def reset(self) -> None:
        """No internal state to reset, provided for interface uniformity."""


# ── quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from env.football_env import FootballEnv

    env   = FootballEnv(seed=0)
    agent = RandomAgent(seed=42)
    obs, _ = env.reset()

    total, done = 0.0, False
    while not done:
        action = agent.select_action(obs)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        done = term or trunc

    print(f"RandomAgent episode return: {total:.1f}")
