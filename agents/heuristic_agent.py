"""
HeuristicAgent — greedy Manhattan-distance baseline.

At each step the agent simulates all 4 moves and picks the one that
minimises Manhattan distance to the goal.  Defenders are ignored.
Wall collisions are handled by clipping (same as the environment).

Provides the same select_action(obs) interface as GeneticAgent so all
agents are interchangeable in experiment loops.

obs layout (values normalised to [0, 1]):
  [agent_r, agent_c,
   def0_r,  def0_c,
   def1_r,  def1_c,
   def2_r,  def2_c,
   goal_r,  goal_c]
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.football_env import GRID_SIZE

_NORM  = float(GRID_SIZE - 1)   # 9.0
_DELTA = {
    0: np.array([-1,  0]),   # up
    1: np.array([ 1,  0]),   # down
    2: np.array([ 0, -1]),   # left
    3: np.array([ 0,  1]),   # right
}


class HeuristicAgent:
    """
    Greedy heuristic: always move toward the goal (Manhattan distance),
    breaking ties by preferring the lowest action index (up > down > left > right).

    Parameters
    ----------
    seed : kept for interface uniformity — not used (deterministic policy).
    """

    def __init__(self, seed: int | None = None):  # noqa: ARG002
        pass

    def select_action(self, obs: np.ndarray) -> int:
        agent = obs[0:2] * _NORM
        goal  = obs[8:10] * _NORM

        best_action = 0
        best_dist   = np.inf

        for action, delta in _DELTA.items():
            new_pos   = np.clip(agent + delta, 0.0, _NORM)
            dist      = float(np.abs(new_pos - goal).sum())
            if dist < best_dist:
                best_dist   = dist
                best_action = action

        return best_action

    def reset(self) -> None:
        """No internal state to reset, provided for interface uniformity."""


# ── quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from env.football_env import FootballEnv

    env   = FootballEnv(seed=0)
    agent = HeuristicAgent()
    obs, _ = env.reset()

    total, done, steps = 0.0, False, 0
    while not done:
        action = agent.select_action(obs)
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        steps += 1
        done = term or trunc

    outcome = "GOAL" if info.get("reached_goal") else "captured/timeout"
    print(f"HeuristicAgent episode return: {total:.1f}  ({steps} steps, {outcome})")
