"""
FootballEnv — Gymnasium environment for the football IA project.

Grid 10x10. Agent tries to reach the goal zone while avoiding 3 defenders.
Observation: normalized positions of agent, 3 defenders, and goal.
Actions: 4 discrete directions (up/down/left/right).
Rewards: +100 goal, -100 captured, -1 per step, +bonus when approaching goal.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── constants ──────────────────────────────────────────────────────────────────
GRID_SIZE = 10
N_DEFENDERS = 3

ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3

DELTA = {
    ACTION_UP:    np.array([-1,  0]),
    ACTION_DOWN:  np.array([ 1,  0]),
    ACTION_LEFT:  np.array([ 0, -1]),
    ACTION_RIGHT: np.array([ 0,  1]),
}

REWARD_GOAL     =  100.0
REWARD_CAPTURED = -100.0
REWARD_STEP     =   -1.0
APPROACH_SCALE  =    2.0   # multiplier for goal-approach bonus

MAX_STEPS = 200


# ── environment ────────────────────────────────────────────────────────────────
class FootballEnv(gym.Env):
    """
    2-D grid football environment.

    Observation space (10 floats, all in [0, 1]):
        agent_row, agent_col,
        def0_row, def0_col,
        def1_row, def1_col,
        def2_row, def2_col,
        goal_row, goal_col

    Action space: Discrete(4)  → up / down / left / right
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: str | None = None, seed: int | None = 42):
        super().__init__()

        self.render_mode = render_mode
        self._init_seed = seed

        # observation: 2 coords each for agent + 3 defenders + 1 goal
        obs_size = 2 + N_DEFENDERS * 2 + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # goal is fixed at top-center of the grid
        self._goal_pos = np.array([0, GRID_SIZE // 2])

        # internal state (populated by reset)
        self._agent_pos: np.ndarray = None
        self._defender_pos: list[np.ndarray] = []
        self._step_count: int = 0
        self._prev_dist_to_goal: float = 0.0
        self._rng: np.random.Generator = None

    # ── helpers ────────────────────────────────────────────────────────────────

    def _norm(self, pos: np.ndarray) -> np.ndarray:
        """Normalize a grid position to [0, 1]."""
        return pos.astype(np.float32) / (GRID_SIZE - 1)

    def _obs(self) -> np.ndarray:
        parts = [self._norm(self._agent_pos)]
        for d in self._defender_pos:
            parts.append(self._norm(d))
        parts.append(self._norm(self._goal_pos))
        return np.concatenate(parts).astype(np.float32)

    def _manhattan(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).sum())

    def _is_captured(self) -> bool:
        return any(
            np.array_equal(self._agent_pos, d) for d in self._defender_pos
        )

    def _reached_goal(self) -> bool:
        return np.array_equal(self._agent_pos, self._goal_pos)

    def _clip(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, 0, GRID_SIZE - 1)

    def _move_defenders(self) -> None:
        """Each defender takes one random step."""
        for i, d in enumerate(self._defender_pos):
            direction = self._rng.integers(0, 4)
            self._defender_pos[i] = self._clip(d + DELTA[int(direction)])

    # ── gym API ────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # use provided seed, else the init seed, else truly random
        rng_seed = seed if seed is not None else self._init_seed
        self._rng = np.random.default_rng(rng_seed)

        # agent starts at bottom-center
        self._agent_pos = np.array([GRID_SIZE - 1, GRID_SIZE // 2])

        # defenders placed randomly, not on agent or goal
        forbidden = {tuple(self._agent_pos), tuple(self._goal_pos)}
        self._defender_pos = []
        while len(self._defender_pos) < N_DEFENDERS:
            candidate = self._rng.integers(0, GRID_SIZE, size=2)
            t = tuple(candidate)
            if t not in forbidden:
                self._defender_pos.append(candidate.copy())
                forbidden.add(t)

        self._step_count = 0
        self._prev_dist_to_goal = self._manhattan(self._agent_pos, self._goal_pos)

        obs = self._obs()
        info = {"step": 0}
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1

        # move agent
        new_pos = self._clip(self._agent_pos + DELTA[int(action)])
        self._agent_pos = new_pos

        # move defenders
        self._move_defenders()

        # compute reward
        reward = REWARD_STEP

        dist_to_goal = self._manhattan(self._agent_pos, self._goal_pos)
        approach = self._prev_dist_to_goal - dist_to_goal
        reward += APPROACH_SCALE * approach
        self._prev_dist_to_goal = dist_to_goal

        terminated = False
        truncated = False

        if self._reached_goal():
            reward += REWARD_GOAL
            terminated = True
        elif self._is_captured():
            reward += REWARD_CAPTURED
            terminated = True
        elif self._step_count >= MAX_STEPS:
            truncated = True

        obs = self._obs()
        info = {
            "step": self._step_count,
            "dist_to_goal": dist_to_goal,
            "reached_goal": self._reached_goal() if not terminated else (reward > 0),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        """
        Text render of the grid.
        Symbols: A=agent, G=goal, D=defender, .=empty
        """
        grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        r, c = self._goal_pos
        grid[r][c] = "G"

        for d in self._defender_pos:
            grid[d[0]][d[1]] = "D"

        r, c = self._agent_pos
        grid[r][c] = "A"   # agent drawn last — visible if on same cell as G

        lines = ["+" + "-" * GRID_SIZE + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * GRID_SIZE + "+")
        board = "\n".join(lines)

        if self.render_mode == "ansi":
            print(board)
            return None
        return board

    def close(self) -> None:
        pass
