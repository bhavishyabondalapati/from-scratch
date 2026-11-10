"""Cliff-walking gridworld (Sutton & Barto, Example 6.6), with optional slipping.

4 x 12 grid. Start bottom-left, goal bottom-right, the cells between them are a
cliff. Every step costs -1; stepping into the cliff costs -100 and sends you back
to the start. With slip > 0, a move goes in a random perpendicular direction
with probability slip (making the environment stochastic).
"""

from __future__ import annotations

import numpy as np

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
ARROWS = "^>v<"


class CliffWorld:
    def __init__(self, rows: int = 4, cols: int = 12, slip: float = 0.0, seed: int = 0) -> None:
        self.rows, self.cols, self.slip = rows, cols, slip
        self.start, self.goal = (rows - 1, 0), (rows - 1, cols - 1)
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self.n_states, self.n_actions = rows * cols, len(ACTIONS)
        self.rng = np.random.default_rng(seed)

    def index(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.cols + cell[1]

    def cell(self, s: int) -> tuple[int, int]:
        return divmod(s, self.cols)

    def is_terminal(self, s: int) -> bool:
        return self.cell(s) == self.goal

    def _move(self, cell, a: int) -> tuple[int, float]:
        dr, dc = ACTIONS[a]
        r = min(max(cell[0] + dr, 0), self.rows - 1)
        c = min(max(cell[1] + dc, 0), self.cols - 1)
        if (r, c) in self.cliff:
            return self.index(self.start), -100.0
        return self.index((r, c)), -1.0

    def transitions(self, s: int, a: int) -> list[tuple[float, int, float]]:
        """The full model: list of (probability, next state, reward). Used by dynamic programming."""
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        cell = self.cell(s)
        if self.slip == 0:
            return [(1.0, *self._move(cell, a))]
        perpendicular = [(a + 1) % 4, (a + 3) % 4]
        out = [(1 - self.slip, *self._move(cell, a))]
        out += [(self.slip / 2, *self._move(cell, b)) for b in perpendicular]
        return out

    # --- Sampling interface for model-free agents -------------------------------------

    def reset(self) -> int:
        return self.index(self.start)

    def step(self, s: int, a: int) -> tuple[int, float, bool]:
        options = self.transitions(s, a)
        i = self.rng.choice(len(options), p=[p for p, _, _ in options])
        _, s2, r = options[i]
        return s2, r, self.is_terminal(s2)

    def render_policy(self, policy: np.ndarray) -> str:
        lines = []
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                if (r, c) == self.goal:
                    row += "G"
                elif (r, c) in self.cliff:
                    row += "C"
                else:
                    row += ARROWS[policy[self.index((r, c))]]
            lines.append(row)
        return "\n".join(lines)


def greedy_return(env: CliffWorld, policy: np.ndarray, max_steps: int = 200) -> float:
    """Return of following a policy from the start in the DETERMINISTIC version of env."""
    s, total = env.reset(), 0.0
    for _ in range(max_steps):
        _, s, r = max(env.transitions(s, int(policy[s])))  # most likely outcome
        total += r
        if env.is_terminal(s):
            return total
    return total
