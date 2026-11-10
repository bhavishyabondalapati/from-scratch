"""Model-free temporal-difference control on the cliff world: Q-learning and SARSA."""

from __future__ import annotations

import numpy as np

from s6_rl.gridworld import CliffWorld


def epsilon_greedy(Q: np.ndarray, s: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))


def train(
    env: CliffWorld,
    method: str = "q_learning",
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 0,
    max_steps: int = 1000,
) -> tuple[np.ndarray, list[float]]:
    """Learn Q(s, a) from experience only (no access to env.transitions).

    After each step (s, a, r, s'), move Q(s, a) toward a TD target:
      SARSA       target = r + gamma * Q(s', a')       a' = the action we WILL take (on-policy)
      Q-learning  target = r + gamma * max_b Q(s', b)  the best action (off-policy)
    Both behave epsilon-greedily while learning. Returns Q and each episode's total reward.
    """
    if method not in {"q_learning", "sarsa"}:
        raise ValueError("method must be 'q_learning' or 'sarsa'")
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed + 1)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)
        total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(s, a)
            total += r
            a2 = epsilon_greedy(Q, s2, epsilon, rng)
            if done:
                target = r  # no future after the goal
            elif method == "sarsa":
                target = r + gamma * Q[s2, a2]
            else:
                target = r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])  # TD error times learning rate
            if done:
                break
            s, a = s2, a2
        returns.append(total)
    return Q, returns
