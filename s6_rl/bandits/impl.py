"""Multi-armed bandits: epsilon-greedy, UCB1, Thompson sampling (Bernoulli rewards)."""

from __future__ import annotations

import numpy as np


class BernoulliBandit:
    """k slot machines; arm a pays 1 with probability p[a], else 0."""

    def __init__(self, probs: np.ndarray, seed: int = 0) -> None:
        self.p = np.asarray(probs, dtype=float)
        self.rng = np.random.default_rng(seed)

    @property
    def k(self) -> int:
        return len(self.p)

    def pull(self, arm: int) -> float:
        return float(self.rng.random() < self.p[arm])


class Agent:
    def __init__(self, k: int, seed: int = 0) -> None:
        self.counts = np.zeros(k)
        self.values = np.zeros(k)  # running mean reward per arm
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def select(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]  # incremental mean


class EpsilonGreedy(Agent):
    """With probability epsilon pick a random arm (explore), otherwise the best so far (exploit).
    A fixed epsilon keeps exploring forever, so regret grows linearly in the long run."""

    def __init__(self, k: int, epsilon: float = 0.1, seed: int = 0) -> None:
        super().__init__(k, seed)
        self.epsilon = epsilon

    def select(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.values)))
        best = np.flatnonzero(self.values == self.values.max())
        return int(self.rng.choice(best))  # break ties randomly


class UCB1(Agent):
    """Optimism in the face of uncertainty: pick argmax  mean + sqrt(2 ln t / n).

    The bonus is large for rarely tried arms and shrinks as an arm is sampled, so
    every arm keeps being checked occasionally, but less and less. Regret O(log T).
    """

    def select(self) -> int:
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(untried[0])
        bonus = np.sqrt(2 * np.log(self.t) / self.counts)
        return int(np.argmax(self.values + bonus))


class ThompsonSampling(Agent):
    """Bayesian: keep a Beta(successes + 1, failures + 1) posterior for each arm, draw one
    sample from each posterior, and play the arm whose sample is largest. An arm is played
    with exactly the probability that it is the best one, given the data so far."""

    def __init__(self, k: int, seed: int = 0) -> None:
        super().__init__(k, seed)
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


def run(bandit: BernoulliBandit, agent: Agent, steps: int) -> np.ndarray:
    """Cumulative (pseudo-)regret: sum over time of (best arm's mean - chosen arm's mean)."""
    best = bandit.p.max()
    regret = np.zeros(steps)
    total = 0.0
    for t in range(steps):
        arm = agent.select()
        agent.update(arm, bandit.pull(arm))
        total += best - bandit.p[arm]
        regret[t] = total
    return regret


def average_regret(agent_factory, probs, steps: int, runs: int, seed: int = 0) -> np.ndarray:
    return np.mean(
        [
            run(BernoulliBandit(probs, seed=seed + r), agent_factory(seed + 1000 + r), steps)
            for r in range(runs)
        ],
        axis=0,
    )
