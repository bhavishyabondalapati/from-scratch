"""Dynamic programming for MDPs with a known model: policy evaluation, policy iteration, value iteration."""

from __future__ import annotations

import numpy as np

from s6_rl.gridworld import CliffWorld


def q_from_v(env: CliffWorld, V: np.ndarray, gamma: float) -> np.ndarray:
    """Q(s, a) = sum_{s'} P(s' | s, a) [r + gamma V(s')]: one-step lookahead."""
    Q = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            Q[s, a] = sum(
                p * (r + gamma * V[s2] * (not env.is_terminal(s)))
                for p, s2, r in env.transitions(s, a)
            )
    return Q


def policy_evaluation(
    env: CliffWorld,
    policy: np.ndarray,
    gamma: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 100_000,
) -> np.ndarray:
    """V_pi by repeatedly applying the Bellman EXPECTATION equation
        V(s) <- sum_{s'} P(s' | s, pi(s)) [r + gamma V(s')]
    until it stops changing (a fixed point)."""
    V = np.zeros(env.n_states)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(env.n_states):
            if env.is_terminal(s):
                continue
            v = sum(p * (r + gamma * V[s2]) for p, s2, r in env.transitions(s, int(policy[s])))
            delta = max(delta, abs(v - V[s]))
            V[s] = v  # in-place ("Gauss-Seidel") updates converge faster
        if delta < tol:
            break
    return V


def policy_evaluation_exact(env: CliffWorld, policy: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Same answer by solving the linear system V = R + gamma P V directly (for testing)."""
    n = env.n_states
    P, R = np.zeros((n, n)), np.zeros(n)
    for s in range(n):
        if env.is_terminal(s):
            continue
        for p, s2, r in env.transitions(s, int(policy[s])):
            P[s, s2] += p
            R[s] += p * r
    return np.linalg.solve(np.eye(n) - gamma * P, R)


def policy_iteration(env: CliffWorld, gamma: float = 1.0, start_policy: np.ndarray | None = None):
    """Alternate (1) evaluate the current policy, (2) make it greedy w.r.t. its own values.
    Each improvement step can only make the policy better; with finitely many
    policies it reaches the optimum in a handful of iterations."""
    policy = np.zeros(env.n_states, dtype=int) if start_policy is None else start_policy.copy()
    for iteration in range(1, 1000):
        V = policy_evaluation(env, policy, gamma)
        Q = q_from_v(env, V, gamma)
        new = Q.argmax(axis=1)
        # Keep the old action on ties so the loop cannot cycle between equally good policies.
        keep = np.isclose(Q[np.arange(env.n_states), policy], Q.max(axis=1))
        new[keep] = policy[keep]
        if np.array_equal(new, policy):
            return policy, V, iteration
        policy = new
    raise RuntimeError("policy iteration did not converge")


def value_iteration(
    env: CliffWorld, gamma: float = 1.0, tol: float = 1e-10, max_iter: int = 100_000
):
    """Apply the Bellman OPTIMALITY equation directly:
        V(s) <- max_a sum_{s'} P(s' | s, a) [r + gamma V(s')]
    It is policy iteration with the evaluation step cut down to a single sweep."""
    V = np.zeros(env.n_states)
    for sweep in range(1, max_iter + 1):
        Q = q_from_v(env, V, gamma)
        new_V = Q.max(axis=1)
        new_V[[s for s in range(env.n_states) if env.is_terminal(s)]] = 0.0
        delta = np.abs(new_V - V).max()
        V = new_V
        if delta < tol:
            return q_from_v(env, V, gamma).argmax(axis=1), V, sweep
    raise RuntimeError("value iteration did not converge")
