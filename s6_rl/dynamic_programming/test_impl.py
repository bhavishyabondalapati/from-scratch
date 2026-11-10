from pathlib import Path

import numpy as np
import pytest

from s6_rl.gridworld import CliffWorld, greedy_return

from .impl import (
    policy_evaluation,
    policy_evaluation_exact,
    policy_iteration,
    q_from_v,
    value_iteration,
)


def safe_policy(env: CliffWorld) -> np.ndarray:
    """Go up to the top row, walk right, come down at the end: avoids the cliff."""
    pol = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        r, c = env.cell(s)
        pol[s] = 0 if (r > 0 and c < env.cols - 1) else (1 if c < env.cols - 1 else 2)
    return pol


@pytest.mark.parametrize("slip", [0.0, 0.1])
def test_iterative_evaluation_matches_linear_solve(slip):
    env = CliffWorld(slip=slip)
    pol = safe_policy(env)
    np.testing.assert_allclose(
        policy_evaluation(env, pol, 0.95), policy_evaluation_exact(env, pol, 0.95), atol=1e-7
    )


def test_optimal_cliff_values_by_hand():
    env = CliffWorld()
    policy, V, _ = value_iteration(env, gamma=1.0)
    assert V[env.reset()] == pytest.approx(-13)  # up, 11 x right, down
    assert greedy_return(env, policy) == -13
    assert V[env.index((0, 11))] == pytest.approx(-3)
    assert V[env.index(env.goal)] == 0
    print("\n" + env.render_policy(policy))


@pytest.mark.parametrize("slip", [0.0, 0.1, 0.2])
def test_policy_and_value_iteration_agree(slip):
    env = CliffWorld(slip=slip)
    gamma = 0.95
    pi_vi, V_vi, sweeps = value_iteration(env, gamma)
    pi_pi, V_pi, iters = policy_iteration(env, gamma, start_policy=safe_policy(env))
    np.testing.assert_allclose(V_vi, V_pi, atol=1e-6)
    # Both are greedy w.r.t. the same optimal values (ties may be broken differently).
    Q = q_from_v(env, V_vi, gamma)
    assert np.all(np.isclose(Q[np.arange(env.n_states), pi_pi], Q.max(1)))
    assert iters < sweeps  # few, expensive iterations vs many cheap sweeps


def test_optimal_values_satisfy_bellman_optimality_and_dominate():
    env = CliffWorld(slip=0.1)
    _, V, _ = value_iteration(env, 0.95)
    Q = q_from_v(env, V, 0.95)
    nonterminal = [s for s in range(env.n_states) if not env.is_terminal(s)]
    np.testing.assert_allclose(V[nonterminal], Q.max(1)[nonterminal], atol=1e-8)
    assert np.all(V >= policy_evaluation_exact(env, safe_policy(env), 0.95) - 1e-8)


def test_slipping_makes_the_optimal_path_keep_away_from_the_cliff():
    gamma = 0.95
    edge = [CliffWorld(slip=s) for s in (0.0, 0.2)]
    policies = [value_iteration(env, gamma)[0] for env in edge]
    start_above = edge[0].index((2, 1))  # the cell just above the cliff's first square
    assert policies[0][start_above] == 1  # deterministic: walk right along the edge
    assert policies[1][start_above] == 0  # slippery: move up, away from the edge
    text = "\n\n".join(
        f"slip={env.slip}:\n{env.render_policy(p)}" for env, p in zip(edge, policies)
    )
    (Path(__file__).parent / "optimal_policies.txt").write_text(text + "\n")
