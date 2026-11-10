from pathlib import Path

import numpy as np
import pytest

from s6_rl.dynamic_programming.impl import policy_evaluation_exact, value_iteration
from s6_rl.gridworld import CliffWorld, greedy_return

from .impl import epsilon_greedy, train


def test_epsilon_greedy():
    Q = np.array([[0.0, 5.0, 1.0, 5.0]])
    rng = np.random.default_rng(0)
    greedy = {epsilon_greedy(Q, 0, 0.0, rng) for _ in range(100)}
    assert greedy == {1, 3}  # ties broken at random among the best
    explore = [epsilon_greedy(Q, 0, 1.0, rng) for _ in range(400)]
    assert set(explore) == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        train(CliffWorld(), method="td-lambda")


def test_q_learning_finds_the_optimal_path():
    env = CliffWorld()
    Q, _ = train(env, "q_learning", episodes=500)
    assert greedy_return(env, Q.argmax(1)) == -13  # same as value iteration's optimum
    _, V_star, _ = value_iteration(env, gamma=1.0)
    assert Q[env.reset()].max() == pytest.approx(V_star[env.reset()], abs=1.0)


def test_sarsa_learns_the_safe_path():
    env = CliffWorld()
    Q, _ = train(env, "sarsa", episodes=500)
    ret = greedy_return(env, Q.argmax(1))
    assert -25 < ret < -13  # reaches the goal, but by a longer route away from the edge
    path_rows = []
    s = env.reset()
    for _ in range(50):
        s = max(env.transitions(s, int(Q[s].argmax())))[1]
        path_rows.append(env.cell(s)[0])
        if env.is_terminal(s):
            break
    assert min(path_rows) <= 1  # it climbs at least two rows above the cliff


def test_classic_result_sarsa_earns_more_while_exploring():
    """Sutton & Barto Figure 6.4: with epsilon = 0.1, Q-learning's edge path keeps falling off the
    cliff during exploration, so SARSA collects more reward ONLINE even though Q-learning's
    greedy policy is the optimal one."""
    env = CliffWorld()
    runs = 10
    q = np.mean([train(env, "q_learning", 500, seed=s)[1] for s in range(runs)], axis=0)
    sarsa = np.mean([train(env, "sarsa", 500, seed=s)[1] for s in range(runs)], axis=0)
    assert sarsa[100:].mean() > q[100:].mean() + 10
    assert q[:20].mean() < q[-100:].mean()  # both actually improve with experience

    import matplotlib.pyplot as plt

    smooth = lambda x: np.convolve(x, np.ones(10) / 10, mode="valid")  # noqa: E731
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(smooth(sarsa), label=f"SARSA (last 400 avg {sarsa[100:].mean():.1f})")
    ax.plot(smooth(q), label=f"Q-learning (last 400 avg {q[100:].mean():.1f})")
    ax.set(
        xlabel="episode",
        ylabel="reward per episode",
        ylim=(-120, 0),
        title="Cliff walking, epsilon = 0.1, 10 runs",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)


def test_q_learning_on_a_slippery_world_approaches_the_dp_optimum():
    env = CliffWorld(slip=0.1)
    Q, _ = train(env, "q_learning", episodes=3000, alpha=0.1, gamma=0.95, seed=3)
    _, V_star, _ = value_iteration(env, gamma=0.95)
    learned = policy_evaluation_exact(env, Q.argmax(1), 0.95)[env.reset()]
    assert learned == pytest.approx(
        V_star[env.reset()], abs=2.0
    )  # near-optimal policy, model never used
