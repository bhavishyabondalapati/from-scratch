from pathlib import Path

import numpy as np
import pytest

from .impl import UCB1, Agent, BernoulliBandit, EpsilonGreedy, ThompsonSampling, average_regret, run

PROBS = np.array([0.1, 0.3, 0.5, 0.55, 0.7])


def test_incremental_mean_and_posterior():
    agent = Agent(2)
    for r in [1, 0, 1, 1]:
        agent.update(0, r)
    assert agent.values[0] == pytest.approx(0.75) and agent.counts[0] == 4
    ts = ThompsonSampling(2)
    for r in [1, 1, 0]:
        ts.update(1, r)
    assert (ts.alpha[1], ts.beta[1]) == (3, 2)


def test_ucb_tries_every_arm_first_then_follows_the_formula():
    agent = UCB1(4)
    chosen = []
    for _ in range(4):
        a = agent.select()
        chosen.append(a)
        agent.update(a, 0.5)
    assert sorted(chosen) == [0, 1, 2, 3]
    agent.values[:] = [0.2, 0.9, 0.5, 0.5]
    agent.counts[:] = [1, 100, 10, 1]
    agent.t = 112
    expected = np.argmax(agent.values + np.sqrt(2 * np.log(112) / agent.counts))
    assert agent.select() == expected == 3  # barely-tried arm wins on optimism


def test_regret_is_zero_when_always_playing_the_best_arm():
    class Oracle(Agent):
        def select(self):
            return int(np.argmax(PROBS))

    assert run(BernoulliBandit(PROBS), Oracle(5), 100)[-1] == 0


@pytest.mark.parametrize(
    "factory",
    [lambda s: EpsilonGreedy(5, 0.1, s), lambda s: UCB1(5, s), lambda s: ThompsonSampling(5, s)],
    ids=["eps-greedy", "ucb1", "thompson"],
)
def test_every_strategy_beats_random_play_and_finds_the_best_arm(factory):
    steps = 3000
    regret = average_regret(factory, PROBS, steps, runs=20)
    random_regret = steps * (PROBS.max() - PROBS.mean())
    assert regret[-1] < 0.3 * random_regret
    agent = factory(7)
    run(BernoulliBandit(PROBS, seed=7), agent, steps)
    assert np.argmax(agent.counts) == 4


def test_logarithmic_vs_linear_regret_and_save_plot():
    steps = 5000
    curves = {
        "random": np.arange(1, steps + 1) * (PROBS.max() - PROBS.mean()),
        "eps-greedy 0.1": average_regret(lambda s: EpsilonGreedy(5, 0.1, s), PROBS, steps, 30),
        "UCB1": average_regret(lambda s: UCB1(5, s), PROBS, steps, 30),
        "Thompson": average_regret(lambda s: ThompsonSampling(5, s), PROBS, steps, 30),
    }
    # Fixed-epsilon exploration never stops: regret keeps growing at a constant rate.
    eps = curves["eps-greedy 0.1"]
    late_rate_eps = (eps[-1] - eps[steps // 2]) / (steps // 2)
    # Exploring with prob. 0.1 costs about 0.1 * (best - average arm) per step, forever.
    expected_rate = 0.1 * (PROBS.max() - PROBS.mean())
    assert late_rate_eps > 0.5 * expected_rate
    for name in ("UCB1", "Thompson"):
        c = curves[name]
        late_rate = (c[-1] - c[steps // 2]) / (steps // 2)
        early_rate = c[steps // 10] / (steps // 10)
        assert late_rate < 0.5 * early_rate  # sub-linear: learning slows the regret growth
    assert curves["Thompson"][-1] < curves["eps-greedy 0.1"][-1]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    for name, c in curves.items():
        ax.plot(c, label=f"{name} ({c[-1]:.0f})")
    ax.set(
        xlabel="step",
        ylabel="cumulative regret",
        yscale="log",
        title="5-armed Bernoulli bandit, 30 runs",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print({k: round(float(v[-1]), 1) for k, v in curves.items()})
