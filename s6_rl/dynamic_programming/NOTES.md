# Value iteration and policy iteration

**Environment:** cliff walking (`s6_rl/gridworld.py`, Sutton & Barto Example 6.6): 4×12 grid, −1 per
step, −100 and back to start for falling off the cliff, optionally *slippery* (with probability `slip`
the move goes sideways).

**Final metric:** with γ = 1 and no slip, the optimal value of the start state is **−13** (up, 11 × right,
down; checked by hand). With γ = 0.95, value iteration needs 15 / 234 / 162 sweeps (slip 0 / 0.1 / 0.2),
while policy iteration (started from a safe hand-written policy) needs ≤ 15 / 6 / 6 improvement steps;
both reach the same optimal values (tested). Optimal policies (`optimal_policies.txt`):

```
slip=0.0             slip=0.2
>>>>>>>>>>>v         >>>>>>>>>>>v
>>>>>>>>>>>v         >>>>>>>>>>>v
>>>>>>>>>>>v         ^^^^^^^^^^^v
^CCCCCCCCCCG         <CCCCCCCCCCG
```
With slipping, the optimal path keeps one row away from the cliff, and from the start it pushes *left*
into the wall: a sideways slip from there can only go up or stay put, never into the cliff.

## Intuition
When you **know the rules** of the world (what each action does, with what probability, and what reward
it gives), you don't need trial and error: you can *compute* how good every state is.
- **Value of a state:** the total reward you expect from there onward.
- **Bellman equation:** a state's value = immediate reward + (discounted) value of where you land.
- **Value iteration:** start with all values 0 and repeatedly replace each value by the best
  one-step lookahead. Information spreads backward from the goal one step per sweep.
- **Policy iteration:** pick a policy, compute exactly how good it is, then switch every state to the
  action that looks best under those values; repeat until nothing changes.

## Core math
- Bellman expectation: `V^π(s) = Σ_{s'} P(s'|s, π(s)) [r + γ V^π(s')]` — a linear system
  `V = R_π + γ P_π V` (iterative evaluation matches `np.linalg.solve`, tested).
- Bellman optimality: `V*(s) = max_a Σ_{s'} P(s'|s,a) [r + γ V*(s')]`; greedy policy `π*(s) = argmax_a Q*(s,a)`.
- Value iteration is a γ-contraction: error shrinks by a factor γ per sweep (geometric convergence).
- Policy improvement theorem: the greedy policy w.r.t. `V^π` is at least as good as π; finitely many
  policies ⇒ policy iteration terminates (ties kept to prevent cycling).

## Complexity
One sweep: O(|S|·|A|·branching). Value iteration: O(log(1/ε)/(1 − γ)) sweeps. Policy iteration: few
iterations, each needing a full evaluation (O(|S|³) by linear solve, or many sweeps).

## Common pitfalls
- Updating terminal states (their value must stay 0).
- γ = 1 with policies that never terminate: values diverge (fine here because every optimal path ends).
- Policy iteration oscillating between equally good actions: break ties consistently.
- Assuming a model: real problems usually don't give you `P` — that's what model-free TD learning is for.
- Huge state spaces make tabular DP impossible (function approximation, then deep RL).

## When to use vs. alternatives
- Small/medium MDPs with a known model: inventory, routing, games with known rules, planning inside
  model-based RL. Unknown model → Q-learning / SARSA (`td_learning`); huge state spaces → approximate DP / deep RL.

## Interview questions
1. **Value iteration vs policy iteration?** VI does one Bellman optimality backup per sweep (many cheap
   sweeps); PI fully evaluates then improves (few expensive iterations). Both converge to the optimum.
2. **Why does value iteration converge?** The Bellman optimality operator is a γ-contraction in the max
   norm, so it has a unique fixed point reached geometrically.
3. **What does γ do?** Weighs future rewards; smaller γ = more short-sighted, and faster convergence.
4. **What is the difference between V and Q?** V(s) is the value of a state; Q(s, a) the value of taking a
   in s then acting optimally; `V(s) = max_a Q(s, a)`.
5. **Why does slipping change the optimal path here?** The expected cost of a −100 fall outweighs the
   few extra −1 steps of a safer route.
