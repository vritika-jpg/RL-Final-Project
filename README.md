# CARL: Cash-Aware Replenishment Learner

Tabular RL (Q-learning vs SARSA) for inventory replenishment under stochastic demand and cash constraints.

## Problem & "Why RL?"

We studied a multi-period inventory replenishment problem in which a firm decides how much inventory to reorder each period under uncertain demand. The goal was to maximize long-run business performance by balancing sales revenue against stockout risk, holding cost, ordering cost, and cash tied up in inventory.

This problem requires Reinforcement Learning because it is a sequential decision-making problem under uncertainty. A replenishment decision made today changes future inventory levels, stockout exposure, holding costs, and cash availability — the quality of an action cannot be judged by its immediate effect alone.

## MDP Formulation

### State Space

A finite, discrete state space of 27 states — the Cartesian product of:

- **Inventory level**: low / medium / high
- **Demand condition**: low / normal / high
- **Cash availability**: tight / normal / ample

### Action Space

At each period, the agent chose a discrete reorder quantity:

- 0 units (no order)
- Small order (5 units)
- Medium order (10 units)
- Large order (20 units)

### Reward Function

```
Reward = Sales Revenue − Ordering Cost − Holding Cost − Stockout Penalty − Excess Inventory Penalty
```

This reward structure encouraged the agent to maintain profitable inventory levels while avoiding both understocking and overstocking.

## Environment

We built a custom simulator in which stochastic demand is generated each period via a Poisson distribution conditioned on the current demand state. Inventory updates based on realized sales and replenishment, and cash availability evolves based on net revenue from each period.

## Algorithms

Two tabular RL algorithms were implemented and compared:

- **Q-learning** — off-policy TD control; updates Q-values using the greedy next action (max Q), regardless of what action the agent actually takes
- **SARSA** — on-policy TD control; updates Q-values using the action the agent actually takes next

Both used identical hyperparameters: α = 0.1, γ = 0.95, ε decaying linearly from 1.0 → 0.05 over 5,000 training episodes of 50 steps each.

## Baselines

Trained RL agents were evaluated against four baseline policies:

- **No reorder** — never orders inventory
- **Fixed** — always places a medium order (10 units)
- **Order-up-to** — orders based on current inventory level: low → large, mid → small, high → nothing
- **Random** — selects a uniformly random action each step

## Results

| Policy | Avg Reward | Stockout Rate | Avg Holding Cost |
|---|---|---|---|
| Order-up-to | 3035 | 62.2% | 1115 |
| **Q-learning** | **3026** | **65.4%** | **1081** |
| Random | 2445 | 68.6% | 1079 |
| SARSA | 2323 | 69.0% | 1095 |
| Fixed (medium) | 2252 | 66.6% | 1106 |
| No reorder | −1011 | 91.2% | 370 |

Q-learning achieved the highest reward among RL algorithms, outperforming SARSA by 703 reward points. It matched the hand-crafted Order-up-to heuristic (3035 vs 3026) while additionally learning to adapt to cash availability — something the heuristic cannot do.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train both algorithms (saves `Q.npy`, `SARSA.npy`, `rewards.npy`, `rewards_sarsa.npy`):

```bash
python main.py
```

Generate evaluation charts and report (saved to `evaluations/`):

```bash
python evaluate_combined.py
```
