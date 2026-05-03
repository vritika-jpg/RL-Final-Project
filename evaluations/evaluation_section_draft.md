# Evaluation Section — Draft

> **Author**: Yizhi  
> **Last updated**: 2026-05-02  
> **Status**: in progress  
> **When finalized**: copy into group's main report document

---

## 1. Setup

We evaluate two RL agents — Q-learning and SARSA — against four baseline policies: no reorder, fixed reorder (medium), order-up-to heuristic, and random. Both RL agents were trained on a 27-state inventory replenishment MDP (3 inventory levels × 3 demand conditions × 3 cash levels) over 5,000 episodes of 50 steps each, using identical hyperparameters (α = 0.1, γ = 0.95, linearly decayed ε from 1.0 to 0.05) to isolate the effect of the update rule. Each policy is evaluated over 500 paired test episodes starting from the neutral state (inventory = mid, demand = normal, cash = normal); "paired" means all six policies face the same demand realizations in each episode, controlling for stochastic noise so that observed differences reflect policy quality rather than luck. We report four metrics: (i) average total reward per episode (a profit proxy: revenue − ordering cost − holding cost − stockout penalty − excess-inventory penalty), (ii) stockout rate (fraction of episodes with at least one unmet demand event), (iii) average per-episode holding cost, and (iv) cash efficiency (total revenue ÷ total ordering cost across evaluation; undefined for the No reorder policy). Pairwise differences in mean reward are tested with a paired t-test.

---

## 2. Reward Comparison

*[to be drafted in next round]*

---

## 3. Risk-Cost Trade-off

*[to be drafted]*

---

## 4. Cash Adaptability

*[to be drafted]*

---

## 5. Policy Divergence

*[to be drafted]*

---

## 6. Limitations

*[to be drafted]*

---

## Data Reference (do not delete — keep at bottom of doc as source of truth)

**Latest evaluation run**: `report_20260502_224942.txt` (timestamp may differ — use most recent)

**Charts to reference in the report**:
- `learning_curve_*.png` — Section 2
- `reward_comparison_*.png` — Section 2
- `stockout_holding_*.png` — Section 3
- `policy_heatmap_*.png` — Section 4 / 5
- `cash_efficiency_*.png` — Section 4
- `reward_by_cash_*.png` — Section 4

**Metrics table (paired evaluation, 500 episodes, seed 1000–1499)**:

| Policy | Avg reward | Std | Stockout% | Holding | EndCash | CashEff |
|---|---|---|---|---|---|---|
| Order-up-to | 3,056 | 1,285 | 66.8% | 1,109.8 | 2.00 | 403.2 |
| Q-learning | 2,869 | 1,287 | 66.8% | 1,109.8 | 2.00 | 29.0 |
| Random | 2,451 | 1,255 | 74.2% | 1,078.3 | 1.86 | 7.4 |
| Fixed (medium) | 2,237 | 1,279 | 67.8% | 1,107.7 | 1.83 | 6.1 |
| SARSA | 2,230 | 1,381 | 66.6% | 1,110.0 | 1.83 | 6.2 |
| No reorder | −934 | 2,787 | 90.8% | 378.2 | 2.00 | N/A |

**Paired t-test results (positive diff = first policy higher)**:

| Comparison | Diff | 95% CI | p |
|---|---|---|---|
| Q-learning vs SARSA | +639 | [+621, +657] | <0.001 *** |
| Q-learning vs Order-up-to | −186 | [−192, −181] | <0.001 *** |
| Q-learning vs Fixed (medium) | +632 | [+609, +655] | <0.001 *** |
| Q-learning vs Random | +418 | [+278, +558] | <0.001 *** |
| Q-learning vs No reorder | +3,803 | [+3,482, +4,124] | <0.001 *** |
| SARSA vs Order-up-to | −825 | [−843, −808] | <0.001 *** |

**Per-step reward by cash state (from `reward_by_cash_*.png`)**:

| Policy | Tight (n=195) | Normal (n=1,758) | Ample (n=23,045) |
|---|---|---|---|
| Q-learning | 0 | 47 | 58 |
| SARSA | −3 | 1 | 53 |
| No reorder | 0 | 66 | −20 |
| Fixed (medium) | 0 | 47 | 52 |
| Order-up-to | 0 | 0 | 61 |
| Random | −1 | 7 | 54 |

**Cash-state action profile (from auto report)**:

| Cash level | Q-learning avg action | SARSA avg action |
|---|---|---|
| Tight | 1.56 | 2.11 |
| Normal | 1.56 | 1.22 |
| Ample | 0.89 | 1.89 |

**Policy divergence**: Q-learning and SARSA agree on only **7/27 states (26%)**; the 20 divergent states are listed in the auto-generated `report_*.txt`.