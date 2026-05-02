# evaluate_sarsa.py
# Run this after main.py has generated SARSA.npy and rewards_sarsa.npy.
# Compares SARSA against four baseline policies and plots results.
#
# Usage:
#   python main.py             <- trains Q-learning + SARSA, saves all .npy files
#   python evaluate_sarsa.py   <- loads SARSA results and produces all charts

import random
import numpy as np
import matplotlib.pyplot as plt
import config
from simulator import step, initial_state, state_to_idx, idx_to_state, sample_demand

random.seed(99)
np.random.seed(99)

N_EVAL_EP    = 500
STEPS        = config.STEPS_PER_EP
START        = state_to_idx(1, 1, 1)   # neutral start: inv=mid, dem=normal, cash=normal
ACTION_NAMES = ['No order', 'Small', 'Medium', 'Large']
INV_NAMES    = ['Low', 'Mid', 'High']
DEM_NAMES    = ['Low', 'Normal', 'High']
CASH_NAMES   = ['Tight', 'Normal', 'Ample']


# ── Extended step that also returns stockout and holding cost ─────────────────

def step_detail(state_idx, action):
    inv, dem, cash = idx_to_state(state_idx)

    if cash == 0 and action >= 2:
        action = 1
    if cash == 0 and action == 1 and random.random() < 0.3:
        action = 0

    qty_ordered  = config.ORDER_QTY[action]
    inv_after    = min(inv + (1 if qty_ordered > 0 else 0), 2)
    actual_demand = sample_demand(dem)
    inv_units    = config.INV_UNITS[inv_after]
    units_sold   = min(inv_units, actual_demand)
    stockout     = max(0, actual_demand - units_sold)
    leftover     = max(0, inv_units - actual_demand)

    revenue      = units_sold * config.PRICE
    order_cost   = config.ORDER_COST[action]
    holding_cost = leftover * config.HOLDING_COST if leftover > 5 else 0
    stockout_pen = stockout * config.STOCKOUT_PEN
    excess_pen   = config.EXCESS_PEN if (inv_after == 2 and qty_ordered > 0) else 0
    reward       = revenue - order_cost - holding_cost - stockout_pen - excess_pen

    if stockout > 3:
        next_inv = max(0, inv_after - 1)
    elif leftover > 8:
        next_inv = min(2, inv_after + 1)
    else:
        next_inv = inv_after

    r = random.random()
    next_dem  = max(0, dem-1) if r < 0.15 else (min(2, dem+1) if r < 0.30 else dem)
    net       = revenue - order_cost
    next_cash = min(2, cash+1) if net > 20 else (max(0, cash-1) if net < 0 else cash)

    return state_to_idx(next_inv, next_dem, next_cash), reward, stockout, holding_cost


# ── Baseline policies ─────────────────────────────────────────────────────────

def policy_no_reorder(state):    return 0
def policy_fixed(state):         return 2
def policy_random(state):        return random.randint(0, config.N_ACTIONS - 1)
def policy_order_up_to(state):
    inv, _, _ = idx_to_state(state)
    return [3, 1, 0][inv]   # low->large, mid->small, high->nothing


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(action_fn, n_ep=N_EVAL_EP):
    ep_rewards, ep_stockouts, ep_holding = [], [], []
    for _ in range(n_ep):
        s = START
        total_r, total_so, total_hc = 0, 0, 0
        for _ in range(STEPS):
            a = action_fn(s)
            s, r, so, hc = step_detail(s, a)
            total_r += r; total_so += so; total_hc += hc
        ep_rewards.append(total_r)
        ep_stockouts.append(total_so)
        ep_holding.append(total_hc)
    return {
        'mean_reward':   np.mean(ep_rewards),
        'std_reward':    np.std(ep_rewards),
        'stockout_rate': np.mean([s > 0 for s in ep_stockouts]),
        'mean_stockout': np.mean(ep_stockouts),
        'mean_holding':  np.mean(ep_holding),
        'rewards':       ep_rewards,
    }


# ── Load training results ─────────────────────────────────────────────────────

Q       = np.load('SARSA.npy')
rewards = list(np.load('rewards_sarsa.npy'))


# ── Run all policies ──────────────────────────────────────────────────────────

results = {
    'SARSA':          evaluate(lambda s: int(np.argmax(Q[s]))),
    'No reorder':     evaluate(policy_no_reorder),
    'Fixed (medium)': evaluate(policy_fixed),
    'Order-up-to':    evaluate(policy_order_up_to),
    'Random':         evaluate(policy_random),
}


# ── Print table ───────────────────────────────────────────────────────────────

print(f"\n{'Policy':<20} {'Avg reward':>12} {'Std':>8} {'Stockout%':>10} {'Avg holding':>12}")
print('-' * 66)
for name, m in results.items():
    print(f"{name:<20} {m['mean_reward']:>12.0f} {m['std_reward']:>8.0f} "
          f"{m['stockout_rate']*100:>9.1f}% {m['mean_holding']:>12.1f}")


# ── Plot 1: Learning curve ────────────────────────────────────────────────────

window = 100
smoothed = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]

plt.figure(figsize=(10, 4))
plt.plot(rewards, color='#B4B2A9', linewidth=0.5, alpha=0.6, label='Raw reward')
plt.plot(smoothed, color='#534AB7', linewidth=2, label=f'Smoothed ({window}-ep window)')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('SARSA: learning curve')
plt.legend()
plt.tight_layout()
plt.savefig('plot_learning_curve_sarsa.png', dpi=150)
plt.show()
print('Saved: plot_learning_curve_sarsa.png')


# ── Plot 2: Average reward comparison ────────────────────────────────────────

names   = list(results.keys())
means   = [results[n]['mean_reward'] for n in names]
stds    = [results[n]['std_reward']  for n in names]
colors  = ['#534AB7'] + ['#B4B2A9'] * 4

plt.figure(figsize=(9, 4))
bars = plt.bar(names, means, yerr=stds, color=colors, edgecolor='white',
               linewidth=0.5, capsize=4, error_kw={'linewidth': 1})
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.ylabel('Average reward per episode')
plt.title('Policy comparison: average reward (SARSA)')
plt.tight_layout()
plt.savefig('plot_reward_comparison_sarsa.png', dpi=150)
plt.show()
print('Saved: plot_reward_comparison_sarsa.png')


# ── Plot 3: Stockout rate vs holding cost (scatter) ──────────────────────────

scatter_colors = ['#534AB7', '#B4B2A9', '#B4B2A9', '#AFA9EC', '#B4B2A9']
plt.figure(figsize=(7, 5))
for i, (name, m) in enumerate(results.items()):
    plt.scatter(m['stockout_rate'] * 100, m['mean_holding'],
                color=scatter_colors[i], s=120 if i == 0 else 80,
                marker='D' if i == 0 else 'o', zorder=3, label=name)
    plt.annotate(name, (m['stockout_rate'] * 100, m['mean_holding']),
                 textcoords='offset points', xytext=(8, 4), fontsize=9,
                 color='#444441')
plt.xlabel('Stockout rate (%)')
plt.ylabel('Average holding cost')
plt.title('Stockout rate vs holding cost by policy (SARSA)')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plot_stockout_holding_sarsa.png', dpi=150)
plt.show()
print('Saved: plot_stockout_holding_sarsa.png')


# ── Plot 4: Q-table heatmap ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for cash_idx, ax in enumerate(axes):
    matrix = np.zeros((3, 3))
    for inv in range(3):
        for dem in range(3):
            s = state_to_idx(inv, dem, cash_idx)
            matrix[inv, dem] = int(np.argmax(Q[s]))
    im = ax.imshow(matrix, cmap='Purples', vmin=0, vmax=3, aspect='auto')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(DEM_NAMES)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(INV_NAMES)
    ax.set_xlabel('Demand')
    ax.set_title(f'Cash: {CASH_NAMES[cash_idx]}')
    for inv in range(3):
        for dem in range(3):
            ax.text(dem, inv, ACTION_NAMES[int(matrix[inv, dem])],
                    ha='center', va='center', fontsize=8, color='#26215C')
axes[0].set_ylabel('Inventory')
fig.suptitle('Learned policy (SARSA): best action per state', y=1.02)
plt.colorbar(im, ax=axes[-1], ticks=[0,1,2,3],
             label='Action (0=none 1=small 2=med 3=large)')
plt.tight_layout()
plt.savefig('plot_policy_heatmap_sarsa.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: plot_policy_heatmap_sarsa.png')
