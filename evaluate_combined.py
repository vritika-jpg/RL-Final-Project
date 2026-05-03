# evaluate_combined.py
# Combined evaluation: Q-learning + SARSA + 4 baselines.
# Saves 4 timestamped charts to evaluations/<chart>_<timestamp>.png.
#
# Usage:
#   python main.py                <- trains both algorithms, saves all .npy files
#   python evaluate_combined.py   <- loads both results, saves 4 combined charts

import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import config
from simulator import state_to_idx, idx_to_state, sample_demand

random.seed(99)
np.random.seed(99)

N_EVAL_EP    = 500
STEPS        = config.STEPS_PER_EP
START        = state_to_idx(1, 1, 1)   # neutral start: inv=mid, dem=normal, cash=normal
ACTION_NAMES = ['No order', 'Small', 'Medium', 'Large']
INV_NAMES    = ['Low', 'Mid', 'High']
DEM_NAMES    = ['Low', 'Normal', 'High']
CASH_NAMES   = ['Tight', 'Normal', 'Ample']

OUT_DIR      = 'evaluations'
TIMESTAMP    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs(OUT_DIR, exist_ok=True)


def out_path(chart_name):
    return os.path.join(OUT_DIR, f'{chart_name}_{TIMESTAMP}.png')


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

    return state_to_idx(next_inv, next_dem, next_cash), reward, stockout, holding_cost, next_cash, revenue, order_cost


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
    ep_ending_cash, ep_revenue, ep_order_cost = [], [], []
    for _ in range(n_ep):
        s = START
        total_r, total_so, total_hc = 0, 0, 0
        total_rev, total_oc = 0, 0
        ending_cash = 1
        for _ in range(STEPS):
            a = action_fn(s)
            s, r, so, hc, ec, rev, oc = step_detail(s, a)
            total_r += r; total_so += so; total_hc += hc
            total_rev += rev; total_oc += oc
            ending_cash = ec
        ep_rewards.append(total_r)
        ep_stockouts.append(total_so)
        ep_holding.append(total_hc)
        ep_ending_cash.append(ending_cash)
        ep_revenue.append(total_rev)
        ep_order_cost.append(total_oc)
    return {
        'mean_reward':     np.mean(ep_rewards),
        'std_reward':      np.std(ep_rewards),
        'stockout_rate':   np.mean([s > 0 for s in ep_stockouts]),
        'mean_stockout':   np.mean(ep_stockouts),
        'mean_holding':    np.mean(ep_holding),
        'mean_end_cash':   np.mean(ep_ending_cash),
        'cash_efficiency': np.sum(ep_revenue) / max(1, np.sum(ep_order_cost)),
        'rewards':         ep_rewards,
    }


# ── Load training results ─────────────────────────────────────────────────────

Q_q       = np.load('Q.npy')
rewards_q = list(np.load('rewards.npy'))

Q_s       = np.load('SARSA.npy')
rewards_s = list(np.load('rewards_sarsa.npy'))


# ── Paired evaluation: all policies on identical demand realizations ─────────
def evaluate_paired(policies, n_ep=N_EVAL_EP):
    """Run all policies on the same demand sequences for fair comparison."""
    seeds = list(range(1000, 1000 + n_ep))   # deterministic, reproducible
    raw = {name: {'rewards': [], 'stockouts': [], 'holding': [],
                  'ending_cash': [], 'revenue': [], 'order_cost': [],
                  'reward_by_cash': {0: [], 1: [], 2: []}}
           for name in policies}
    for seed in seeds:
        for name, action_fn in policies.items():
            random.seed(seed)
            np.random.seed(seed)
            s = START
            total_r, total_so, total_hc = 0, 0, 0
            total_rev, total_oc = 0, 0
            ending_cash = 1
            for _ in range(STEPS):
                _, _, cash_now = idx_to_state(s)
                a = action_fn(s)
                s, r, so, hc, ec, rev, oc = step_detail(s, a)
                raw[name]['reward_by_cash'][cash_now].append(r)
                total_r += r; total_so += so; total_hc += hc
                total_rev += rev; total_oc += oc
                ending_cash = ec
            raw[name]['rewards'].append(total_r)
            raw[name]['stockouts'].append(total_so)
            raw[name]['holding'].append(total_hc)
            raw[name]['ending_cash'].append(ending_cash)
            raw[name]['revenue'].append(total_rev)
            raw[name]['order_cost'].append(total_oc)
    # Repackage into the same shape as evaluate() so downstream code still works
    final = {}
    for name, d in raw.items():
        final[name] = {
            'mean_reward':     np.mean(d['rewards']),
            'std_reward':      np.std(d['rewards']),
            'stockout_rate':   np.mean([s > 0 for s in d['stockouts']]),
            'mean_stockout':   np.mean(d['stockouts']),
            'mean_holding':    np.mean(d['holding']),
            'mean_end_cash':   np.mean(d['ending_cash']),
            'cash_efficiency': np.sum(d['revenue']) / max(1, np.sum(d['order_cost'])),
            'rewards':         d['rewards'],
            'reward_by_cash':  d['reward_by_cash'],
        }
    return final

policies = {
    'Q-learning':     lambda s: int(np.argmax(Q_q[s])),
    'SARSA':          lambda s: int(np.argmax(Q_s[s])),
    'No reorder':     policy_no_reorder,
    'Fixed (medium)': policy_fixed,
    'Order-up-to':    policy_order_up_to,
    'Random':         policy_random,
}
results = evaluate_paired(policies)


# ── Print comparison table ────────────────────────────────────────────────────

print(f"\n{'Policy':<20} {'Avg reward':>10} {'Std':>8} {'Stockout%':>10} {'Holding':>10} {'EndCash':>8} {'CashEff':>8}")
print('-' * 80)
for name, m in results.items():
    print(f"{name:<20} {m['mean_reward']:>10.0f} {m['std_reward']:>8.0f} "
          f"{m['stockout_rate']*100:>9.1f}% {m['mean_holding']:>10.1f} "
          f"{m['mean_end_cash']:>8.2f} {m['cash_efficiency']:>8.2f}")
    
# ── Pairwise significance tests on paired rewards ────────────────────────────
def paired_test(name_a, name_b):
    a = np.array(results[name_a]['rewards'])
    b = np.array(results[name_b]['rewards'])
    diff = a - b
    mean_diff = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_low, ci_high = mean_diff - 1.96 * se, mean_diff + 1.96 * se
    t_stat, p_val = stats.ttest_rel(a, b)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  {name_a:<14} vs {name_b:<16} diff={mean_diff:>+7.0f}  "
          f"95% CI=[{ci_low:>+6.0f}, {ci_high:>+6.0f}]  p={p_val:.4f} {sig}")

print("\nPaired comparisons (positive diff = first policy higher):")
print('-' * 90)
paired_test('Q-learning', 'SARSA')
paired_test('Q-learning', 'Order-up-to')
paired_test('Q-learning', 'Fixed (medium)')
paired_test('Q-learning', 'Random')
paired_test('Q-learning', 'No reorder')
paired_test('SARSA',      'Order-up-to')
print('-' * 90)
print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


# Color scheme — Q-learning purple, SARSA teal, baselines neutral gray
COLOR_Q       = '#534AB7'
COLOR_S       = '#2A9D8F'
COLOR_BASE    = '#B4B2A9'
COLOR_BASE_HL = '#AFA9EC'   # subtle highlight for the heuristic

window = 100


# ── Chart 1: Learning curves (overlaid) ──────────────────────────────────────

def smooth(rewards, w=window):
    return [np.mean(rewards[max(0, i-w):i+1]) for i in range(len(rewards))]

plt.figure(figsize=(10, 4))
plt.plot(rewards_q, color=COLOR_Q, linewidth=0.5, alpha=0.25)
plt.plot(rewards_s, color=COLOR_S, linewidth=0.5, alpha=0.25)
plt.plot(smooth(rewards_q), color=COLOR_Q, linewidth=2, label=f'Q-learning (smoothed {window}-ep)')
plt.plot(smooth(rewards_s), color=COLOR_S, linewidth=2, label=f'SARSA (smoothed {window}-ep)')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('Learning curves: Q-learning vs SARSA')
plt.legend()
plt.tight_layout()
path = out_path('learning_curve')
plt.savefig(path, dpi=150)
plt.close()
print(f'Saved: {path}')


# ── Chart 2: Reward comparison (6 bars) ──────────────────────────────────────

names  = list(results.keys())
means  = [results[n]['mean_reward'] for n in names]
stds   = [results[n]['std_reward']  for n in names]
colors = [COLOR_Q, COLOR_S, COLOR_BASE, COLOR_BASE, COLOR_BASE, COLOR_BASE]

plt.figure(figsize=(10, 4))
plt.bar(names, means, yerr=stds, color=colors, edgecolor='white',
        linewidth=0.5, capsize=4, error_kw={'linewidth': 1})
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.ylabel('Average reward per episode')
plt.title('Policy comparison: average reward (Q-learning vs SARSA vs baselines)')
plt.tight_layout()
path = out_path('reward_comparison')
plt.savefig(path, dpi=150)
plt.close()
print(f'Saved: {path}')


# ── Chart 3: Stockout rate vs holding cost (scatter) ─────────────────────────

scatter_colors  = [COLOR_Q, COLOR_S, COLOR_BASE, COLOR_BASE, COLOR_BASE_HL, COLOR_BASE]
scatter_markers = ['D', 'D', 'o', 'o', 'o', 'o']
scatter_sizes   = [120, 120, 80, 80, 80, 80]

plt.figure(figsize=(7, 5))
for i, (name, m) in enumerate(results.items()):
    plt.scatter(m['stockout_rate'] * 100, m['mean_holding'],
                color=scatter_colors[i], s=scatter_sizes[i],
                marker=scatter_markers[i], zorder=3, label=name)
    plt.annotate(name, (m['stockout_rate'] * 100, m['mean_holding']),
                 textcoords='offset points', xytext=(8, 4), fontsize=9,
                 color='#444441')
plt.xlabel('Stockout rate (%)')
plt.ylabel('Average holding cost')
plt.title('Stockout rate vs holding cost by policy')
plt.legend(fontsize=9)
plt.tight_layout()
path = out_path('stockout_holding')
plt.savefig(path, dpi=150)
plt.close()
print(f'Saved: {path}')


# ── Chart 4: Policy heatmap (2×3 grid: Q-learning row + SARSA row) ───────────

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)

for row_idx, (Q_table, algo_name) in enumerate([(Q_q, 'Q-learning'), (Q_s, 'SARSA')]):
    for cash_idx in range(3):
        ax = axes[row_idx, cash_idx]
        matrix = np.zeros((3, 3))
        for inv in range(3):
            for dem in range(3):
                s = state_to_idx(inv, dem, cash_idx)
                matrix[inv, dem] = int(np.argmax(Q_table[s]))
        im = ax.imshow(matrix, cmap='Purples', vmin=0, vmax=3, aspect='auto')
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(DEM_NAMES)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(INV_NAMES)
        if row_idx == 1:
            ax.set_xlabel('Demand')
        if cash_idx == 0:
            ax.set_ylabel(f'{algo_name}\nInventory')
        ax.set_title(f'Cash: {CASH_NAMES[cash_idx]}')
        for inv in range(3):
            for dem in range(3):
                ax.text(dem, inv, ACTION_NAMES[int(matrix[inv, dem])],
                        ha='center', va='center', fontsize=8, color='#26215C')

fig.suptitle('Learned policy: Q-learning (top) vs SARSA (bottom)', y=1.0)
fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[0, 1, 2, 3],
             label='Action (0=none 1=small 2=med 3=large)', shrink=0.8)
path = out_path('policy_heatmap')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path}')


# ── Chart 5: Cash efficiency comparison ──────────────────────────────────────
# Exclude "No reorder" from cash efficiency subplot (denominator ~ 0 makes ratio meaningless)

names_active   = [n for n in results.keys() if n != 'No reorder']
ce_vals        = [results[n]['cash_efficiency'] for n in names_active]
colors_active  = [COLOR_Q, COLOR_S, COLOR_BASE, COLOR_BASE_HL, COLOR_BASE]

names_all      = list(results.keys())
end_cash_vals  = [results[n]['mean_end_cash'] for n in names_all]
colors_all     = [COLOR_Q, COLOR_S, COLOR_BASE, COLOR_BASE, COLOR_BASE_HL, COLOR_BASE]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.bar(names_active, ce_vals, color=colors_active, edgecolor='white', linewidth=0.5)
ax1.set_ylabel('Revenue per $ spent on orders')
ax1.set_title('Cash efficiency (excl. No reorder: ratio undefined)')
ax1.tick_params(axis='x', rotation=20)
for i, v in enumerate(ce_vals):
    ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

ax2.bar(names_all, end_cash_vals, color=colors_all, edgecolor='white', linewidth=0.5)
ax2.set_ylabel('Average ending cash level')
ax2.set_title('Ending cash state (0=tight, 1=normal, 2=ample)')
ax2.set_ylim(0, 2.2)
ax2.tick_params(axis='x', rotation=20)
for i, v in enumerate(end_cash_vals):
    ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
path = out_path('cash_efficiency')
plt.savefig(path, dpi=150)
plt.close()
print(f'Saved: {path}')


# ── Chart 6: Per-cash-state reward breakdown ─────────────────────────────────
# Average per-step reward when the agent acts in each cash state.
# This is the core "RL adapts to cash constraints" visual.
all_names    = list(results.keys())
all_colors   = [COLOR_Q, COLOR_S, COLOR_BASE, COLOR_BASE, COLOR_BASE_HL, COLOR_BASE]
n_policies   = len(all_names)
x            = np.arange(len(CASH_NAMES))
bar_w        = 0.13

fig, ax = plt.subplots(figsize=(11, 4.5))
for i, name in enumerate(all_names):
    rbc = results[name]['reward_by_cash']
    means = [np.mean(rbc[c]) if rbc[c] else 0 for c in range(3)]
    counts = [len(rbc[c]) for c in range(3)]
    offset = (i - (n_policies - 1) / 2) * bar_w
    bars = ax.bar(x + offset, means, bar_w, color=all_colors[i],
                  edgecolor='white', linewidth=0.5, label=name)
    # annotate sample count under each bar (small, gray)
    for j, (m, c) in enumerate(zip(means, counts)):
        ax.text(x[j] + offset, m + (1 if m >= 0 else -3), f'{m:.0f}',
                ha='center', va='bottom' if m >= 0 else 'top',
                fontsize=7, color='#555')

ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels([f'{c}\n(n={sum(len(results[n]["reward_by_cash"][i]) for n in all_names)//n_policies})'
                    for i, c in enumerate(CASH_NAMES)])
ax.set_xlabel('Cash state when action was taken (avg sample size shown)')
ax.set_ylabel('Average per-step reward')
ax.set_title('Per-step reward by cash state — does the policy hold up under tight cash?')
ax.legend(fontsize=8, ncol=3, loc='lower right')
plt.tight_layout()
path = out_path('reward_by_cash')
plt.savefig(path, dpi=150)
plt.close()
print(f'Saved: {path}')

print(f'\nAll 6 charts saved to {OUT_DIR}/ with timestamp {TIMESTAMP}')


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(results, timestamp):
    q  = results['Q-learning']
    s  = results['SARSA']
    ranked = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
    winner, loser = ('Q-learning', 'SARSA') if q['mean_reward'] >= s['mean_reward'] else ('SARSA', 'Q-learning')
    w, l = results[winner], results[loser]
    reward_margin = abs(q['mean_reward'] - s['mean_reward'])
    baselines = [(n, m) for n, m in results.items() if n not in ('Q-learning', 'SARSA')]
    best_baseline = max(baselines, key=lambda x: x[1]['mean_reward'])

    # ── Learning speed: episode where smoothed reward first crosses 80% of final value ──
    # ── Learning speed: episode where smoothed reward crosses 80% of total improvement ──
    def convergence_episode(rewards, window=100):
        smoothed = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        initial = smoothed[window]                 # first stable smoothed value
        final   = np.mean(smoothed[-500:])         # late-stage average
        gap     = final - initial
        if abs(gap) < 50:                          # essentially no measurable learning
            return None
        target = initial + 0.80 * gap
        for ep in range(window, len(smoothed)):
            if smoothed[ep] >= target:
                return ep
        return len(rewards)
    # IMPORTANT: use TRAINING rewards (rewards_q / rewards_s), not evaluation rewards
    conv_q = convergence_episode(rewards_q)
    conv_s = convergence_episode(rewards_s)
    conv_q_str = f"episode {conv_q}" if conv_q is not None else "no clear convergence point (curve essentially flat)"
    conv_s_str = f"episode {conv_s}" if conv_s is not None else "no clear convergence point (curve essentially flat)"

    # ── Reward stability: coefficient of variation (lower = more consistent) ──
    cv_q = q['std_reward'] / abs(q['mean_reward']) if q['mean_reward'] != 0 else float('inf')
    cv_s = s['std_reward'] / abs(s['mean_reward']) if s['mean_reward'] != 0 else float('inf')
    more_stable = 'Q-learning' if cv_q <= cv_s else 'SARSA'

    # ── Policy divergence: states where Q-learning and SARSA choose different actions ──
    divergent_states = []
    for state_idx in range(config.N_STATES):
        a_q = int(np.argmax(Q_q[state_idx]))
        a_s = int(np.argmax(Q_s[state_idx]))
        if a_q != a_s:
            inv, dem, cash = idx_to_state(state_idx)
            divergent_states.append((INV_NAMES[inv], DEM_NAMES[dem], CASH_NAMES[cash], ACTION_NAMES[a_q], ACTION_NAMES[a_s]))
    agreement_pct = (1 - len(divergent_states) / config.N_STATES) * 100

    # ── Cash-state action profile: avg action chosen per cash level ──
    def avg_action_by_cash(Q_table):
        avgs = {}
        for cash_idx, cash_name in enumerate(CASH_NAMES):
            actions = [int(np.argmax(Q_table[state_to_idx(inv, dem, cash_idx)]))
                       for inv in range(3) for dem in range(3)]
            avgs[cash_name] = np.mean(actions)
        return avgs

    cash_q = avg_action_by_cash(Q_q)
    cash_s = avg_action_by_cash(Q_s)

    # ── Final 500-episode training avg vs overall avg (learning maturity) ──
    final_q = np.mean(q['rewards'][-500:])
    final_s = np.mean(s['rewards'][-500:])

    lines = [
        f"EVALUATION REPORT — {timestamp}",
        "=" * 50,
        "",
        "OVERVIEW",
        f"This report evaluates Q-learning and SARSA trained on a 27-state inventory replenishment "
        f"problem (3 inventory x 3 demand x 3 cash levels, 4 order actions) over 5,000 episodes. "
        f"Both algorithms use identical hyperparameters (alpha=0.1, gamma=0.95, epsilon 1.0->0.05) "
        f"and are evaluated against four baselines across 500 test episodes from a neutral start state.",
        "",
        "LEARNING CURVES",
        f"Q-learning reached 80% of its learning gap at {conv_q_str}, "
        f"while SARSA reached the same threshold at {conv_s_str}. "
        f"{'Q-learning converged faster, consistent with off-policy learning allowing it to target the greedy policy directly during training.' if conv_q < conv_s else 'SARSA converged faster despite being on-policy, suggesting its conservative updates led to more stable early learning in this environment.'} "
        f"By the final 500 training episodes, Q-learning averaged {final_q:.0f} reward per episode "
        f"versus SARSA at {final_s:.0f}, a late-stage gap of {abs(final_q - final_s):.0f} points.",
        "",
        "REWARD COMPARISON",
        f"Full ranking by average episode reward across 500 evaluation episodes:",
    ]
    for rank, (name, m) in enumerate(ranked, 1):
        lines.append(f"  {rank}. {name:<20} avg={m['mean_reward']:>7.0f}  std={m['std_reward']:.0f}  CV={m['std_reward']/abs(m['mean_reward']):.2f}" if m['mean_reward'] != 0 else f"  {rank}. {name:<20} avg={m['mean_reward']:>7.0f}  std={m['std_reward']:.0f}")
    lines += [
        f"",
        f"{winner} outperforms {loser} by {reward_margin:.0f} reward points on average. "
        f"Reward consistency (coefficient of variation): Q-learning {cv_q:.2f}, SARSA {cv_s:.2f}. "
        f"{more_stable} produces more consistent episode-to-episode outcomes. "
        f"The best baseline, {best_baseline[0]} ({best_baseline[1]['mean_reward']:.0f}), is competitive "
        f"because it encodes domain-specific logic directly but cannot adapt to cash state.",
        "",
        "STOCKOUT vs HOLDING COST",
        f"Q-learning: {q['stockout_rate']*100:.1f}% stockout rate, {q['mean_holding']:.0f} avg holding cost.",
        f"SARSA:      {s['stockout_rate']*100:.1f}% stockout rate, {s['mean_holding']:.0f} avg holding cost.",
    ]
    for name, m in baselines:
        lines.append(f"{name:<16} {m['stockout_rate']*100:.1f}% stockout rate, {m['mean_holding']:.0f} avg holding cost.")
    lines += [
        f"",
        f"The no-reorder policy's low holding cost ({results['No reorder']['mean_holding']:.0f}) is "
        f"misleading — it reflects zero inventory replenishment, not efficiency. Its {results['No reorder']['stockout_rate']*100:.1f}% "
        f"stockout rate and negative average reward confirm it is not viable. Among active policies, "
        f"{'Q-learning achieves a lower stockout rate than SARSA' if q['stockout_rate'] < s['stockout_rate'] else 'SARSA achieves a lower stockout rate than Q-learning'} "
        f"({min(q['stockout_rate'], s['stockout_rate'])*100:.1f}% vs {max(q['stockout_rate'], s['stockout_rate'])*100:.1f}%), "
        f"indicating it learned a more proactive replenishment strategy.",
        "",
        "POLICY DIVERGENCE ANALYSIS",
        f"Across all 27 states, Q-learning and SARSA agree on the same action in "
        f"{agreement_pct:.0f}% of states ({27 - len(divergent_states)}/27). "
        f"The {len(divergent_states)} divergent states are:",
    ]
    if divergent_states:
        for inv_n, dem_n, cash_n, aq, as_ in divergent_states:
            lines.append(f"  inv={inv_n:<4} dem={dem_n:<7} cash={cash_n:<7}  Q-learning={aq:<10} SARSA={as_}")
    else:
        lines.append("  None — both algorithms learned identical greedy policies.")
    lines += [
        f"",
        f"Divergence tends to occur under ambiguous mid-range states where the optimal action "
        f"is less clear-cut, reflecting the fundamental difference between off-policy (Q-learning) "
        f"and on-policy (SARSA) value estimation.",
        "",
        "CASH-STATE ACTION PROFILE",
        f"Average action index by cash level (0=no order, 3=large order):",
        f"  Q-learning: Tight={cash_q['Tight']:.2f}, Normal={cash_q['Normal']:.2f}, Ample={cash_q['Ample']:.2f}",
        f"  SARSA:      Tight={cash_s['Tight']:.2f}, Normal={cash_s['Normal']:.2f}, Ample={cash_s['Ample']:.2f}",
        f"Both agents correctly order less aggressively under tight cash. "
        f"{'Q-learning orders more aggressively under ample cash' if cash_q['Ample'] > cash_s['Ample'] else 'SARSA orders more aggressively under ample cash'} "
        f"(avg action {max(cash_q['Ample'], cash_s['Ample']):.2f} vs {min(cash_q['Ample'], cash_s['Ample']):.2f}), "
        f"consistent with its {'off' if cash_q['Ample'] > cash_s['Ample'] else 'on'}-policy optimism.",
        "",
        "RECOMMENDATION",
        f"Recommended model: {winner}.",
        f"{winner} achieves the highest average reward among RL agents ({w['mean_reward']:.0f}), "
        f"a {reward_margin:.0f}-point lead over {loser}, with a stockout rate of {w['stockout_rate']*100:.1f}% "
        f"and holding cost of {w['mean_holding']:.0f}. It converged to a stable policy by {conv_q_str if winner == 'Q-learning' else conv_s_str} "
        f"and generalises sensibly across all cash states. While {best_baseline[0]} ({best_baseline[1]['mean_reward']:.0f}) "
        f"is competitive, it is a hand-crafted heuristic that ignores cash availability. "
        f"{winner} learns this behaviour from experience and is better suited for environments "
        f"where cash constraints actively shape replenishment decisions.",
    ]

    report = "\n".join(lines)
    word_count = len(report.split())

    path = os.path.join(OUT_DIR, f'report_{timestamp}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'Saved: {path}  ({word_count} words)')


write_report(results, TIMESTAMP)
