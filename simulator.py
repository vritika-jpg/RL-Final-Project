# simulator.py
# Environment setup — owned by your partner.
# Provides: state_to_idx, idx_to_state, step, initial_state

import random
import numpy as np
import config


# ── State encoding ────────────────────────────────────────────────────────────

def state_to_idx(inv, dem, cash):
    """Encode (inv, dem, cash) into a single integer index 0–26."""
    return inv * 9 + dem * 3 + cash

def idx_to_state(idx):
    """Decode a state index back into (inv, dem, cash)."""
    inv  = idx // 9
    dem  = (idx % 9) // 3
    cash = idx % 3
    return inv, dem, cash


# ── Demand sampling ───────────────────────────────────────────────────────────

def sample_demand(dem_state):
    """
    Sample actual demand units this period.
    Uses a Poisson distribution parameterised by config.DEMAND_MEAN.
    """
    mean = config.DEMAND_MEAN[dem_state]
    return max(0, int(np.random.poisson(mean)))


# ── Environment step ──────────────────────────────────────────────────────────

def step(state_idx, action):
    """
    Simulate one period of the inventory problem.

    Parameters
    ----------
    state_idx : int   Current state index (0–26)
    action    : int   Replenishment action (0–3)

    Returns
    -------
    next_state_idx : int    Next state index
    reward         : float  Reward for this period
    """
    inv, dem, cash = idx_to_state(state_idx)

    # ── Cash constraint: limit order size when cash is tight ─────────────────
    if cash == 0 and action >= 2:
        action = 1                          # downgrade to small order
    if cash == 0 and action == 1 and random.random() < 0.3:
        action = 0                          # occasionally can't afford even small

    # ── Receive order: inventory level moves up one step if anything ordered ─
    qty_ordered  = config.ORDER_QTY[action]
    inv_after    = min(inv + (1 if qty_ordered > 0 else 0), 2)

    # ── Realise demand ────────────────────────────────────────────────────────
    actual_demand = sample_demand(dem)
    inv_units     = config.INV_UNITS[inv_after]
    units_sold    = min(inv_units, actual_demand)
    stockout      = max(0, actual_demand - units_sold)
    leftover      = max(0, inv_units - actual_demand)

    # ── Compute reward ────────────────────────────────────────────────────────
    revenue      = units_sold * config.PRICE
    order_cost   = config.ORDER_COST[action]
    holding_cost = leftover * config.HOLDING_COST if leftover > 5 else 0
    stockout_pen = stockout * config.STOCKOUT_PEN
    excess_pen   = config.EXCESS_PEN if (inv_after == 2 and qty_ordered > 0) else 0

    reward = revenue - order_cost - holding_cost - stockout_pen - excess_pen

    # ── Next inventory level ──────────────────────────────────────────────────
    if stockout > 3:
        next_inv = max(0, inv_after - 1)    # stockout → inventory perceived lower
    elif leftover > 8:
        next_inv = min(2, inv_after + 1)    # large surplus → stay high
    else:
        next_inv = inv_after

    # ── Next demand state (slow random walk) ─────────────────────────────────
    r = random.random()
    if r < 0.15:
        next_dem = max(0, dem - 1)
    elif r < 0.30:
        next_dem = min(2, dem + 1)
    else:
        next_dem = dem

    # ── Next cash state ───────────────────────────────────────────────────────
    net = revenue - order_cost
    if net > 20:
        next_cash = min(2, cash + 1)
    elif net < 0:
        next_cash = max(0, cash - 1)
    else:
        next_cash = cash

    next_state_idx = state_to_idx(next_inv, next_dem, next_cash)
    return next_state_idx, reward


# ── Initial state ─────────────────────────────────────────────────────────────

def initial_state():
    """
    Return a random starting state for training.
    Using random start ensures all 27 states are visited during training.
    For evaluation, use state_to_idx(1, 1, 1) as a fixed neutral start.
    """
    return state_to_idx(
        random.randint(0, config.N_INV  - 1),
        random.randint(0, config.N_DEM  - 1),
        random.randint(0, config.N_CASH - 1)
    )
