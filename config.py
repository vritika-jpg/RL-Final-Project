# config.py includes all hyperparameters and variable names that should be used in different algorithms
# Shared configuration for all team members.
# Only edit this file to change parameters — never hardcode values elsewhere.

# ── Training hyperparameters ──────────────────────────────────────────────────
EPISODES      = 5000
STEPS_PER_EP  = 50
ALPHA         = 0.1     # learning rate
GAMMA         = 0.95    # discount factor
EPSILON_START = 1.0     # exploration rate at start
EPSILON_END   = 0.05    # minimum exploration rate
SEED          = 42      # random seed for reproducibility

# ── State space ───────────────────────────────────────────────────────────────
# inventory:  0=low,    1=medium, 2=high
# demand:     0=low,    1=normal, 2=high
# cash:       0=tight,  1=normal, 2=ample
N_INV     = 3
N_DEM     = 3
N_CASH    = 3
N_STATES  = N_INV * N_DEM * N_CASH   # 27

# ── Action space ──────────────────────────────────────────────────────────────
# 0=no order, 1=small, 2=medium, 3=large
N_ACTIONS  = 4
ORDER_QTY  = [0, 5, 10, 20]   # units ordered per action
ORDER_COST = [0, 8, 14, 25]   # cost to place each order

# ── Reward parameters ─────────────────────────────────────────────────────────
PRICE        = 10    # revenue per unit sold
HOLDING_COST = 2     # cost per leftover unit (only charged when leftover > 5)
STOCKOUT_PEN = 15    # penalty per unit of unmet demand
EXCESS_PEN   = 3     # penalty when inventory is high after a large order

# ── Demand distribution (Poisson mean by demand state) ───────────────────────
DEMAND_MEAN = {0: 3, 1: 8, 2: 15}

# ── Inventory units by level (used in reward calculation) ────────────────────
INV_UNITS = {0: 3, 1: 10, 2: 20}
