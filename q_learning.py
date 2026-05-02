# q_learning.py
# Q-learning implementation — your file.
# Off-policy TD control: updates Q using the greedy next action (max Q),
# regardless of what action the agent actually takes next.

import random
import numpy as np
import config
from simulator import step, initial_state


# ── Action selection ──────────────────────────────────────────────────────────

def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using epsilon-greedy policy.
    With probability epsilon: explore (random action).
    Otherwise:               exploit (greedy, highest Q value).
    """
    if random.random() < epsilon:
        return random.randint(0, config.N_ACTIONS - 1)
    return int(np.argmax(Q[state]))


# ── Training ──────────────────────────────────────────────────────────────────

def train_q_learning():
    """
    Train the agent using Q-learning.
    All hyperparameters are read from config.py.

    Returns
    -------
    Q              : np.ndarray, shape (27, 4)   Trained Q-table
    rewards_per_ep : list of float               Total reward per episode
    """
    # Fix random seeds for reproducibility — must match sarsa.py
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    # Initialise Q-table to zero
    Q = np.zeros((config.N_STATES, config.N_ACTIONS))
    rewards_per_ep = []

    for ep in range(config.EPISODES):

        # Linear epsilon decay
        eps = max(
            config.EPSILON_END,
            config.EPSILON_START - (config.EPSILON_START - config.EPSILON_END)
            * ep / config.EPISODES
        )

        s = initial_state()
        total_reward = 0

        for _ in range(config.STEPS_PER_EP):
            # 1. Select action under current epsilon-greedy policy
            a = epsilon_greedy(Q, s, eps)

            # 2. Observe next state and reward from environment
            s2, r = step(s, a)

            # 3. Q-learning update (off-policy)
            #    Use max Q(s', .) — the best possible next action —
            #    NOT the action actually taken next.
            best_next = np.max(Q[s2])
            Q[s, a] += config.ALPHA * (r + config.GAMMA * best_next - Q[s, a])

            s = s2
            total_reward += r

        rewards_per_ep.append(total_reward)

    return Q, rewards_per_ep
