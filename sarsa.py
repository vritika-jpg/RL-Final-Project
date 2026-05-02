# sarsa.py
# SARSA implementation — on-policy TD control.
# Updates Q using the action actually chosen for the next state under the same
# epsilon-greedy policy (NOT the greedy max as in Q-learning).

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

def train_sarsa():
    """
    Train the agent using SARSA (on-policy TD control).
    All hyperparameters are read from config.py.

    Returns
    -------
    Q              : np.ndarray, shape (27, 4)   Trained Q-table
    rewards_per_ep : list of float               Total reward per episode
    """
    # Fix random seeds for reproducibility — must match q_learning.py
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
        a = epsilon_greedy(Q, s, eps)
        total_reward = 0

        for _ in range(config.STEPS_PER_EP):
            # 1. Observe next state and reward from environment
            s2, r = step(s, a)

            # 2. Choose next action under the SAME epsilon-greedy policy
            a2 = epsilon_greedy(Q, s2, eps)

            # 3. SARSA update (on-policy)
            #    Use Q(s', a') — the action we will actually take next —
            #    NOT max over actions (that would be Q-learning).
            Q[s, a] += config.ALPHA * (r + config.GAMMA * Q[s2, a2] - Q[s, a])

            s = s2
            a = a2
            total_reward += r

        rewards_per_ep.append(total_reward)

    return Q, rewards_per_ep
