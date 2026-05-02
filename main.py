# main.py
# Entry point — trains Q-learning AND SARSA, saves results for the evaluation team.

import numpy as np
from q_learning import train_q_learning
from sarsa import train_sarsa

if __name__ == "__main__":
    # ── Q-learning ────────────────────────────────────────────────────────────
    print("Training Q-learning...")
    Q, rewards = train_q_learning()
    np.save("Q.npy",       Q)
    np.save("rewards.npy", rewards)
    print("Done. Saved: Q.npy, rewards.npy")

    # ── SARSA ─────────────────────────────────────────────────────────────────
    print("\nTraining SARSA...")
    Q_sarsa, rewards_sarsa = train_sarsa()
    np.save("SARSA.npy",         Q_sarsa)
    np.save("rewards_sarsa.npy", rewards_sarsa)
    print("Done. Saved: SARSA.npy, rewards_sarsa.npy")
