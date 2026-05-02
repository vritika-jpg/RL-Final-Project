# main.py
# Entry point — trains Q-learning and saves results for the evaluation team.

import numpy as np
from q_learning import train_q_learning

if __name__ == "__main__":
    print("Training Q-learning...")
    Q, rewards = train_q_learning()

    # Save outputs for the evaluation team
    np.save("Q.npy",       Q)
    np.save("rewards.npy", rewards)

    print("Done. Saved: Q.npy, rewards.npy")
