import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# ==========================================
# Q.1 a) Probability distribution for two dice
# ==========================================
# Using convolution to find the sum of two uniform distributions
dice_faces = np.ones(6) / 6
prob_dice = np.convolve(dice_faces, dice_faces)
states_dice = np.arange(2, 13)

print("--- Q.1 a) Pair of Fair Dice Distribution ---")
for s, p in zip(states_dice, prob_dice):
    print(f"Sum = {s:2d} | Probability = {p:.4f}")

# ==========================================
# Q.1 b) & c) 10 Coin Tosses Random Walk
# ==========================================
n_tosses = 10
k_heads = np.arange(0, n_tosses + 1)
positions_10 = 2 * k_heads - n_tosses
prob_10 = binom.pmf(k_heads, n_tosses, 0.5)

print("\n--- Q.1 b) 10 Coin Tosses Distribution ---")
for pos, p in zip(positions_10, prob_10):
    print(f"Position {pos:3d} | Probability = {p:.4f}")

print("\n--- Q.1 c) Specific Probabilities (10 Tosses) ---")
print(f"i)   P(Position = 0)  = {binom.pmf(5, n_tosses, 0.5):.4f}")
print(f"ii)  P(Position = 5)  = 0.0000 (Odd position is mathematically impossible)")
print(f"iii) P(Position = 10) = {binom.pmf(10, n_tosses, 0.5):.4f}")

# ==========================================
# Q.2 Drone Random Walk (8 seconds)
# ==========================================
n_sec = 8
k_forward = np.arange(0, n_sec + 1)
positions_8 = 2 * k_forward - n_sec
prob_8 = binom.pmf(k_forward, n_sec, 0.5)

print("\n--- Q.2 a) Drone Position Distribution (8 sec) ---")
for pos, p in zip(positions_8, prob_8):
    print(f"Position {pos:2d} | Probability = {p:.4f}")

print("\n--- Q.2 c) Specific Probabilities (Drone) ---")
# Using k = (Position + n) / 2 to find required successful steps
print(f"i)   P(Position = 0) = {binom.pmf(4, n_sec, 0.5):.4f}")
print(f"ii)  P(Position = 4) = {binom.pmf(6, n_sec, 0.5):.4f}")
print(f"iii) P(Position = 8) = {binom.pmf(8, n_sec, 0.5):.4f}")

# Q.2 b) Plotting the probability distribution
plt.figure(figsize=(8, 5))
plt.stem(positions_8, prob_8, basefmt="b-")
plt.xlabel("Drone Position")
plt.ylabel("Probability")
plt.title("Probability Distribution of Drone Position After 8 Seconds")
plt.xticks(positions_8)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
